#!/usr/bin/env python3
"""Offline retrieval evaluation (component metric for the RAG layer).

For every requirement (vague and precise) and every retrieval variant, rank the service's
operations and compare with gold/operations.json (operations the reference test calls).

Metrics: Recall@k, all-gold-retrieved@k ("complete@k"), MRR, nDCG@k (binary relevance),
and prompt size in Llama tokens for the top-5 (top-all for "full").
Writes results/offline/retrieval/{per_query.jsonl, summary.json} and logs to W&B.
"""
import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from agent.retrieval import VARIANTS, Retriever, rewrite_queries  # noqa: E402
from common.paths import HARNESS_ROOT, RESULTS_ROOT, SERVICES  # noqa: E402

KS = (1, 3, 5, 10)
OUT = RESULTS_ROOT / "offline" / "retrieval"


def ndcg(ranked: list[str], gold: set, k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, op in enumerate(ranked[:k]) if op in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def load_requirements(service: str) -> dict:
    reqs = {}
    for f in (HARNESS_ROOT / "data/requirements" / service).glob("requirement_*.json"):
        d = json.loads(f.read_text())
        reqs[d["id"]] = d
    return dict(sorted(reqs.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
    gold = json.loads((PROJECT_ROOT / "gold/operations.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    rewrites_path = OUT / "query_rewrites.json"
    rewrites = json.loads(rewrites_path.read_text()) if rewrites_path.exists() else {}
    llm = None

    rows = []
    for service in SERVICES:
        retriever = Retriever(service)
        for rid, req in load_requirements(service).items():
            g = set(gold[service][str(rid)])
            for kind in ("vague", "precise"):
                query = req[f"requirement_{kind}"]
                key = f"{service}/{rid}/{kind}"
                for variant in args.variants:
                    extra = None
                    if variant == "query_rewrite":
                        if key not in rewrites:
                            if llm is None:
                                from agent.llm import LLM
                                llm = LLM()
                            rewrites[key] = rewrite_queries(query, llm)
                            rewrites_path.write_text(json.dumps(rewrites, indent=1))
                        extra = rewrites[key]
                    ranked_ops = retriever.rank(query, variant, extra)
                    ranked = [op.op_id for op in ranked_ops]
                    first_hit = next((i for i, op in enumerate(ranked) if op in g), None)
                    k_ctx = len(ranked) if variant == "full" else 5
                    ctx = "\n\n".join(op.to_text() for op in ranked_ops[:k_ctx])
                    row = {"service": service, "requirement": rid, "kind": kind, "variant": variant,
                           "gold": sorted(g), "ranked": ranked[:10],
                           "mrr": 1 / (first_hit + 1) if first_hit is not None else 0.0,
                           "context_tokens": len(tok(ctx)["input_ids"])}
                    for k in KS:
                        hits = len(set(ranked[:k]) & g)
                        row[f"recall@{k}"] = hits / len(g)
                        row[f"complete@{k}"] = float(hits == len(g))
                        row[f"ndcg@{k}"] = ndcg(ranked, g, k)
                    rows.append(row)
        print(f"done {service}", flush=True)

    with (OUT / "per_query.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    metrics = ["mrr", "context_tokens"] + [f"{m}@{k}" for m in ("recall", "complete", "ndcg") for k in KS]
    summary = {}
    for kind in ("vague", "precise"):
        for variant in args.variants:
            sel = [r for r in rows if r["kind"] == kind and r["variant"] == variant]
            summary[f"{kind}/{variant}"] = {m: round(sum(r[m] for r in sel) / len(sel), 4) for m in metrics}
            summary[f"{kind}/{variant}"]["n"] = len(sel)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{'query/variant':28} {'R@3':>6} {'R@5':>6} {'C@5':>6} {'MRR':>6} {'nDCG@5':>7} {'tokens':>7}")
    for name, m in summary.items():
        print(f"{name:28} {m['recall@3']:6.3f} {m['recall@5']:6.3f} {m['complete@5']:6.3f} "
              f"{m['mrr']:6.3f} {m['ndcg@5']:7.3f} {m['context_tokens']:7.0f}")

    if not args.no_wandb:
        try:
            import os
            import wandb
            from common.secrets import export_env
            export_env()
            os.environ.setdefault("WANDB_DIR", str(RESULTS_ROOT))
            run = wandb.init(project="pretrained-restestbench", name="offline_retrieval", group="offline",
                             job_type="retrieval_eval", reinit="finish_previous")
            cols = ["query_kind", "variant"] + metrics
            run.log({"retrieval_summary": wandb.Table(
                columns=cols, data=[[k.split("/")[0], k.split("/")[1]] + [v[m] for m in metrics]
                                    for k, v in summary.items()])})
            run.summary.update({f"{k}/{m}": v[m] for k, v in summary.items() for m in metrics})
            run.finish()
        except Exception as e:
            print(f"[wandb] logging failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
