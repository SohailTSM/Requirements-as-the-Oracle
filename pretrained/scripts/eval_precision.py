#!/usr/bin/env python3
"""Offline evaluation of the precision layer (component metric).

For each requirement, derive checks from the VAGUE text with each variant, then compare
them to the gold atomic checks (gold/checks.json, written from the PRECISE text) with an
LLM judge (Llama 3.1 8B; validated later against human labels).

Metrics per variant:
    check_recall        gold checks fully covered / gold checks
    check_recall_soft   (full + 0.5 * partial) / gold checks
    contradiction_rate  derived checks that contradict a gold check / derived checks
    unsupported_rate    derived checks with no gold counterpart / derived checks
    n_checks            derived checks per requirement
Outputs results/offline/precision/<prompt version>/<variant>__<kind>.jsonl and summary.json; logs to W&B.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from agent.llm import LLM  # noqa: E402
from agent.precision import PROMPT_VERSION, derive_checks, support_filter  # noqa: E402
from agent.retrieval import Retriever  # noqa: E402
from common.paths import HARNESS_ROOT, RESULTS_ROOT, SERVICES  # noqa: E402

OUT = RESULTS_ROOT / "offline" / "precision" / PROMPT_VERSION   # one folder per prompt version

JUDGE_PROMPT = """You compare two lists of checks for an API test.
GOLD checks are what a correct test must verify. DERIVED checks were written by a system.

GOLD:
{gold}

DERIVED:
{derived}

For each GOLD check, decide:
- "full": a derived check verifies the same thing (same field/status/value, or an equivalent relationship)
- "partial": a derived check verifies it only loosely (e.g. "4xx" when gold says exactly 400, or checks the field exists but not its value)
- "none": no derived check verifies it
For each DERIVED check, decide:
- "match": it corresponds to some gold check
- "contradiction": it expects something different from a gold check (e.g. a different status code or value)
- "extra": it has no gold counterpart

Answer with JSON only, exactly in this form:
{{"gold": {{"G1": "full", "G2": "none"}}, "derived": {{"D1": "match", "D2": "extra"}}}}"""


def load_requirements(service):
    out = {}
    for f in (HARNESS_ROOT / "data/requirements" / service).glob("requirement_*.json"):
        d = json.loads(f.read_text())
        out[d["id"]] = d
    return dict(sorted(out.items()))


def judge(llm, gold, derived):
    if not derived:
        return {"gold": {f"G{i+1}": "none" for i in range(len(gold))}, "derived": {}}, ""
    g = "\n".join(f"G{i+1}: {c['check']}" for i, c in enumerate(gold))
    d = "\n".join(f"D{i+1}: {c['description']}" for i, c in enumerate(derived))
    raw = llm.complete(JUDGE_PROMPT.format(gold=g, derived=d), max_tokens=600, tag="precision.judge")
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        verdict = json.loads(m.group(0)) if m else {}
    except Exception:
        verdict = {}
    gv = {f"G{i+1}": str(verdict.get("gold", {}).get(f"G{i+1}", "none")).lower() for i in range(len(gold))}
    dv = {f"D{i+1}": str(verdict.get("derived", {}).get(f"D{i+1}", "extra")).lower() for i in range(len(derived))}
    return {"gold": gv, "derived": dv}, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["direct_rewrite", "catalogue", "self_consistency"])
    ap.add_argument("--kinds", nargs="+", default=["vague"])
    ap.add_argument("--retrieval", default="hybrid_rerank")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    gold_all = json.loads((PROJECT_ROOT / "gold/checks.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    llm = LLM()
    summary = {}
    for variant in args.variants:
        for kind in args.kinds:
            path = OUT / f"{variant}__{kind}.jsonl"
            done = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    done[(r["service"], r["requirement"])] = r
            with path.open("a") as f:
                for service in SERVICES:
                    retriever = Retriever(service)
                    for rid, req in load_requirements(service).items():
                        if (service, rid) in done:
                            continue
                        text = req[f"requirement_{kind}"]
                        ops = retriever.rank(text, args.retrieval)[:args.top_k]
                        base, _, extra = variant.partition("+")        # "<variant>+filter" adds the support filter
                        out = derive_checks(text, ops, base, llm)
                        if extra == "filter":
                            out["unfiltered"] = out["checks"]
                            out["checks"], out["filter"] = support_filter(text, out["checks"], llm)
                        gold = gold_all[service][str(rid)]
                        verdict, raw = judge(llm, gold, out["checks"])
                        row = {"service": service, "requirement": rid, "kind": kind, "variant": variant, "prompt_version": PROMPT_VERSION,
                               "requirement_text": text, "operations": [o.op_id for o in ops],
                               "derived": out["checks"], "gold": gold, "filter": out.get("filter"), "verdict": verdict,
                               "judge_raw": raw, "derive_raw": out["raw"]}
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        done[(service, rid)] = row
                print(f"done {variant}/{kind}", flush=True)
            rows = list(done.values())
            gold_labels = [v for r in rows for v in r["verdict"]["gold"].values()]
            der_labels = [v for r in rows for v in r["verdict"]["derived"].values()]
            n_der = max(len(der_labels), 1)
            summary[f"{variant}/{kind}"] = {
                "check_recall": round(gold_labels.count("full") / len(gold_labels), 4),
                "check_recall_soft": round((gold_labels.count("full") + 0.5 * gold_labels.count("partial")) / len(gold_labels), 4),
                "contradiction_rate": round(der_labels.count("contradiction") / n_der, 4),
                "unsupported_rate": round(der_labels.count("extra") / n_der, 4),
                "n_checks": round(len(der_labels) / len(rows), 2),
                "parse_failures": sum(1 for r in rows if not r["derived"]),
                "n": len(rows),
            }
            print(variant, kind, summary[f"{variant}/{kind}"], flush=True)
    old = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    old.update(summary)
    (OUT / "summary.json").write_text(json.dumps(old, indent=1))

    if not args.no_wandb:
        try:
            import os
            import wandb
            from common.secrets import export_env
            export_env()
            os.environ.setdefault("WANDB_DIR", str(RESULTS_ROOT))
            run = wandb.init(project="pretrained-restestbench", name=f"offline_precision_{PROMPT_VERSION}", group="offline",
                             job_type="precision_eval", reinit="finish_previous")
            run.summary.update({f"{k}/{m}": v for k, d in old.items() for m, v in d.items()})
            run.finish()
        except Exception as e:
            print(f"[wandb] logging failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
