#!/usr/bin/env python3
"""Retrieval metrics without login/registration operations (the compiler adds those itself, D-25).

    .venv/bin/python scripts/retrieval_non_auth.py      # reads per_query.jsonl, writes summary_non_auth.json
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "offline" / "retrieval"
sys.path.insert(0, str(ROOT))
from agent.compiler import infer_auth  # noqa: E402


def auth_ops(service: str) -> set:
    """The login, registration and "who am I" operations the compiler uses on its own."""
    a = infer_auth(service)
    return ({f"POST {a['login_path']}", f"POST {a['register_path']}"}
            | ({f"GET {a['me_path']}"} if a.get("me_path") else set()))


def metrics(gold: set, ranked: list) -> dict:
    out = {}
    for k in (3, 5):
        top = ranked[:k]
        out[f"R@{k}"] = len(gold & set(top)) / len(gold)
        out[f"C@{k}"] = float(gold <= set(top))
    first = next((i for i, op in enumerate(ranked) if op in gold), None)
    out["MRR"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(1.0 / math.log2(i + 2) for i, op in enumerate(ranked[:5]) if op in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    out["nDCG@5"] = dcg / idcg
    return out


def main() -> int:
    acc, auth = defaultdict(list), {}
    for line in (OUT / "per_query.jsonl").open():
        r = json.loads(line)
        if r["variant"] == "full":
            continue
        if r["service"] not in auth:
            auth[r["service"]] = auth_ops(r["service"])
        gold = set(r["gold"]) - auth[r["service"]]
        if not gold:
            continue
        ranked = [op for op in r["ranked"] if op not in auth[r["service"]]]
        acc[f"{r['kind']}/{r['variant']}"].append(metrics(gold, ranked))
    summary = {k: {m: round(sum(x[m] for x in v) / len(v), 4) for m in v[0]} | {"n": len(v)}
               for k, v in sorted(acc.items())}
    (OUT / "summary_non_auth.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(f"{k:28} R@5 {v['R@5']:.3f}  C@5 {v['C@5']:.3f}  MRR {v['MRR']:.3f}  n={v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
