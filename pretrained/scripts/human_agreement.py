#!/usr/bin/env python3
"""Agreement between human labels (Check Labelling artifact) and the Llama judge.

    # 1. export the labels (Claude: ArtifactData list "labels" with out_dir=gold/human_validation/export)
    # 2. .venv/bin/python scripts/human_agreement.py gold/human_validation/export/labels

Reports, per labeller and pooled:
- gold checks: share judged correct / wrong / unsure (quality of the Claude-drafted gold data)
- coverage: Cohen's kappa between human and Llama-judge labels, for gold-check coverage
  (full/partial/none) and for derived checks (match/extra/contradiction), plus raw agreement
- between two labellers (if two labelled the same keys): Cohen's kappa human-human
Writes gold/human_validation/agreement.json.
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HV = ROOT / "gold" / "human_validation"


def kappa(pairs):
    """Cohen's kappa for a list of (a, b) labels."""
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def main() -> int:
    label_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HV / "export" / "labels"
    items = {i["id"]: i for i in json.loads((HV / "items.json").read_text())}
    llm = json.loads((HV / "llm_labels.json").read_text())
    people = {}
    for f in sorted(label_dir.glob("*.json")):
        doc = json.loads(f.read_text())
        people[f.stem] = (doc.get("data") or doc).get("items", {})
    out = {"labellers": len(people), "per_labeller": {}}
    pooled_gold, pooled_cov_g, pooled_cov_d = Counter(), [], []
    for who, labs in people.items():
        gold_counts, cov_g, cov_d = Counter(), [], []
        for iid, lab in labs.items():
            it = items.get(iid)
            if not it:
                continue
            if it["task"] == "gold":
                gold_counts.update(v for k, v in lab.items() if k != "note")
            else:
                v = llm.get(iid, {})
                cov_g += [(lab[c["key"]], v.get("gold", {}).get(c["key"], "none"))
                          for c in it["checks"] if lab.get(c["key"])]
                cov_d += [(lab[d["key"]], v.get("derived", {}).get(d["key"], "extra"))
                          for d in it.get("derived", []) if lab.get(d["key"])]
        out["per_labeller"][who] = {
            "gold_checks": dict(gold_counts),
            "coverage_gold": {"n": len(cov_g), "kappa_vs_llm": kappa(cov_g),
                              "agreement": round(sum(a == b for a, b in cov_g) / len(cov_g), 3) if cov_g else None},
            "coverage_derived": {"n": len(cov_d), "kappa_vs_llm": kappa(cov_d),
                                 "agreement": round(sum(a == b for a, b in cov_d) / len(cov_d), 3) if cov_d else None}}
        pooled_gold.update(gold_counts)
        pooled_cov_g += cov_g
        pooled_cov_d += cov_d
    out["pooled"] = {"gold_checks": dict(pooled_gold),
                     "gold_checks_wrong_rate": round(pooled_gold["wrong"] / max(1, sum(pooled_gold.values())), 3),
                     "coverage_gold_kappa_vs_llm": kappa(pooled_cov_g), "n_coverage_gold": len(pooled_cov_g),
                     "coverage_derived_kappa_vs_llm": kappa(pooled_cov_d), "n_coverage_derived": len(pooled_cov_d)}
    hh = {}
    for (a, la), (b, lb) in combinations(people.items(), 2):
        pairs = [(la[i][k], lb[i][k]) for i in set(la) & set(lb) for k in la[i] if k != "note" and k in lb[i]]
        hh[f"{a}~{b}"] = {"n": len(pairs), "kappa": kappa(pairs)}
    out["human_human"] = hh
    (HV / "agreement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
