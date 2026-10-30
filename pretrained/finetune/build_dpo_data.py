#!/usr/bin/env python3
"""DPO pairs for the refinement role: "keep the expectation, don't copy the response" (D-54).

Source B (our runs): every refinement round where the guard had to step in. The prompt is the
exact refinement prompt the model saw; "rejected" is the model's raw answer (it changed a locked
expectation or copied an observed value); "chosen" is the guarded plan the pipeline went on with.
Only guarded variants (strengthen_only, combined) record the guarded plan.

    .venv/bin/python finetune/build_dpo_data.py [--groups sweep finals dpo_data smoke]

Output: finetune/data/dpo_fold_<held_out>.jsonl  ({"prompt", "chosen", "rejected", ...}) and a count
summary. Leave-one-service-out: the fold for service S contains no pair from S.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.paths import RESULTS_ROOT, SERVICES  # noqa: E402

COPY_EVENTS = {"value_copy_blocked", "new_check_with_copied_value_blocked"}


def pair_kind(events):
    types = {e["type"] for e in events}
    if types & COPY_EVENTS or any(e.get("copied_value") for e in events):
        return "copied_value"
    if "locked_check_restored" in types or "field_repair_rejected_not_in_spec" in types:
        return "changed_expectation"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["sweep", "finals", "dpo_data", "smoke"])
    a = ap.parse_args()
    pairs, counts = [], Counter()
    for g in a.groups:
        for h_path in (RESULTS_ROOT / g).glob("*/*/generated_tests/*/agent_history.json"):
            h = json.loads(h_path.read_text())
            service = h.get("service")
            calls = [c for c in h.get("llm_calls", []) if str(c.get("tag", "")).startswith("refine")]
            for r in h.get("rounds", []):
                if r.get("decision") != "refine" or not r.get("refine_guarded"):
                    continue
                kind = pair_kind(r.get("refine_events") or [])
                if not kind:
                    continue
                call = next((c for c in calls if c.get("response") == r.get("refine_raw")), None)
                if call is None:
                    continue
                chosen = json.dumps(r["refine_guarded"], indent=1)
                if chosen.strip() == (r.get("refine_raw") or "").strip():
                    continue
                pairs.append({"service": service, "group": g, "source": str(h_path.relative_to(RESULTS_ROOT)),
                              "kind": kind, "prompt": call["messages"], "chosen": chosen,
                              "rejected": r["refine_raw"]})
                counts[(service, kind)] += 1
    out = ROOT / "finetune" / "data"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pairs_by_service_kind": {f"{s}/{k}": n for (s, k), n in sorted(counts.items())}}
    for held_out in SERVICES:
        fold = [p for p in pairs if p["service"] != held_out]
        with (out / f"dpo_fold_{held_out}.jsonl").open("w") as f:
            for p in fold:
                f.write(json.dumps(p) + "\n")
        summary[f"fold_{held_out}"] = len(fold)
    (out / "dpo_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
