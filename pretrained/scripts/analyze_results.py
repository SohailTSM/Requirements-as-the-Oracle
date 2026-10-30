#!/usr/bin/env python3
"""Collect every finished benchmark job into one table.

    .venv/bin/python scripts/analyze_results.py            # prints table, writes results/summary_table.csv

Per job: bug-catching score on tests generated on the correct service (valid%) and on
buggy services (mutated%), adaptation gap, share of tests failing on the correct service,
time and tokens. For refinement jobs with round history (B2/B3 logged wrapper, our agent)
it also reports how often an expected value was changed to a value the service returned.
"""
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import RESULTS_ROOT  # noqa: E402
from common.wandb_log import summarize  # noqa: E402

ASSERT_LITERAL = re.compile(r"""==\s*(['"][^'"]{1,80}['"]|\d{3})""")


def refinement_signals(job_dir: Path) -> dict:
    """Value copying in RESTestBench-style histories: a refinement round adds an asserted literal
    that appeared in the previous run's HTTP trace."""
    rounds = copies = api_bug = 0
    for hist in job_dir.glob("*/generated_tests/*/refinement_history.json"):
        events = json.loads(hist.read_text()).get("events", [])
        last_trace, last_code = "", None
        for e in events:
            if e["type"] == "execute":
                last_trace = e.get("http_trace") or ""
            elif e["type"] == "api_bug":
                api_bug += 1
            elif e["type"] == "generate":
                if e.get("refine") and last_code is not None:
                    rounds += 1
                    before = set(ASSERT_LITERAL.findall(last_code))
                    after = set(ASSERT_LITERAL.findall(e["test_code"]))
                    new = {lit.strip("'\"") for lit in after - before}
                    if any(lit and lit in last_trace for lit in new):
                        copies += 1
                last_code = e["test_code"]
    judged = {"valid": 0, "valid_api_bug": 0, "mutant": 0, "mutant_api_bug": 0}
    for hist in job_dir.glob("*/generated_tests/*/agent_history.json"):
        h = json.loads(hist.read_text())
        on = "mutant" if "_mut_" in hist.parent.name else "valid"   # req_N = correct service, req_N_mut_k = mutant
        for r in h.get("rounds", []):
            if r.get("judge"):
                judged[on] += 1
                judged[on + "_api_bug"] += r["judge"].get("verdict") == "API_BUG"
            if r.get("decision") == "api_bug":
                api_bug += 1
            if r.get("decision") == "refine":
                rounds += 1
                if any(ev.get("type") in ("value_copy_blocked", "new_check_with_copied_value_blocked")
                       or ev.get("copied_value") for ev in r.get("refine_events", [])):
                    copies += 1
    rate = lambda a, b: round(a / b, 4) if b else None
    return {"refine_rounds": rounds, "value_copy_rounds": copies,
            "value_copy_rate": rate(copies, rounds), "api_bug_verdicts": api_bug,
            # every failure on the correct service is the test's fault, so API_BUG there is a false alarm
            "judge_false_api_bug_rate": rate(judged["valid_api_bug"], judged["valid"]),
            "judge_api_bug_rate_on_mutants": rate(judged["mutant_api_bug"], judged["mutant"]),
            "judged_rounds": judged["valid"] + judged["mutant"]}


SERVICE_SUFFIX = re.compile(r"__(fastapi|nestjs-realworld|todoapp)(__part\d+)?(__r15)?$")   # parts of one run (D-51, D-85)


def main() -> int:
    # Jobs that are per-service parts of one run (exp_id ends in __<service>) are merged.
    runs = {}
    for meta_path in sorted(RESULTS_ROOT.glob("*/*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("group") in ("smoke",):
            continue
        key = (meta["group"], SERVICE_SUFFIX.sub("", meta["exp_id"]))
        runs.setdefault(key, []).append((meta_path.parent, meta))
    rows = []
    for (group, run_id), parts in sorted(runs.items()):
        metas = [m for _, m in parts]
        s = summarize([d for d, _ in parts])
        p = s["pooled"]
        total = lambda k: sum(m.get(k) or 0 for m in metas) if any(m.get(k) is not None for m in metas) else None
        row = {"group": group, "exp_id": run_id, "tool": metas[0]["tool"], "variant": metas[0]["variant"],
               "rep": metas[0]["rep"], "status": "ok" if all(m.get("status") == "ok" for m in metas) else
               ";".join(str(m.get("status")) for m in metas), "services": "+".join(s["per_service"]),
               "parts": len(parts),
               "valid_pct": p["valid_pct"], "mutated_pct": p["mutated_pct"], "adaptation_gap": p["adaptation_gap"],
               "tests_failing_on_correct_pct": p["valid_tests_failing_pct"],
               "hours": round(sum(m.get("seconds") or 0 for m in metas) / 3600, 2),
               "llm_requests": total("llm_requests"), "input_tokens": total("input_tokens"),
               "output_tokens": total("output_tokens")}
        sig = [refinement_signals(d) for d, _ in parts]
        merged = {k: sum(x[k] for x in sig) for k in ("refine_rounds", "value_copy_rounds", "api_bug_verdicts", "judged_rounds")}
        for k in ("judge_false_api_bug_rate", "judge_api_bug_rate_on_mutants"):     # pooled over parts by judged rounds
            num = sum((x[k] or 0) * x["judged_rounds"] for x in sig)
            merged[k] = round(num / merged["judged_rounds"], 4) if merged["judged_rounds"] else None
        merged["value_copy_rate"] = (round(merged["value_copy_rounds"] / merged["refine_rounds"], 4)
                                     if merged["refine_rounds"] else None)
        row.update(merged)
        rows.append(row)
    if not rows:
        print("no finished jobs yet")
        return 0
    out = RESULTS_ROOT / "summary_table.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
    print(f"{'job':45} {'valid%':>7} {'mut%':>6} {'gap':>6} {'failOK%':>8} {'copy%':>6} {'h':>5}")
    for r in rows:
        copy = None if r["value_copy_rate"] is None else 100 * r["value_copy_rate"]
        print(f"{r['group'] + '/' + r['exp_id']:45} {fmt(r['valid_pct']):>7} {fmt(r['mutated_pct']):>6} "
              f"{fmt(r['adaptation_gap']):>6} {fmt(r['tests_failing_on_correct_pct']):>8} {fmt(copy):>6} {fmt(r['hours']):>5}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
