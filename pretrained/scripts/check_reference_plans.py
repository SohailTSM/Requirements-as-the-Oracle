#!/usr/bin/env python3
"""Executor check: hand-written reference plans must pass on the correct service and kill
the requirement's planted bugs (same procedure as the harness's execute phase).

    .venv/bin/python scripts/check_reference_plans.py --service fastapi
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from agent.compiler import compile_plan  # noqa: E402
from common.harness import (SERVICE_PORTS, load_requirement, prepare_mutant, prepare_valid,  # noqa: E402
                            run_pytest, service_manager)
from common.paths import RESULTS_ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="fastapi")
    ap.add_argument("--requirements", nargs="*", type=int)
    ap.add_argument("--no-mutants", action="store_true")
    args = ap.parse_args()

    plans = json.loads((PROJECT_ROOT / "gold/reference_plans" / f"{args.service}.json").read_text())
    out_dir = RESULTS_ROOT / "executor_check" / args.service
    out_dir.mkdir(parents=True, exist_ok=True)
    port = SERVICE_PORTS[args.service]
    sm = service_manager()
    sm.start_service(args.service, port)
    summary = []
    try:
        for rid, plan in plans.items():
            if args.requirements and int(rid) not in args.requirements:
                continue
            req = load_requirement(args.service, int(rid))
            test_file = out_dir / f"test_req_{rid}.py"
            test_file.write_text(compile_plan(plan, args.service, f"http://localhost:{port}", req.requirement_precise))
            prepare_valid(sm, args.service)
            ok, output, report = run_pytest(test_file)
            row = {"requirement": int(rid), "valid_passed": ok,
                   "setup_errors": (report or {}).get("setup_errors"),
                   "failed_checks": [c for c in (report or {}).get("checks", []) if not c["passed"]],
                   "mutants_killed": [], "mutants_survived": []}
            if not ok:
                row["output_tail"] = output[-800:]
            if ok and not args.no_mutants:
                for i, mutant in enumerate(req.mutants):
                    prepare_mutant(sm, args.service, mutant)
                    m_ok, _, _ = run_pytest(test_file)
                    (row["mutants_survived"] if m_ok else row["mutants_killed"]).append(i)
            summary.append(row)
            print(json.dumps(row)[:600], flush=True)
    finally:
        prepare_valid(sm, args.service)
        sm.stop_all_services()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    passed = sum(r["valid_passed"] for r in summary)
    killed = sum(len(r["mutants_killed"]) for r in summary)
    total = sum(len(r["mutants_killed"]) + len(r["mutants_survived"]) for r in summary)
    print(f"\nvalid passed: {passed}/{len(summary)}; mutants killed: {killed}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
