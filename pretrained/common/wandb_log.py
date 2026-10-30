"""Log a finished benchmark job to Weights & Biases.

Reads results/<group>/<exp_id>/run_meta.json and each service's evaluation_results.json.
Metrics follow RESTestBench's scoring: a test earns credit for killed mutants only if it
passes on the correct service.
"""
import json
from pathlib import Path

from .secrets import export_env

WANDB_PROJECT = "pretrained-restestbench"


PORT_BOUND = {"fastapi": [15]}        # requirements tied to fixed host ports (D-85)


def _recount(reqs: list) -> dict:
    """The harness summary recomputed over a subset of requirements."""
    vs = sum(r.get("valid_test", {}).get("score", 0) for r in reqs)
    vm = sum(r.get("valid_test", {}).get("max_score", 0) for r in reqs)
    mts = [r.get("mutated_tests") or {} for r in reqs]
    ms = sum(m.get("average_score", 0) or 0 for m in mts)
    mm = sum(m.get("max_score", 0) or 0 for m in mts if m.get("count"))
    return {"valid_score": vs, "valid_max_score": vm, "valid_percentage": 100 * vs / vm if vm else 0.0,
            "mutated_score": ms, "mutated_max_score": mm, "mutated_percentage": 100 * ms / mm if mm else 0.0}


def summarize(out_dir) -> dict:
    """Per-service and pooled scores for one job, or for several job folders that are parts of
    one run (one folder per service; also used by analysis scripts)."""
    dirs = [Path(d) for d in out_dir] if isinstance(out_dir, (list, tuple)) else [Path(out_dir)]
    pooled = {"valid_score": 0.0, "valid_max": 0, "mutated_score": 0.0, "mutated_max": 0,
              "valid_tests": 0, "valid_tests_passing": 0}
    per_service, rows = {}, []
    svc_dirs = sorted((p for d in dirs for p in d.iterdir() if (p / "evaluation_results.json").exists()),
                      key=lambda p: p.name)
    for svc_dir in svc_dirs:
        ev = json.loads((svc_dir / "evaluation_results.json").read_text())
        reqs = ev["requirements"]
        meta_f = svc_dir.parent / "run_meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        drop = set(meta.get("excluded_requirements", []))
        if meta.get("harness_instance", "original") != "original":
            drop |= set(PORT_BOUND.get(svc_dir.name, []))      # tested on the original instance instead (D-85)
        if drop:
            reqs = [r for r in reqs if r["requirement_id"] not in drop]
        s = _recount(reqs) if drop else ev["summary"]
        passing = sum(1 for r in reqs if r.get("valid_test", {}).get("passes_on_valid"))
        acc = per_service.setdefault(svc_dir.name, {"_vs": 0, "_vm": 0, "_ms": 0, "_mm": 0, "_pass": 0, "_n": 0})
        acc["_vs"] += s.get("valid_score", 0); acc["_vm"] += s.get("valid_max_score", 0)
        acc["_ms"] += s.get("mutated_score", 0); acc["_mm"] += s.get("mutated_max_score", 0)
        acc["_pass"] += passing; acc["_n"] += len(reqs)
        acc.update(valid_pct=100 * acc["_vs"] / acc["_vm"] if acc["_vm"] else 0.0,
                   mutated_pct=100 * acc["_ms"] / acc["_mm"] if acc["_mm"] else 0.0,
                   valid_tests_passing_pct=100 * acc["_pass"] / acc["_n"] if acc["_n"] else None)
        pooled["valid_score"] += s.get("valid_score", 0)
        pooled["valid_max"] += s.get("valid_max_score", 0)
        pooled["mutated_score"] += s.get("mutated_score", 0)
        pooled["mutated_max"] += s.get("mutated_max_score", 0)
        pooled["valid_tests"] += len(reqs)
        pooled["valid_tests_passing"] += passing
        for r in reqs:
            vt, mt = r.get("valid_test", {}), r.get("mutated_tests", {})
            rows.append([svc_dir.name, r["requirement_id"], r["num_mutants"],
                         vt.get("passes_on_valid"), vt.get("score"),
                         mt.get("average_score") if mt else None])
    pct = lambda a, b: 100 * a / b if b else None
    return {
        "per_service": per_service,
        "pooled": {
            "valid_pct": pct(pooled["valid_score"], pooled["valid_max"]),
            "mutated_pct": pct(pooled["mutated_score"], pooled["mutated_max"]),
            "adaptation_gap": (pct(pooled["valid_score"], pooled["valid_max"]) - pct(pooled["mutated_score"], pooled["mutated_max"]))
                              if pooled["mutated_max"] else None,
            "valid_tests_passing_pct": pct(pooled["valid_tests_passing"], pooled["valid_tests"]),
            "valid_tests_failing_pct": pct(pooled["valid_tests"] - pooled["valid_tests_passing"], pooled["valid_tests"]),
        },
        "rows": rows,
    }


def log_benchmark_run(out_dir: Path) -> None:
    import wandb
    export_env()
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "run_meta.json").read_text())
    summary = summarize(out_dir)
    run = wandb.init(project=WANDB_PROJECT, name=meta["exp_id"], group=meta["group"],
                     job_type=meta["tool"], config=meta,
                     tags=[meta["tool"], meta["variant"], f"rep{meta['rep']}"] + list(meta["services"]),
                     reinit="finish_previous")
    flat = {f"pooled/{k}": v for k, v in summary["pooled"].items() if v is not None}
    for svc, m in summary["per_service"].items():
        flat.update({f"{svc}/{k}": v for k, v in m.items() if v is not None})
    flat.update({k: meta.get(k) for k in ("seconds", "llm_requests", "input_tokens", "output_tokens")
                 if meta.get(k) is not None})
    run.summary.update(flat)
    run.log({"requirements": wandb.Table(
        columns=["service", "requirement", "mutants", "passes_on_valid", "valid_score", "mutated_avg_score"],
        data=summary["rows"])})
    run.finish()
