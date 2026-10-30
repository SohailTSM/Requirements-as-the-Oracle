#!/usr/bin/env python3
"""Run one RESTestBench benchmark job with one of our tools, without modifying the harness.

Example:
    .venv/bin/python scripts/run_benchmark.py --tool b1_single_step --services fastapi \
        --variant vague --rep 1 --group baselines

Results go to results/<group>/<exp_id>/ ; the harness summary CSV goes to
results/benchmark_results.csv. A run_meta.json records settings and timing, and the
run is logged to Weights & Biases (failures there never fail the run).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import HARNESS_ROOT, RESULTS_ROOT, OLLAMA_BASE_URL, LLAMA_MODEL_NAME, SERVICES, public_llm_url  # noqa: E402

sys.path.insert(0, str(HARNESS_ROOT))


def _load_harness_runner():
    spec = importlib.util.spec_from_file_location(
        "rtb_runner", HARNESS_ROOT / "scripts" / "run_mutation_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUTPUT_DIR = RESULTS_ROOT          # summary CSV + relative output folders under our results/
    return mod


def build_tool(name: str, agent_config: str | None = None, rep: int = 1):
    """Tool registry. Baselines wrap RESTestBench's own code unchanged."""
    if name == "b1_single_step":
        from tools.single_step_tool.single_step_tool import SingleStepTool
        return SingleStepTool(model_id="llama-3-1-8b")
    if name == "b2_refinement":
        from tools_ext.logged_refinement import LoggedRefinementTool
        return LoggedRefinementTool(model_id="llama-3-1-8b")
    if name == "b3_prompt_fixed":
        from tools_ext.prompt_fixed_refinement import PromptFixedRefinementTool
        return PromptFixedRefinementTool(model_id="llama-3-1-8b")
    if name == "golden":
        # RESTestBench's hand-written reference tests (no LLM): checks the environment (services,
        # mutants, resets, service copies). Only the hard-coded base URL is pointed at the instance (D-86).
        import re as _re
        from tools.golden_tests import GoldenTests

        class _Golden(GoldenTests):
            def generate_test(self, scenario_description, test_file_path, test_dir_path, spec_path,
                              reset_script_path, service_base_url="http://localhost:8000"):
                out = super().generate_test(scenario_description, test_file_path, test_dir_path, spec_path,
                                            reset_script_path, service_base_url)
                f = Path(test_file_path)
                f.write_text(_re.sub(r'BASE_URL = "http://localhost:\d+"', f'BASE_URL = "{service_base_url}"', f.read_text()))
                return out
        return _Golden()
    if name == "e_fuzzer":
        from tools_ext.fuzzer_tool import FuzzerTool
        return FuzzerTool(seed=rep)
    if name == "b4_response_judge":
        # LogiAgent-style oracle: our retrieval + planner make the calls, an LLM judges the responses.
        from tools_ext.pretrained_tool import PretrainedTool
        return PretrainedTool({"oracle": "response_judge", "precision": "none", "refinement": "none",
                               "judge": "none"}, name="b4_response_judge")
    if name == "ours":
        import yaml
        from tools_ext.pretrained_tool import PretrainedTool
        cfg = yaml.safe_load(Path(agent_config).read_text()) if agent_config else {}
        return PretrainedTool(cfg, name=Path(agent_config).stem if agent_config else "default")
    raise ValueError(f"Unknown tool: {name}")


def _git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--services", nargs="+", default=SERVICES)
    ap.add_argument("--variant", choices=["vague", "precise"], required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--group", default="adhoc")
    ap.add_argument("--requirements", nargs="*", type=int, default=None)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--agent-config", default=None, help="YAML with component variants (tool 'ours')")
    ap.add_argument("--phases", nargs="+", default=["generate", "execute", "evaluate"],
                    help="harness phases; 'execute evaluate' re-scores already generated tests (no LLM calls)")
    args = ap.parse_args()

    # Point RESTestBench's llama8b provider at the local Ollama server.
    os.environ.setdefault("AZURE_AI_FOUNDRY_API_KEY", "ollama")
    os.environ.setdefault("AZURE_AI_FOUNDRY_ENDPOINT_OPENAI", OLLAMA_BASE_URL)
    os.environ.setdefault("WANDB_DIR", str(RESULTS_ROOT))
    os.environ.setdefault("WANDB_SILENT", "true")
    from common.secrets import export_env
    export_env()                                # HF/W&B keys for model downloads and logging

    exp_id = args.exp_id or f"{args.tool}__{args.variant}__rep{args.rep}"
    out_dir = RESULTS_ROOT / args.group / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _load_harness_runner()
    from common.harness import patch_health_wait, patch_working_copy_ownership
    patch_working_copy_ownership()
    patch_health_wait()
    tool = build_tool(args.tool, args.agent_config, args.rep)
    config = runner.BenchmarkConfig(
        tool_name=args.tool, services=args.services, requirements=args.requirements,
        phases=args.phases, output_dir=out_dir,
        services_root=runner.SERVICES_DIR, requirements_root=runner.REQUIREMENTS_DIR,
        use_precise=(args.variant == "precise"), timeout=args.timeout)

    meta = {
        "exp_id": exp_id, "group": args.group, "tool": args.tool, "variant": args.variant,
        "rep": args.rep, "services": args.services, "requirements": args.requirements,
        "agent_config": getattr(tool, "config", None),
        "model": LLAMA_MODEL_NAME, "model_server": public_llm_url(),
        "model_settings": "Ollama llama3.1:8b Q4_K_M, num_ctx 24576, flash attention, q8_0 KV cache",
        "project_commit": _git_rev(PROJECT_ROOT), "harness_commit": _git_rev(PROJECT_ROOT / "harness"),
        "harness_instance": "original" if HARNESS_ROOT == PROJECT_ROOT / "harness" else HARNESS_ROOT.name,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    if "generate" not in args.phases and (out_dir / "run_meta.json").exists():
        # Re-scoring existing tests: keep the generation record, note the re-score.
        old = json.loads((out_dir / "run_meta.json").read_text())
        old.setdefault("rescored", []).append({"phases": args.phases, "at": meta["started"],
                                               "project_commit": meta["project_commit"]})
        meta = old
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    status = "ok"
    try:
        runner.MutationBenchmark(config, tool_instance=tool).run_all()
        if "evaluate" in args.phases:
            # The harness skips a service it cannot reach and still exits normally; that is a failure.
            missing = [svc for svc in args.services if not (out_dir / svc / "evaluation_results.json").exists()]
            if missing:
                status = f"error: no evaluation for {missing} (service unreachable or nothing generated)"
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        if "generate" not in args.phases and meta.get("rescored"):
            meta["rescored"][-1].update(seconds=round(time.time() - t0, 1), status=status)
        else:
            meta.update(finished=datetime.now().isoformat(timespec="seconds"),
                        seconds=round(time.time() - t0, 1), status=status,
                        llm_requests=getattr(tool, "request_count", None),
                        input_tokens=getattr(tool, "total_input_tokens", None),
                        output_tokens=getattr(tool, "total_output_tokens", None))
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        if not args.no_wandb:
            try:
                from common.wandb_log import log_benchmark_run
                log_benchmark_run(out_dir)
            except Exception as e:  # logging must never break a run
                print(f"[wandb] logging failed: {type(e).__name__}: {e}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
