"""Our agent as a RESTestBench tool.

Pipeline per requirement:
    retrieve operations -> derive checks (precision layer) -> plan -> compile to pytest
    -> run on the service -> (refine / judge, up to MAX_ROUNDS) -> final test file.
Each component's variant comes from the experiment config (configs/agent/*.yaml).
Everything is recorded in agent_history.json next to the generated test.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agent.compiler import compile_plan, infer_auth
from agent.llm import LLM
from agent.lint import (REPAIR_PROMPT, autofill_path_params, define_missing_actors, drop_impossible,
                        field_problems, fix_misdirected_refs, generalize_unstated_status, lint_plan,
                        remove_redundant_signup, step_operations)
from agent.planner import make_plan, parse_plan
from agent.refine import guard
from agent.precision import derive_checks, support_filter
from agent.refine import judge, refine
from agent.retrieval import Retriever, rewrite_queries
from agent.spec import load_operations
from common.harness import run_pytest

MAX_ROUNDS = 3        # same budget as RESTestBench's refinement tool (initial + 2 refinements)

DEFAULT_CONFIG = {
    "retrieval": "adaptive", "top_k": 5,
    "precision": "catalogue",
    "generation": "plan",            # plan (compiler) | raw (LLM writes pytest code directly)
    "refinement": "combined",
    "judge": "evidence",
    "architecture": "pipeline",
    "model": "base",
}


class PretrainedTool:
    test_file_name = "test.py"
    APPROACH_NAME = "pretrained-agent"
    logger = None

    def __init__(self, config: dict | None = None, name: str = "pretrained"):
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.model_id = name
        self.INTERACTIVE = self.config["refinement"] != "none"
        self.llm = LLM()
        self._retrievers, self._ops, self._auth = {}, {}, {}
        self.total_input_tokens = self.total_output_tokens = self.request_count = 0

    # harness interface ----------------------------------------------------------
    def get_test_command(self, test_file_path: str) -> str:
        # Same form as RESTestBench's own tools: this interpreter's pytest, quoted path (the
        # workspace path contains spaces and the harness runs the command through a shell).
        return f'"{sys.executable}" -m pytest "{test_file_path}" -v --tb=short'


    def generate_test(self, scenario_description, test_file_path, test_dir_path, spec_path,
                      reset_script_path, service_base_url="http://localhost:8000"):
        service = Path(spec_path).parent.name
        Path(test_dir_path).mkdir(parents=True, exist_ok=True)
        calls_before = len(self.llm.calls)
        t0 = time.time()
        history = {"config": self.config, "requirement": scenario_description, "service": service, "rounds": []}
        try:
            self._generate(service, scenario_description, Path(test_file_path), service_base_url, history)
        except Exception as e:  # never crash the benchmark; leave a failing test behind
            history["error"] = f"{type(e).__name__}: {e}"
            if not Path(test_file_path).exists():
                Path(test_file_path).write_text(
                    f'def test_requirement():\n    raise AssertionError("generation failed: {type(e).__name__}")\n')
        finally:
            new_calls = self.llm.calls[calls_before:]
            history["llm_calls"] = new_calls
            if getattr(self, "_a2a", None) is not None:
                history["a2a_timings"] = self._a2a.timings[:]
                self._a2a.timings.clear()
            if getattr(self, "_mcp", None) is not None:
                history["mcp_timings"] = self._mcp.timings[:]
                self._mcp.timings.clear()
            history["seconds"] = round(time.time() - t0, 1)
            self.request_count += len(new_calls)
            self.total_input_tokens += sum(c["prompt_tokens"] or 0 for c in new_calls)
            self.total_output_tokens += sum(c["completion_tokens"] or 0 for c in new_calls)
            (Path(test_dir_path) / "agent_history.json").write_text(json.dumps(history, indent=1, default=str))
        return 0.0

    # tool calls (in-process, or over MCP for the communication variant) ----------
    def _mcp_tools(self):
        if getattr(self, "_mcp", None) is None:
            from tools_ext.mcp_client import MCPTools
            self._mcp = MCPTools()
        return self._mcp

    def _role_llm(self, role: str, service: str):
        """Which model serves a role (sweep row "model", Decision Log D-68). Fine-tuned models exist
        per held-out service (leave-one-service-out), so the name ends in the service."""
        model = self.config.get("model", "base")
        sft = f"{self.config.get('sft_model', 'pt-precision-sft')}-{service}"
        dpo = f"{self.config.get('dpo_model', 'pt-dpo')}-{service}"
        name = {"sft": sft, "sft_dpo": {"precision": sft, "judge": dpo, "refine": dpo}.get(role)}.get(model)
        if not name:
            return self.llm
        return LLM(model=name, calls=self.llm.calls)        # same call log, so tokens are counted

    def _a2a_ask(self, role, payload):
        if getattr(self, "_a2a", None) is None:
            from tools_ext.a2a_client import A2ATeam
            self._a2a = A2ATeam()
        out = self._a2a.ask(role, payload)
        self.llm.calls.extend(out["llm_calls"])          # keep token accounting complete
        return out["result"]

    def _lint(self, service, plan, ops):
        if self.config.get("communication") == "mcp":
            return self._mcp_tools().check_plan(service, plan)
        return lint_plan(plan, ops)

    def _run(self, test_file):
        if self.config.get("communication") == "mcp":
            return self._mcp_tools().run_test(test_file)
        return run_pytest(test_file)

    # pipeline -------------------------------------------------------------------
    def _service(self, service):
        if service not in self._ops:
            self._ops[service] = load_operations(service)
            self._auth[service] = infer_auth(service, self._ops[service])
            self._retrievers[service] = Retriever(service)
        return self._ops[service], self._auth[service], self._retrievers[service]

    def _generate(self, service, requirement, test_file: Path, base_url, history):
        cfg = self.config
        if cfg.get("architecture") == "react":
            from tools_ext.react_agent import run_react
            return run_react(self, service, requirement, test_file, base_url, history)
        if cfg.get("architecture") in ("multi_persona", "blackboard"):
            from tools_ext.multi_agent import run_multi_agent
            return run_multi_agent(self, service, requirement, test_file, base_url, history)
        ops, auth, retriever = self._service(service)
        extra = rewrite_queries(requirement, self.llm) if cfg["retrieval"] == "query_rewrite" else None
        if cfg.get("communication") == "mcp":       # same tools, served over MCP (D-66)
            by_id = {o.op_id: o for o in ops}
            k = len(ops) if cfg["retrieval"] == "full" else max(cfg["top_k"], 12)
            ranked = [by_id[i] for i in self._mcp_tools().search_operations(service, requirement, cfg["retrieval"], k)
                      if i in by_id]
        else:
            ranked = retriever.rank(requirement, cfg["retrieval"], extra)
        selected = ranked if cfg["retrieval"] == "full" else ranked[: cfg["top_k"]]
        if cfg.get("dependency_closure", True):
            selected = selected + producers_for(selected, ops)
        ops_text = "\n\n".join(o.to_text() for o in selected)
        history["operations"] = [o.op_id for o in selected]

        if cfg["precision"] == "finetuned":
            # Leave-one-service-out adapter: trained without this service's requirements.
            ft = LLM(model=f"{cfg.get('precision_model', 'pt-precision-sft')}-{service}")
            derived = derive_checks(requirement, selected, "finetuned", ft)
            self.llm.calls.extend(ft.calls)
        else:
            derived = derive_checks(requirement, selected, cfg["precision"], self._role_llm("precision", service))
        checks = derived["checks"]
        if cfg.get("support_filter", False):
            history["derived_checks_unfiltered"] = checks
            checks, history["support_filter"] = support_filter(requirement, checks, self.llm)
        history["derived_checks"] = checks

        if cfg["generation"] == "raw":
            code = self.llm.complete(RAW_PROMPT.format(requirement=requirement, base_url=base_url,
                                                       operations=ops_text,
                                                       checks="\n".join(f"- {c['description']}" for c in checks) or "-"),
                                     max_tokens=2500, tag="generate.raw")
            test_file.write_text(_extract_code(code))
            history["rounds"].append({"round": 1, "raw_code": code, "decision": "single_step"})
            return

        plan, raw = make_plan(requirement, checks, selected, self._role_llm("planner", service))
        if plan is None:                                  # one retry on unparseable output
            plan, raw = make_plan(requirement, checks, selected, self._role_llm("planner", service), tag="planner.retry")
        history["initial_plan_raw"] = raw
        if plan is None:
            raise ValueError("planner produced no valid plan")
        _mark_meaning(plan, checks)

        # Static repair (spec-only, before any service call): at most one round.
        issues = self._lint(service, plan, ops)
        history["lint_issues"] = issues
        if issues and cfg.get("lint", True):
            raw = self._role_llm("planner", service).complete(REPAIR_PROMPT.format(issues="\n".join(f"- {i}" for i in issues),
                                                         operations=ops_text, plan=json.dumps(plan, indent=1)),
                                    max_tokens=1800, tag="lint.repair")
            repaired = parse_plan(raw)
            if repaired is not None:
                repaired, events = guard(plan, repaired, {"steps": []}, set())
                _mark_meaning(repaired, checks)
                history["lint_repair"] = {"raw": raw, "events": events,
                                          "remaining_issues": lint_plan(repaired, ops)}
                plan = repaired
        plan, history["static_events"] = _spec_cleanup(plan, ops, cfg, requirement, auth)

        if cfg.get("oracle") == "response_judge":
            # B4 baseline: same scenario, but the oracle is an LLM verdict computed at test time.
            from agent.compiler import compile_judged_plan
            from common.paths import LLAMA_MODEL_NAME, OLLAMA_BASE_URL
            test_file.write_text(compile_judged_plan(plan, service, base_url, requirement, ops_text,
                                                     OLLAMA_BASE_URL, LLAMA_MODEL_NAME, ops, auth))
            history["rounds"].append({"round": 1, "plan": plan, "decision": "response_judge"})
            history["final_plan"] = plan
            return

        spec_codes = {code for o in selected for code in o.responses}
        max_rounds = 1 if cfg["refinement"] == "none" else MAX_ROUNDS
        rnd, relaxed = 1, False
        while True:
            test_file.write_text(compile_plan(plan, service, base_url, requirement, ops, auth))
            record = {"round": rnd, "plan": plan}
            if cfg["refinement"] == "none":
                history["rounds"].append(record)
                break
            passed, output, report = self._run(test_file)
            report = report or {"steps": [], "checks": [], "setup_errors": [f"no report: {output[-300:]}"]}
            record.update(passed=passed, report=report)
            if passed:
                record["decision"] = "finish"
                history["rounds"].append(record)
                break
            failed_ids = {c["id"] for c in report.get("checks", []) if not c["passed"]}
            failed = [c for c in plan.get("checks", []) if isinstance(c, dict) and c.get("id") in failed_ids]
            relax = _relaxable_format_checks(failed, plan, report, ops) if not report.get("setup_errors") else []
            if relax and not relaxed and cfg.get("format_relaxation", True):
                # Failed format details the spec does not back are dropped deterministically (never
                # evidence of an API bug). Free pass: no LLM call, does not use a refinement round.
                plan = dict(plan, checks=[c for c in plan["checks"] if c.get("id") not in relax])
                record["decision"] = "format_checks_relaxed"
                record["relaxed_checks"] = sorted(relax)
                history["rounds"].append(record)
                relaxed = True
                continue
            if rnd >= max_rounds:
                record["decision"] = "rounds_exhausted"
                history["rounds"].append(record)
                break
            review = ""
            if not report.get("setup_errors"):
                if cfg.get("communication") == "a2a" and cfg["judge"] != "none":   # reviewer agent over A2A (D-67)
                    verdict = self._a2a_ask("reviewer", {
                        "judge": cfg["judge"], "requirement": requirement, "ops_text": ops_text, "plan": plan,
                        "report": report, "hide_values": cfg["refinement"] in ("hidden", "combined")})
                else:
                    verdict = judge(cfg["judge"], self._role_llm("judge", service), requirement, ops_text, plan, report,
                                    hide_values=cfg["refinement"] in ("hidden", "combined"))
                record["judge"] = verdict
                if verdict["verdict"] == "API_BUG":
                    record["decision"] = "api_bug"
                    history["rounds"].append(record)
                    break
                from tools_ext.multi_agent import _review_for_debugger
                review = _review_for_debugger(verdict, report, cfg.get("content", "summary"))
            if cfg.get("communication") == "a2a":          # debugger agent over A2A (D-67)
                out = self._a2a_ask("debugger", {
                    "refinement": cfg["refinement"], "service": service, "requirement": requirement,
                    "ops_text": ops_text, "plan": plan, "report": report, "review": review,
                    "spec_codes": sorted(spec_codes), "spec_cleanup": cfg.get("spec_cleanup", True)})
                new_plan, info = out["plan"], out["info"]
            else:
                new_plan, info = refine(cfg["refinement"], self._role_llm("refine", service), requirement, ops_text, plan, report,
                                        review=review, spec_codes=spec_codes,
                                        spec_problems=(lambda p: field_problems(p, ops)) if cfg.get("spec_cleanup", True) else None)
            record.update(decision="refine", refine_events=info["events"], refine_raw=info["raw"],
                          refine_guarded=info.get("guarded_plan"))
            history["rounds"].append(record)
            if new_plan is None:
                break
            _mark_meaning(new_plan, checks)
            plan, record["static_events"] = _spec_cleanup(new_plan, ops, cfg, requirement, auth)
            rnd += 1
        history["final_plan"] = plan


RAW_PROMPT = """You are an expert API tester who writes pytest tests to validate that API implementations follow their intended human-written requirements.
Write a complete, executable, self-contained pytest test (use the requests library, hardcode the base URL
{base_url}, no environment variables, no pytest.skip) for the requirement below.

Requirement: "{requirement}"

Checks the test must make:
{checks}

API operations:
{operations}

Generate ONLY the Python test code, no explanations."""


def _extract_code(text: str) -> str:
    import re
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return max(blocks, key=len).strip() + "\n" if blocks else text.strip() + "\n"


def producers_for(selected, ops, limit: int = 3):
    """Dependency closure: for every path parameter of a selected operation, add the POST that
    creates that resource (e.g. GET /items/{id} -> POST /items/), if it is not selected yet."""
    import re
    have = {o.op_id for o in selected}
    extra = []
    for o in selected:
        for m in re.finditer(r"/\{[^}]+\}", o.url_path):
            collection = o.url_path[: m.start()].rstrip("/")
            for cand in ops:
                if (cand.method == "POST" and cand.url_path.rstrip("/") == collection
                        and cand.op_id not in have):
                    extra.append(cand)
                    have.add(cand.op_id)
    return extra[:limit]


def _spec_cleanup(plan: dict, ops, cfg, requirement: str = "", auth: dict | None = None) -> tuple[dict, list[dict]]:
    """Deterministic, spec-only fixes: define undefined actors as new users, fill missing path
    parameters from earlier steps, then drop checks the spec shows can never pass and format status
    checks with undocumented codes."""
    if not cfg.get("spec_cleanup", True):
        return plan, []
    events = []
    for step_fn in (lambda p: define_missing_actors(p, requirement),
                    lambda p: remove_redundant_signup(p, requirement, auth),
                    lambda p: fix_misdirected_refs(p, ops),
                    lambda p: autofill_path_params(p, ops),
                    lambda p: generalize_unstated_status(p, requirement),
                    lambda p: drop_impossible(p, ops)):
        plan, ev = step_fn(plan)
        events += ev
    return plan, events


def _relaxable_format_checks(failed: list[dict], plan: dict, report: dict, ops) -> list[str]:
    """Failed format checks that the spec does not back up: a status format check is relaxed when
    the status the service returned is also documented for that operation (the guess just picked
    the wrong documented code); any other format check is relaxed. A status format check that
    failed with an UNDOCUMENTED status is kept, because that is evidence against the spec."""
    step_ops = step_operations(plan, ops)
    observed = {st.get("id"): str(st.get("status")) for st in report.get("steps", [])}
    out = []
    for c in failed:
        if c.get("kind") != "format":
            continue
        op = step_ops.get(c.get("step"))
        if c.get("expect") in ("status", "status_in") and op is not None and op.responses:
            if observed.get(c.get("step")) not in op.responses:
                continue
        out.append(c["id"])
    return out


def _mark_meaning(plan: dict, checks: list[dict]) -> None:
    """Every planned check carries a kind; anything unlabelled is treated as meaning (locked)."""
    for c in plan.get("checks", []):
        if isinstance(c, dict) and c.get("kind") not in ("meaning", "format"):
            c["kind"] = "meaning"
