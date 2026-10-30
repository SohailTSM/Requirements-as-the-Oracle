"""Architecture variant "react": one tool-calling agent decides the order of the steps itself.

The pipeline's guarantees are enforced inside the tools, not left to the agent:
  - run_plan reports status codes, response field names and which checks failed, never values
  - after the first run, changes to a meaning check's expectation are reverted (same guard)
  - every plan goes through the same spec-only clean-up; failed format checks are relaxed once
  - at most MAX_RUNS runs against the service (same budget as the pipeline's rounds)
So "react" vs "pipeline" measures who controls the order of steps, not the safety rules.
"""
from __future__ import annotations

import json

from agent.compiler import compile_plan
from agent.lint import field_problems, lint_plan
from agent.planner import PLAN_FORMAT, make_plan, parse_plan
from agent.precision import derive_checks
from agent.refine import _failed_checks, _trace, guard
from common.harness import run_pytest

MAX_RUNS = 3
MAX_TURNS = 14

PLAN_PARAM = {"type": "object", "properties": {"plan": {"type": "object", "description":
                       "the complete test plan: an object with \"actors\", \"steps\" and \"checks\", e.g. "
                       "{\"actors\": {\"A\": {\"type\": \"new_user\"}}, \"steps\": [{\"id\": \"s1\", \"actor\": \"A\", "
                       "\"call\": \"POST /api/items\", \"body\": {\"title\": \"x\"}, \"role\": \"action\"}], "
                       "\"checks\": [{\"id\": \"C1\", \"step\": \"s1\", \"expect\": \"status_class\", \"value\": \"2xx\"}]}"}},
                       "required": ["plan"]}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_operations",
        "description": "Find API operations relevant to a short query. Returns method, path, parameters "
                       "and response fields of the best matches.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "derive_checks",
        "description": "Expand the requirement into the concrete checks a good test must make.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "check_plan",
        "description": "Check a test plan against the API specification without calling the service.",
        "parameters": PLAN_PARAM}},
    {"type": "function", "function": {
        "name": "run_plan",
        "description": "Run a test plan against the service. Returns the calls' status codes, response "
                       "field names and which checks failed (not the response values).",
        "parameters": PLAN_PARAM}},
    {"type": "function", "function": {
        "name": "submit_plan",
        "description": "Submit the final test plan and stop.",
        "parameters": PLAN_PARAM}},
]

SYSTEM = """You are an expert API tester. Your job: write a test plan for one requirement of a REST API,
using the tools. Suggested order: search_operations, derive_checks, write the plan, check_plan,
run_plan, then submit_plan.

{plan_format}

Rules:
- The service may contain bugs. A failed check can mean the test found a bug; then submit the plan.
- After a plan has run, you may fix how the test runs (calls, ids, request bodies, setup steps), but
  not what a check expects. Such changes are reverted automatically.
- You can run a plan at most {max_runs} times. Always finish with submit_plan."""


def _text_tool_call(text: str):
    """Llama 3.1 sometimes writes a tool call as text in its own JSON format, e.g.
    {"name": "write_plan", "parameters": {...plan...}} (even with an invented name).
    Returns (name, args) or None."""
    if not text or "{" not in text:
        return None
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return None
    if isinstance(obj, dict) and "name" in obj and isinstance(obj.get("parameters", obj.get("arguments")), dict):
        return str(obj["name"]), obj.get("parameters", obj.get("arguments"))
    return None


def _ops_text(ops) -> str:
    return "\n\n".join(o.to_text() for o in ops)


def run_react(tool, service: str, requirement: str, test_file, base_url: str, history: dict) -> None:
    """Generate the test for one requirement with the ReAct agent; writes test_file."""
    from tools_ext.pretrained_tool import _mark_meaning, _relaxable_format_checks, _spec_cleanup
    cfg = tool.config
    ops, auth, retriever = tool._service(service)
    llm = tool.llm
    state = {"plan": None, "ran": False, "runs": 0, "relaxed": False, "seen_ops": [], "checks": [],
             "submitted": None, "report": None}
    spec_codes = {c for o in ops for c in o.responses}
    events = history.setdefault("react", [])

    def accept(raw) -> tuple[dict | None, str]:
        plan = parse_plan(raw if isinstance(raw, str) else json.dumps(raw))
        if plan is None or not plan.get("steps"):
            return None, ('That is not a usable plan. Pass the whole plan object, e.g. {"actors": {...}, '
                          '"steps": [{"id": "s1", "call": "<METHOD> <path>", ...}], "checks": [...]}.')
        _mark_meaning(plan, state["checks"])
        note = ""
        if state["ran"] and state["plan"] is not None:
            plan, ev = guard(state["plan"], plan, state["report"] or {"steps": []}, spec_codes,
                             lambda p: field_problems(p, ops))
            reverted = [e["check"] for e in ev if e["type"] in ("locked_check_restored", "value_copy_blocked",
                                                                 "field_repair_rejected_not_in_spec")]
            if reverted:
                note = f"Changes to the expectations of {', '.join(reverted)} were reverted. "
            events.append({"guard": ev})
        plan, ev = _spec_cleanup(plan, ops, cfg, requirement, auth)
        events.append({"cleanup": ev})
        return plan, note

    def t_search(args):
        found = retriever.rank(str(args.get("query") or requirement), cfg["retrieval"])[: cfg["top_k"]]
        for o in found:
            if o.op_id not in [x.op_id for x in state["seen_ops"]]:
                state["seen_ops"].append(o)
        return _ops_text(found)

    def t_checks(args):
        out = derive_checks(requirement, state["seen_ops"] or retriever.rank(requirement, cfg["retrieval"])[:5],
                            cfg["precision"], llm) if cfg["precision"] != "none" else {"checks": []}
        state["checks"] = out["checks"]
        return "\n".join(f"- {c['id']} ({c['kind']}): {c['description']}" for c in out["checks"]) or "(none)"

    def t_check_plan(args):
        plan, note = accept(args.get("plan", ""))
        if plan is None:
            return note
        issues = lint_plan(plan, ops)
        return note + ("No problems found." if not issues else "Problems:\n" + "\n".join(f"- {i}" for i in issues))

    def t_run(args):
        if state["runs"] >= MAX_RUNS:
            return f"Run limit reached ({MAX_RUNS}). Call submit_plan."
        plan, note = accept(args.get("plan", ""))
        if plan is None:
            return note
        state["runs"] += 1
        while True:
            test_file.write_text(compile_plan(plan, service, base_url, requirement, ops, auth))
            passed, output, report = run_pytest(test_file)
            report = report or {"steps": [], "checks": [], "setup_errors": [f"no report: {output[-300:]}"]}
            failed_ids = {c["id"] for c in report.get("checks", []) if not c["passed"]}
            failed = [c for c in plan.get("checks", []) if c.get("id") in failed_ids]
            relax = _relaxable_format_checks(failed, plan, report, ops) if not report.get("setup_errors") else []
            if relax and not state["relaxed"]:
                plan = dict(plan, checks=[c for c in plan["checks"] if c.get("id") not in relax])
                state["relaxed"] = True
                note += f"Format checks {', '.join(relax)} were not supported by the specification and were dropped. "
                continue
            break
        state.update(plan=plan, ran=True, report=report)
        events.append({"run": state["runs"], "passed": passed, "report": report})
        if passed:
            return note + "All checks passed. Call submit_plan with this plan."
        return (note + "Calls:\n" + _trace(report, hide_values=True) + "\nSetup errors:\n"
                + ("\n".join(report.get("setup_errors", [])) or "(none)")
                + "\nFailed checks:\n" + _failed_checks(report, plan, hide_values=True))

    def t_submit(args):
        plan, note = accept(args.get("plan", ""))
        if plan is None:
            return note + " Submit a valid plan."
        state["submitted"] = plan
        return "Submitted."

    handlers = {"search_operations": t_search, "derive_checks": t_checks, "check_plan": t_check_plan,
                "run_plan": t_run, "submit_plan": t_submit}
    messages = [{"role": "system", "content": SYSTEM.format(plan_format=PLAN_FORMAT, max_runs=MAX_RUNS)},
                {"role": "user", "content": f'Requirement: "{requirement}"\nStart by searching for the operations you need.'}]
    nudges = 0
    for turn in range(MAX_TURNS):
        out = llm.chat_tools(messages, TOOLS, max_tokens=1800, tag="react.turn")
        if not out["tool_calls"]:
            call = _text_tool_call(out["content"] or "")
            if call and call[0] in handlers:              # a real tool, written as text: execute it
                name, args = call
                messages.append({"role": "assistant", "content": out["content"]})
                result = handlers[name](args)
                events.append({"turn": turn, "tool": f"{name}(text)", "result": result[:1500]})
                messages.append({"role": "user", "content": f"Result of {name}:\n{result}"})
                if state["submitted"] is not None:
                    break
                continue
            # A plan written as plain text (or inside an invented tool call) is what the agent means to
            # use: run it while no run has happened, otherwise submit it.
            text_plan = parse_plan(out["content"] or "")
            if (text_plan is None or not text_plan.get("steps")) and call:
                args = call[1]
                text_plan = parse_plan(json.dumps(args.get("plan", args)))
                if text_plan is not None:
                    out = dict(out, content=json.dumps(text_plan))
            if text_plan is not None and text_plan.get("steps"):
                messages.append({"role": "assistant", "content": out["content"]})
                if state["runs"] < MAX_RUNS and not state["ran"]:
                    result = t_run({"plan": out["content"]})
                    events.append({"turn": turn, "tool": "run_plan(text)", "result": result[:1500]})
                    messages.append({"role": "user", "content": "I ran that plan for you:\n" + result +
                                     "\nFix it with run_plan if needed, or call submit_plan."})
                    continue
                t_submit({"plan": out["content"]})
                break
            nudges += 1
            messages.append({"role": "assistant", "content": out["content"]})
            messages.append({"role": "user", "content": "Use one of the tools. Call submit_plan when the plan is final."})
            if nudges > 2:
                break
            continue
        messages.append({"role": "assistant", "content": out["content"], "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in out["tool_calls"]]})
        for c in out["tool_calls"]:
            try:
                args = json.loads(c["arguments"]) if isinstance(c["arguments"], str) else (c["arguments"] or {})
            except Exception:
                args = {}
            fn = handlers.get(c["name"])
            result = fn(args) if fn else f"Unknown tool {c['name']}."
            events.append({"turn": turn, "tool": c["name"], "result": result[:1500]})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        if state["submitted"] is not None:
            break

    final = state["submitted"] or state["plan"]
    if final is None:      # the agent never produced a usable plan: fall back to the planner once
        selected = state["seen_ops"] or retriever.rank(requirement, cfg["retrieval"])[: cfg["top_k"]]
        final, _ = make_plan(requirement, state["checks"], selected, llm, tag="react.fallback_planner")
        history["react_fallback"] = True
        if final is None:
            raise ValueError("react agent produced no plan")
        final, _ = _spec_cleanup(final, ops, cfg, requirement, auth)
    history["final_plan"] = final
    history["rounds"].append({"round": state["runs"], "plan": final, "decision": "react_submit"
                              if state["submitted"] else "react_last_plan"})
    test_file.write_text(compile_plan(final, service, base_url, requirement, ops, auth))
