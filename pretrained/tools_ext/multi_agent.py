"""Architecture variants with several personas (same Llama 3.1 8B, different roles).

multi_persona  fixed order: Analyst -> Designer -> Critic -> Designer (revision)
               -> up to MAX_ROUNDS x (Executor -> Reviewer -> Debugger)
blackboard     the same personas share a board; a Controller persona picks who acts next

Compared with the pipeline, the personas add role prompts and a Critic that reviews the plan
before any call to the service. The safety rules are the pipeline's and are enforced in code:
hidden values after a run, the guard on every post-run revision, spec clean-up, the run limit.
The Reviewer is the evidence judge; what it passes to the Debugger follows cfg["content"].
"""
from __future__ import annotations

import json

from agent.compiler import compile_plan
from agent.lint import REPAIR_PROMPT, field_problems, lint_plan
from agent.planner import PLAN_FORMAT, make_plan, parse_plan
from agent.precision import derive_checks
from agent.refine import _failed_checks, _trace, guard, judge, refine
from common.harness import run_pytest

MAX_ROUNDS = 3
MAX_CONTROLLER_STEPS = 12

PERSONAS = {
    "analyst": "You are the Requirements Analyst of an API testing team. You turn vague requirements "
               "into concrete, checkable expectations, and you never invent features.",
    "designer": "You are the Test Designer of an API testing team. You write precise, executable test "
                "plans that call only operations from the specification.",
    "critic": "You are the Test Critic of an API testing team. Before a test runs, you find mistakes in "
              "the plan: checks that do not follow from the requirement, wrong steps, missing setup.",
    "reviewer": "You are the Test Reviewer of an API testing team. You decide whether a failed test "
                "found a bug in the service or is itself wrong.",
    "debugger": "You are the Test Debugger of an API testing team. You fix how a test runs without "
                "changing what it expects.",
}

CRITIC_PROMPT = """Requirement: "{requirement}"

API operations:
{operations}

Test plan:
{plan}

List the problems of this plan, one per line, starting with "- ". Only report: checks that do not
follow from the requirement, checks on the wrong step, calls or fields not in the operations, missing
setup steps. If the plan is fine, answer exactly "NO PROBLEMS"."""

CONTROLLER_PROMPT = """You coordinate an API testing team working on a shared board.

Requirement: "{requirement}"

Board:
{board}

Team members you can call next: {options}
- analyst: derive the checks  - designer: write or revise the plan  - critic: review the plan
- executor: run the plan against the service  - reviewer: judge a failed run
- debugger: fix the plan after the reviewer says the test is wrong  - finish: stop
Answer with one name only."""


def _with_persona(llm, persona: str, prompt: str, max_tokens: int, tag: str) -> str:
    return llm.chat([{"role": "system", "content": PERSONAS[persona]}, {"role": "user", "content": prompt}],
                    max_tokens=max_tokens, tag=tag)


class _PersonaLLM:
    """Wraps the shared LLM so pipeline components (derive_checks, make_plan, judge, refine) run with a
    persona system prompt."""
    def __init__(self, llm, persona):
        self.llm, self.persona = llm, persona
        self.calls = llm.calls

    def complete(self, prompt, max_tokens=2048, tag="", temperature_override=None, **kw):
        return self.llm.chat([{"role": "system", "content": PERSONAS[self.persona]},
                              {"role": "user", "content": prompt}], max_tokens=max_tokens,
                             tag=f"{self.persona}.{tag}", temperature_override=temperature_override, **kw)


def _review_for_debugger(verdict: dict, report: dict, content: str) -> str:
    """What the Reviewer passes on (communication-content variants)."""
    if content == "verdict":
        return ""
    if content == "full":
        return (verdict.get("raw", "") + "\nTrace seen by the reviewer:\n" + _trace(report, hide_values=True))
    return verdict.get("reason", "")                                  # "summary" (default)


def run_multi_agent(tool, service: str, requirement: str, test_file, base_url: str, history: dict) -> None:
    from tools_ext.pretrained_tool import _mark_meaning, _relaxable_format_checks, _spec_cleanup, producers_for
    cfg = tool.config
    ops, auth, retriever = tool._service(service)
    llm = tool.llm
    selected = retriever.rank(requirement, cfg["retrieval"])[: cfg["top_k"]]
    if cfg.get("dependency_closure", True):
        selected = selected + producers_for(selected, ops)
    ops_text = "\n\n".join(o.to_text() for o in selected)
    spec_codes = {c for o in selected for c in o.responses}
    hide = cfg["refinement"] in ("hidden", "combined")
    content = cfg.get("content", "summary")
    board = {"operations": [o.op_id for o in selected], "checks": None, "plan": None, "critique": None,
             "runs": [], "review": None, "relaxed": False, "done": False}
    log = history.setdefault("multi_agent", [])

    def analyst():
        out = derive_checks(requirement, selected, cfg["precision"], _PersonaLLM(llm, "analyst"))
        board["checks"] = out["checks"]
        log.append({"persona": "analyst", "checks": board["checks"]})

    def designer():
        plan_llm = _PersonaLLM(llm, "designer")
        if board["plan"] is None:
            plan, raw = make_plan(requirement, board["checks"] or [], selected, plan_llm)
            if plan is None:
                plan, raw = make_plan(requirement, board["checks"] or [], selected, plan_llm, tag="planner.retry")
            if plan is None:
                log.append({"persona": "designer", "error": "no plan"})
                return
            _mark_meaning(plan, board["checks"] or [])
        else:                                     # revision after the critic (or lint), before any run
            issues = (board["critique"] or []) + lint_plan(board["plan"], ops)
            if not issues:
                return
            raw = plan_llm.complete(REPAIR_PROMPT.format(issues="\n".join(f"- {i}" for i in issues),
                                                         operations=ops_text, plan=json.dumps(board["plan"], indent=1)),
                                    max_tokens=1800, tag="revise")
            plan = parse_plan(raw)
            if plan is None:
                return
            _mark_meaning(plan, board["checks"] or [])
            if board["runs"]:                     # after a run, the guard applies to every revision
                plan, ev = guard(board["plan"], plan, board["runs"][-1]["report"], spec_codes,
                                 lambda p: field_problems(p, ops))
                log.append({"persona": "designer", "guard": ev})
        plan, ev = _spec_cleanup(plan, ops, cfg, requirement, auth)
        board["plan"], board["critique"] = plan, None
        board["revised_since_run"] = True
        log.append({"persona": "designer", "cleanup": ev})

    def critic():
        if board["plan"] is None:
            return
        raw = _with_persona(llm, "critic", CRITIC_PROMPT.format(requirement=requirement, operations=ops_text,
                                                                plan=json.dumps(board["plan"], indent=1)),
                            500, "critic.review")
        items = [l[2:].strip() for l in raw.splitlines() if l.strip().startswith("- ")]
        board["critique"] = items if items and "NO PROBLEMS" not in raw.upper() else []
        log.append({"persona": "critic", "raw": raw})

    def executor():
        plan = board["plan"]
        if plan is None or len(board["runs"]) >= MAX_ROUNDS:
            return
        while True:
            test_file.write_text(compile_plan(plan, service, base_url, requirement, ops, auth))
            passed, output, report = run_pytest(test_file)
            report = report or {"steps": [], "checks": [], "setup_errors": [f"no report: {output[-300:]}"]}
            failed_ids = {c["id"] for c in report.get("checks", []) if not c["passed"]}
            failed = [c for c in plan.get("checks", []) if c.get("id") in failed_ids]
            relax = _relaxable_format_checks(failed, plan, report, ops) if not report.get("setup_errors") else []
            if relax and not board["relaxed"] and cfg.get("format_relaxation", True):
                plan = dict(plan, checks=[c for c in plan["checks"] if c.get("id") not in relax])
                board["relaxed"] = True
                continue
            break
        board["plan"] = plan
        board["runs"].append({"passed": passed, "report": report})
        board["review"] = None
        board["revised_since_run"] = False
        log.append({"persona": "executor", "passed": passed, "report": report})
        if passed:
            board["done"] = True

    def reviewer():
        if not board["runs"] or board["runs"][-1]["passed"]:
            return
        report = board["runs"][-1]["report"]
        if report.get("setup_errors"):
            board["review"] = {"verdict": "TEST_BUG", "reason": "setup error: " + "; ".join(report["setup_errors"])}
        else:
            board["review"] = judge(cfg["judge"], _PersonaLLM(llm, "reviewer"), requirement, ops_text,
                                    board["plan"], report, hide_values=hide)
        log.append({"persona": "reviewer", "review": board["review"]})
        if board["review"]["verdict"] == "API_BUG":
            board["done"] = True

    def debugger():
        if not board["review"] or board["review"]["verdict"] != "TEST_BUG":
            return
        report = board["runs"][-1]["report"]
        new_plan, info = refine(cfg["refinement"], _PersonaLLM(llm, "debugger"), requirement, ops_text,
                                board["plan"], report, review=_review_for_debugger(board["review"], report, content),
                                spec_codes=spec_codes, spec_problems=lambda p: field_problems(p, ops))
        log.append({"persona": "debugger", "events": info["events"]})
        board["review"] = None
        if new_plan is not None:
            _mark_meaning(new_plan, board["checks"] or [])
            board["plan"], ev = _spec_cleanup(new_plan, ops, cfg, requirement, auth)
            board["revised_since_run"] = True

    act = {"analyst": analyst, "designer": designer, "critic": critic, "executor": executor,
           "reviewer": reviewer, "debugger": debugger}

    if cfg["architecture"] == "multi_persona":
        analyst(); designer(); critic(); designer()
        while not board["done"] and len(board["runs"]) < MAX_ROUNDS and board["plan"] is not None:
            executor()
            if board["done"] or cfg["refinement"] == "none":
                break
            reviewer()
            if board["done"] or len(board["runs"]) >= MAX_ROUNDS:
                break
            debugger()
    else:                                         # blackboard with a controller persona
        for step in range(MAX_CONTROLLER_STEPS):
            options = _options(board, cfg)
            if options == ["finish"]:
                break
            summary = _board_summary(board)
            raw = llm.complete(CONTROLLER_PROMPT.format(requirement=requirement, board=summary,
                                                        options=", ".join(options)), max_tokens=10,
                               tag="controller.pick")
            pick = next((o for o in options if o in raw.lower()), options[0])
            log.append({"controller": step, "options": options, "pick": pick})
            if pick == "finish":
                break
            act[pick]()
            if board["done"]:
                break

    final = board["plan"]
    if final is None:
        raise ValueError("multi-agent team produced no plan")
    history["final_plan"] = final
    history["derived_checks"] = board["checks"]
    history["rounds"].append({"round": len(board["runs"]), "plan": final,
                              "decision": f"{cfg['architecture']}_done"})
    test_file.write_text(compile_plan(final, service, base_url, requirement, ops, auth))


def _options(board: dict, cfg: dict) -> list[str]:
    """Which personas may act now (the rules the controller cannot break)."""
    opts = []
    if board["checks"] is None:
        return ["analyst"]
    if board["plan"] is None:
        return ["designer"]
    if not board["runs"]:
        opts += ["critic"] if board["critique"] is None else []
        opts += ["designer"] if board["critique"] else []
        opts += ["executor"]
        return opts
    last = board["runs"][-1]
    if last["passed"]:
        return ["finish"]
    if board["review"] is None and cfg["refinement"] != "none":
        opts.append("reviewer")
    if board["review"] and board["review"]["verdict"] == "TEST_BUG" and len(board["runs"]) < MAX_ROUNDS:
        opts.append("debugger")
    if board["review"] is None and len(board["runs"]) < MAX_ROUNDS and board.get("revised_since_run"):
        opts.append("executor")
    opts.append("finish")
    return opts


def _board_summary(board: dict) -> str:
    lines = [f"operations: {', '.join(board['operations'])}",
             f"checks: {len(board['checks'] or [])} derived" if board["checks"] is not None else "checks: none yet",
             "plan: written" if board["plan"] else "plan: none yet"]
    if board["critique"] is not None:
        lines.append(f"critique: {len(board['critique'])} problems" if board["critique"] else "critique: no problems")
    for i, r in enumerate(board["runs"], 1):
        failed = [c["id"] for c in r["report"].get("checks", []) if not c["passed"]]
        lines.append(f"run {i}: {'passed' if r['passed'] else 'failed checks ' + ', '.join(failed)}"
                     + (" (setup error)" if r["report"].get("setup_errors") else ""))
    if board["review"]:
        lines.append(f"review: {board['review']['verdict']}")
    return "\n".join(lines)
