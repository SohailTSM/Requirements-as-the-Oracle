"""Refinement after running a compiled test on the (possibly buggy) service, plus the judge.

Refinement variants (component sweep):
    none             no refinement: the first plan is final
    unrestricted     the LLM sees everything (incl. observed values) and may rewrite the whole plan
    prompt_fixed     sees observed values, told not to copy them; no structural guard
    hidden           failed checks are reported WITHOUT observed values; no guard
    strengthen_only  observed values visible; guard rejects weakened/changed locked checks + copied values
    combined         hidden values + guard + evidence judge (our method)

Judge variants:
    none             every failed check is treated as a test bug (refine until rounds run out)
    restestbench     RESTestBench-style judge (sees observed values; REFINEMENT or FINISH)
    evidence         must justify TEST_BUG with requirement/spec evidence; API_BUG ends the loop
"""
from __future__ import annotations

import copy
import json

from .planner import PLAN_FORMAT, parse_plan

REFINE_VARIANTS = ["none", "unrestricted", "prompt_fixed", "hidden", "strengthen_only", "combined"]
JUDGE_VARIANTS = ["none", "restestbench", "evidence"]

LOCKED_FIELDS = ("step", "expect", "field", "key", "value")


def _trim(obj, n=1500) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def _shape(obj, depth: int = 0) -> str:
    """Field names of a response without any values (lists show their item shape, never a count)."""
    if isinstance(obj, dict):
        if depth >= 2:
            return "{...}"
        return "{" + ", ".join(f"{k}: {_shape(v, depth + 1)}" if isinstance(v, (dict, list)) else str(k)
                               for k in list(obj)[:20] for v in [obj[k]]) + "}"
    if isinstance(obj, list):
        return f"[{_shape(obj[0], depth + 1)}]" if obj else "[]"
    return "text" if isinstance(obj, str) else "value"


def _trace(report: dict, hide_values: bool) -> str:
    """Calls and status codes. With hidden values only field names are shown, except the error body
    of a failed setup step (it explains how the request was malformed, e.g. a missing field)."""
    lines = []
    errors = " ".join(report.get("setup_errors", []))
    for st in report.get("steps", []):
        line = f"{st['id']} {st['call']} as {st.get('actor')} -> status {st['status']}"
        failed_setup = f"step {st['id']}" in errors and not str(st.get("status", "")).startswith("2")
        if hide_values and failed_setup:
            line += f"\n  error response: {_trim(st.get('response'), 400)}"
        elif hide_values:
            line += f"\n  response fields: {_shape(st.get('response'))}"
        else:
            line += f"\n  request: {_trim(st.get('request_body'), 400)}\n  response: {_trim(st.get('response'), 800)}"
        lines.append(line)
    return "\n".join(lines) or "(no steps ran)"


def _failed_checks(report: dict, plan: dict, hide_values: bool) -> str:
    by_id = {c["id"]: c for c in plan.get("checks", [])}
    out = []
    for r in report.get("checks", []):
        if r["passed"]:
            continue
        c = by_id.get(r["id"], {})
        desc = c.get("description") or c.get("expect")
        line = f"{r['id']} ({c.get('kind', 'meaning')}): {desc} [expect {c.get('expect')} on {c.get('step')}.{c.get('field', '')}]"
        if not hide_values:
            line += f" -> observed: {r['observed']}"
        else:
            line += " -> FAILED"
        out.append(line)
    return "\n".join(out) or "(none)"


# ---------------------------------------------------------------- the judge --
EVIDENCE_JUDGE = """You review a failed API test. The service under test MAY CONTAIN BUGS; a test that
fails because the service violates the requirement is a GOOD test.

Requirement: "{requirement}"

API operations (the specification):
{operations}

Test plan:
{plan}

What happened (calls made and their status codes):
{trace}

Failed checks:
{failed}

Decide whether the failure is a TEST_BUG (the plan calls the wrong operation, uses a field name the
specification does not have, references the wrong step, or misses a setup step) or an API_BUG (the
service does not do what the requirement says). A check is also a TEST_BUG if it could never pass
on a correct service (for example it compares a field with itself, or reads a field that the
specification's response for that operation does not contain). You may only answer TEST_BUG if you
can quote the requirement or the specification to show the plan is wrong. What the service returned
is never evidence that the expectation is wrong.

Answer in exactly one line, one of:
TEST_BUG: <quote from the requirement or specification> | <what to change in the plan>
API_BUG: <which requirement the service violates>"""

RESTESTBENCH_JUDGE = """You are a test quality reviewer. Analyze the test execution results and determine if the test is correct or needs refinement.
The service with which you are interacting with could have bugs, so the test might be valid even if it fails.

Requirement: "{requirement}"

API operations:
{operations}

Test plan:
{plan}

Calls made and responses:
{trace}

Failed checks:
{failed}

Respond with EXACTLY one of:
REFINEMENT: <short description of what needs to be changed>
FINISH: <short description why the test is correct>"""


def judge(variant: str, llm, requirement: str, ops_text: str, plan: dict, report: dict, hide_values: bool) -> dict:
    """Return {"verdict": "TEST_BUG"|"API_BUG", "reason": str, "raw": str}."""
    if variant == "none":
        return {"verdict": "TEST_BUG", "reason": "no judge: always refine", "raw": ""}
    tmpl = EVIDENCE_JUDGE if variant == "evidence" else RESTESTBENCH_JUDGE
    raw = llm.complete(tmpl.format(requirement=requirement, operations=ops_text, plan=_trim(plan, 3000),
                                   trace=_trace(report, hide_values), failed=_failed_checks(report, plan, hide_values)),
                       max_tokens=300, tag=f"judge.{variant}").strip()
    upper = raw.upper()
    if variant == "evidence":
        if "API_BUG" in upper and ("TEST_BUG" not in upper or upper.find("API_BUG") < upper.find("TEST_BUG")):
            return {"verdict": "API_BUG", "reason": raw, "raw": raw}
        has_quote = "|" in raw or '"' in raw or "'" in raw
        return {"verdict": "TEST_BUG" if has_quote else "API_BUG", "reason": raw, "raw": raw,
                "evidence_missing": not has_quote}
    # restestbench-style: FINISH means keep the test
    if "FINISH" in upper and "REFINEMENT" not in upper:
        return {"verdict": "API_BUG", "reason": raw, "raw": raw}
    return {"verdict": "TEST_BUG", "reason": raw, "raw": raw}


# ------------------------------------------------------------- the refiner --
REFINE_PROMPT = """You are fixing an API test plan that did not run as intended.

{plan_format}

Requirement: "{requirement}"

API operations you may call:
{operations}

Current plan:
{plan}

What happened:
{trace}

Setup errors:
{setup_errors}

Failed checks:
{failed}

{review}
{instructions}
Write the corrected plan JSON only."""

INSTRUCTIONS = {
    "unrestricted": "Use the information above to improve the test so it is correct.",
    "prompt_fixed": ("Fix only how the test runs (calls, actors, path parameters, request bodies, setup steps). "
                     "The service may be buggy: do NOT change an expected value just because the service returned "
                     "something different."),
    "hidden": ("Fix only how the test runs (calls, actors, path parameters, request bodies, setup steps). "
               "Keep every check's expected value."),
}


def _is_ref(v) -> bool:
    return isinstance(v, str) and "{{" in v


def _locked_view(check: dict) -> tuple:
    """What may never change for a meaning check: the step it is about, the expectation type,
    the list key, and any LITERAL expected value. Field paths and {{references}} may be
    repaired (copied buggy values are always literals)."""
    value = check.get("value")
    return (check.get("step"), check.get("expect"), check.get("key"),
            "<ref>" if _is_ref(value) else json.dumps(value, sort_keys=True))


def _values_in(obj, acc: set) -> set:
    if isinstance(obj, dict):
        for v in obj.values():
            _values_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _values_in(v, acc)
    elif obj is not None and not isinstance(obj, bool) and str(obj) != "":
        acc.add(str(obj))
    return acc


def guard(old_plan: dict, new_plan: dict, report: dict, spec_codes: set[str],
          spec_problems=None) -> tuple[dict, list[dict]]:
    """Strengthen-only guard + value-copy detector.

    - Locked (meaning) checks are restored exactly if the new plan changed or dropped them.
    - New or changed expected values equal to a value seen in the last responses are rejected
      (unless it is a status code documented in the spec, for format checks).
    - Adding checks is allowed. Returns (guarded_plan, events)."""
    events = []
    observed = set()
    for st in report.get("steps", []):
        _values_in(st.get("response"), observed)
        observed.add(str(st.get("status")))
    old_checks = {c["id"]: c for c in old_plan.get("checks", [])}
    new_checks = {c.get("id"): c for c in new_plan.get("checks", []) if isinstance(c, dict)}
    result = []
    for cid, old in old_checks.items():
        new = new_checks.get(cid)
        if old.get("kind", "meaning") == "meaning":
            if new is None or _locked_view(new) != _locked_view(old):
                events.append({"type": "locked_check_restored", "check": cid,
                               "attempted": new, "copied_value": bool(new and str(new.get("value")) in observed
                                                                       and str(new.get("value")) != str(old.get("value")))})
                result.append(old)
            else:
                if new.get("field") != old.get("field") or new.get("value") != old.get("value"):
                    events.append({"type": "locked_check_reference_repaired", "check": cid})
                result.append(dict(new, kind="meaning"))
            continue
        if new is None:
            events.append({"type": "format_check_dropped", "check": cid})
            continue
        if json.dumps(new.get("value")) != json.dumps(old.get("value")) and str(new.get("value")) in observed:
            if new.get("expect") in ("status", "status_in") and str(new.get("value")) in spec_codes:
                events.append({"type": "format_check_changed_with_spec_evidence", "check": cid})
                result.append(new)
            else:
                events.append({"type": "value_copy_blocked", "check": cid, "attempted": new})
                result.append(old)
            continue
        result.append(new)
    for cid, new in new_checks.items():
        if cid not in old_checks:
            if str(new.get("value")) in observed and new.get("value") not in (None, "") and not str(new.get("value")).startswith("{{"):
                events.append({"type": "new_check_with_copied_value_blocked", "check": cid, "attempted": new})
                continue
            events.append({"type": "check_added", "check": cid})
            result.append(new)
    guarded = dict(new_plan, checks=result)
    if spec_problems is not None:
        # A reference/field repair of a meaning check must land on a field the spec documents;
        # otherwise a mutant that renames or drops a field could be "repaired" into passing.
        bad = spec_problems(guarded)
        repaired = {e["check"] for e in events if e["type"] == "locked_check_reference_repaired"}
        if bad and repaired & set(bad):
            guarded["checks"] = [old_checks[c["id"]] if c.get("id") in repaired & set(bad) else c
                                 for c in guarded["checks"]]
            events += [{"type": "field_repair_rejected_not_in_spec", "check": cid, "reason": bad[cid]}
                       for cid in sorted(repaired & set(bad))]
    return guarded, events


def refine(variant: str, llm, requirement: str, ops_text: str, plan: dict, report: dict,
           review: str = "", spec_codes: set[str] | None = None, spec_problems=None) -> tuple[dict | None, dict]:
    """One refinement round. Returns (new_plan or None if parsing failed, info)."""
    hide = variant in ("hidden", "combined")
    instr = INSTRUCTIONS.get(variant, INSTRUCTIONS["hidden"] if hide else INSTRUCTIONS["prompt_fixed"])
    raw = llm.complete(REFINE_PROMPT.format(
        plan_format=PLAN_FORMAT, requirement=requirement, operations=ops_text, plan=json.dumps(plan, indent=1),
        trace=_trace(report, hide_values=hide),
        setup_errors="\n".join(report.get("setup_errors", [])) or "(none)",
        failed=_failed_checks(report, plan, hide_values=hide),
        review=f"Reviewer's note: {review}\n" if review else "", instructions=instr),
        max_tokens=1800, tag=f"refine.{variant}")
    new_plan = parse_plan(raw)
    info = {"raw": raw, "events": []}
    if new_plan is None:
        return None, info
    if variant in ("strengthen_only", "combined"):
        new_plan, info["events"] = guard(plan, new_plan, report, spec_codes or set(), spec_problems)
        info["guarded_plan"] = new_plan            # the "chosen" side of a DPO pair (D-54)
    return new_plan, info
