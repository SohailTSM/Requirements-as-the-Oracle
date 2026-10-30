"""Planner: requirement + derived checks + retrieved operations -> structured test plan (JSON).

The LLM decides WHAT to call and WHAT to check; the compiler/runtime handle HOW
(authentication, envelopes, content types, ids). The example uses a made-up library API.
"""
from __future__ import annotations

import json
import re

from .spec import Operation

PLAN_FORMAT = """\
Plan JSON format:
{
 "actors": {"<name>": {"type": "new_user"}                      (registered and logged in automatically)
            | {"type": "existing", "email": "...", "password": "..."}   (only if the requirement gives credentials)
            | {"type": "anonymous"}},                            (no token)
 "steps": [{"id": "s1", "actor": "<actor name or omit>", "call": "<METHOD> <path exactly as listed>",
            "path": {"<param>": <value>}, "query": {...}, "body": {...},
            "role": "setup" | "action" | "verify"}],
 "checks": [{"id": "C1", "step": "<step id>", "expect": "<expectation>", "field": "<dotted path in the response>",
             "key": "<item key for list checks>", "value": <expected value>,
             "kind": "meaning" | "format", "description": "<short text>"}]
}
Expectations: status (exact code), status_class ("2xx"/"4xx"), field_equals, field_not_equals, field_exists,
field_absent, field_contains, field_not_contains, list_contains_item, list_not_contains_item, count_equals,
field_startswith.
Values can reference earlier results: "{{s1.id}}" (field of step s1's response), "{{s1.req.title}}"
(field of the request body sent in s1), "{{A.email}}", "{{A.id}}", "{{A.password}}" (actor data),
"{{rand.uuid}}", "{{rand.email}}", "{{rand.str}}" (random values).
Login and registration happen automatically for actors: do NOT add login or signup steps unless the
requirement is about logging in or signing up itself.
"setup" steps prepare data and must succeed; "action" is the call the requirement is about; "verify"
steps read state afterwards (for example to confirm something was saved or deleted)."""

EXAMPLE = """\
Example (library API)
Requirement: "A member must not be able to cancel someone else's reservation."
Checks to implement:
- C1 (meaning): member B cancelling member A's reservation is rejected (4xx)
- C2 (meaning): member A's reservation still exists afterwards
Plan:
{"actors": {"A": {"type": "new_user"}, "B": {"type": "new_user"}},
 "steps": [
  {"id": "s1", "actor": "A", "call": "POST /api/reservations", "body": {"book_id": 7}, "role": "setup"},
  {"id": "s2", "actor": "B", "call": "DELETE /api/reservations/{id}", "path": {"id": "{{s1.id}}"}, "role": "action"},
  {"id": "s3", "actor": "A", "call": "GET /api/reservations/{id}", "path": {"id": "{{s1.id}}"}, "role": "verify"}],
 "checks": [
  {"id": "C1", "step": "s2", "expect": "status_class", "value": "4xx", "kind": "meaning", "description": "B cannot cancel A's reservation"},
  {"id": "C2", "step": "s3", "expect": "status", "value": 200, "kind": "meaning", "description": "reservation still exists"},
  {"id": "C3", "step": "s3", "expect": "field_equals", "field": "id", "value": "{{s1.id}}", "kind": "meaning", "description": "same reservation returned"}]}"""

PLANNER_PROMPT = """You are an expert API tester. Write a structured test plan for the requirement below.

{plan_format}

{example}

API operations you may call (use these exact method and path strings):
{operations}

Requirement: "{requirement}"
Checks to implement:
{checks}

Write the plan JSON only, no explanations."""


def _checks_text(checks: list[dict]) -> str:
    if not checks:
        return "- (derive the checks yourself from the requirement)"
    return "\n".join(f"- {c['id']} ({c['kind']}): {c['description']}" for c in checks)


_BROKEN_PH = re.compile(r"\{\{([^{}\"]+)\}(?!\})")       # "{{s1.id}"  -> "{{s1.id}}"


def parse_plan(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    stack, end, in_str, esc = [], None, False, False
    for i, ch in enumerate(text[start:], start):       # brackets inside JSON strings do not count
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            if not stack:
                end = i + 1
                break
    # an answer that stops before closing everything (e.g. the last "}") is closed for it
    raw = text[start:end] if end else text[start:].rstrip().rstrip(",") + "".join(reversed(stack))
    raw = _BROKEN_PH.sub(lambda m: "{{" + m.group(1) + "}}", raw)
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):   # tolerate trailing commas
        try:
            plan = json.loads(candidate)
            if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
                plan.setdefault("actors", {})
                plan.setdefault("checks", [])
                return plan
        except Exception:
            continue
    return None


def make_plan(requirement: str, checks: list[dict], ops: list[Operation], llm,
              tag: str = "planner") -> tuple[dict | None, str]:
    prompt = PLANNER_PROMPT.format(plan_format=PLAN_FORMAT, example=EXAMPLE,
                                   operations="\n\n".join(o.to_text() for o in ops),
                                   requirement=requirement, checks=_checks_text(checks))
    raw = llm.complete(prompt, max_tokens=1800, tag=tag)
    return parse_plan(raw), raw
