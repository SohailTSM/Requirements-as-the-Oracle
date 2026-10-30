"""Static plan checks against the spec (no service call, so no risk of learning a bug).

Problems found here are sent back to the planner once before the first run. After that repair,
checks that the spec shows can never pass are dropped deterministically (drop_impossible), missing
path parameters are filled from earlier steps (autofill_path_params), and format status checks
with codes the spec does not document are dropped. All of this uses the spec only.
"""
from __future__ import annotations

import re

from .spec import Operation, find_operation, normalize_field, response_schema, schema_fields

_REF = re.compile(r"\{\{\s*([A-Za-z0-9_\.\-\[\]]+)\s*\}\}")
EXPECTATIONS = {"status", "status_in", "status_class", "field_equals", "field_not_equals", "field_exists",
                "field_absent", "field_contains", "field_not_contains", "list_contains_item",
                "list_not_contains_item", "count_equals", "field_startswith"}


def _refs(obj):
    if isinstance(obj, str):
        return _REF.findall(obj)
    if isinstance(obj, dict):
        return [r for v in obj.values() for r in _refs(v)]
    if isinstance(obj, list):
        return [r for v in obj for r in _refs(v)]
    return []


def step_operations(plan: dict, ops: list[Operation]) -> dict:
    """step id -> Operation (None if the call is not in the spec)."""
    by_id = {o.op_id: o for o in ops}
    out = {}
    for step in plan.get("steps", []):
        method, _, path = str(step.get("call", "")).partition(" ")
        out[step.get("id")] = by_id.get(f"{method.upper()} {path}") or find_operation(
            ops, method, _PLACEHOLDER_SEG.sub("x", path.split("?")[0]))
    return out


_PLACEHOLDER_SEG = re.compile(r"\{\{[^}]*\}\}")
_TEMPLATE_PARAM = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")


def template_params(call: str) -> list[str]:
    """{param} templates still present in a call string (embedded {{refs}} are concrete values)."""
    return _TEMPLATE_PARAM.findall(str(call).partition(" ")[2])


def error_fields(ops: list[Operation]) -> set[str]:
    """Fields documented for any error (4xx/5xx) response in the spec, e.g. "detail"."""
    out = set()
    for o in ops:
        for code, r in o.responses.items():
            if code[:1] in "45":
                out |= set(schema_fields(r.get("schema")))
    return out


_FAIL_STATUS = ("status", "status_in", "status_class")


def _expected_status(plan: dict, step_id: str):
    """The status a plan expects from a step: an exact code, "4xx"-style class, or None."""
    for c in plan.get("checks", []):
        if not isinstance(c, dict) or c.get("step") != step_id or c.get("expect") not in _FAIL_STATUS:
            continue
        v = c.get("value")
        if isinstance(v, list) and v:
            v = v[0]
        return str(v)
    return None


def field_problems(plan: dict, ops: list[Operation]) -> dict:
    """check id -> reason the check can never pass on a service that follows the spec.

    Only judged when the spec documents the response schema; otherwise nothing is reported."""
    step_ops = step_operations(plan, ops)
    steps = {s.get("id"): s for s in plan.get("steps", [])}
    err_fields = error_fields(ops)
    problems = {}
    for c in plan.get("checks", []):
        if not isinstance(c, dict):
            continue
        cid, step_id, exp = c.get("id"), c.get("step"), c.get("expect")
        op = step_ops.get(step_id)
        if op is None or exp in _FAIL_STATUS:
            continue
        if exp in ("field_equals", "field_not_equals") and isinstance(c.get("value"), str):
            if c["value"].replace(" ", "") == "{{%s.%s}}" % (step_id, c.get("field")):
                problems[cid] = f"compares {step_id}.{c.get('field')} with itself"
                continue
        if exp == "field_not_equals" and isinstance(c.get("value"), str) and _REF.fullmatch(c["value"].strip()):
            path_vals = {str(v).replace(" ", "") for v in (steps.get(step_id, {}).get("path") or {}).values()}
            if c["value"].replace(" ", "") in path_vals:
                problems[cid] = (f"{step_id} was called with {c['value']} in its path, so its "
                                 f"{c.get('field')} equals that value by construction")
                continue
        field = c.get("field")
        if not field or exp == "field_absent":
            continue
        expected = _expected_status(plan, step_id)
        if expected and expected[0] in "45":
            schema = response_schema(op, expected if expected.isdigit() else None)
            documented = set(schema_fields(schema)) | err_fields
            if normalize_field(field) not in documented:
                problems[cid] = (f"reads '{field}' from {step_id}, which is expected to fail ({expected}); "
                                 f"error responses in the specification have only: {', '.join(sorted(err_fields)) or 'no fields'}")
            continue
        schema = response_schema(op)
        fields = schema_fields(schema) if schema else {}
        if not fields:
            continue
        norm = normalize_field(field)
        if exp in ("list_contains_item", "list_not_contains_item", "count_equals"):
            ok = fields.get(norm) == "array" or any(f.startswith(norm + "[]") for f in fields)
        else:
            ok = norm in fields
        if not ok:
            top = sorted(f for f in fields if "[]" not in f or exp.startswith("list"))[:15]
            problems[cid] = (f"'{field}' is not a field of the {op.op_id} response; its fields are: "
                             + ", ".join(top))
    return problems


def lint_plan(plan: dict, ops: list[Operation]) -> list[str]:
    issues = []
    by_id = {o.op_id: o for o in ops}
    step_ops = step_operations(plan, ops)
    actors = set((plan.get("actors") or {}).keys())
    seen_steps, step_body = [], {}
    for step in plan.get("steps", []):
        sid = step.get("id", "?")
        call = step.get("call", "")
        op = step_ops.get(sid)
        if op is None:
            issues.append(f"step {sid}: operation '{call}' is not in the API; use one of: "
                          + ", ".join(sorted(by_id)))
        else:
            needed = set(template_params(call))
            given = set((step.get("path") or {}).keys())
            if needed - given:
                issues.append(f"step {sid}: path parameter(s) {sorted(needed - given)} of {op.op_id} are not given")
            if step.get("role") == "setup" and op.method in ("DELETE", "GET", "PUT", "PATCH"):
                for v in (step.get("path") or {}).values():
                    if isinstance(v, str) and "rand." in v:
                        issues.append(f"step {sid}: a setup step acts on a random id; create the resource first "
                                      f"(with a POST) and use its id")
        if step.get("actor") and step["actor"] not in actors:
            issues.append(f"step {sid}: actor '{step['actor']}' is not defined in actors")
        for ref in _refs([step.get("path"), step.get("body"), step.get("query")]):
            head, _, rest = ref.partition(".")
            if head in ("rand", "var") or head in actors:
                continue
            if head not in seen_steps:
                issues.append(f"step {sid}: references {{{{{ref}}}}} but step {head} has not run before it")
            elif rest.startswith("req") and not rest.startswith("req.") and rest != "req":
                issues.append(f"step {sid}: {{{{{ref}}}}} is not valid; use {{{{{head}.req.<field>}}}} for a field "
                              f"{head} sent, or {{{{{head}.<field>}}}} for its response (e.g. {{{{{head}.data[0].id}}}})")
            elif rest.startswith("req.") and not step_body.get(head):
                issues.append(f"step {sid}: {{{{{ref}}}}} reads the request body of {head}, but {head} sent no body; "
                              f"read its response instead with {{{{{head}.<field>}}}}")
            elif rest.startswith("req."):
                key = rest[4:].split(".")[0]
                body = step_body.get(head) or {}
                flat = body if not (len(body) == 1 and isinstance(next(iter(body.values())), dict)) else next(iter(body.values()))
                if key not in body and key not in flat:
                    issues.append(f"step {sid}: {{{{{ref}}}}} refers to field '{key}' that step {head} did not send; "
                                  f"use {{{{{head}.{key}}}}} to read it from {head}'s response")
        seen_steps.append(sid)
        step_body[sid] = step.get("body") if isinstance(step.get("body"), dict) else {}
    step_ids = set(seen_steps)
    for c in plan.get("checks", []):
        if not isinstance(c, dict):
            continue
        cid = c.get("id", "?")
        if c.get("step") not in step_ids:
            issues.append(f"check {cid}: step '{c.get('step')}' does not exist")
        if c.get("expect") not in EXPECTATIONS:
            issues.append(f"check {cid}: unknown expectation '{c.get('expect')}'")
        if c.get("expect") in ("list_contains_item", "list_not_contains_item") and not c.get("key"):
            issues.append(f"check {cid}: {c.get('expect')} needs a 'key' (the item field to compare, e.g. \"id\")")
        if c.get("expect") in ("field_equals", "field_not_equals") and isinstance(c.get("value"), str):
            if c["value"].replace(" ", "") == "{{%s.%s}}" % (c.get("step"), c.get("field")):
                issues.append(f"check {cid}: compares {c.get('step')}.{c.get('field')} with itself; it can never "
                              f"check anything. Compare with an earlier step or a request value instead")
        for ref in _refs(c.get("value")):
            head = ref.split(".")[0]
            if head not in ("rand", "var") and head not in actors and head not in step_ids:
                issues.append(f"check {cid}: value references unknown '{head}'")
    for cid, why in field_problems(plan, ops).items():
        if not any(i.startswith(f"check {cid}:") and "itself" in i for i in issues):
            issues.append(f"check {cid}: {why}")
    issues += [f"step {sid}: {why}" for sid, why in ref_problems(plan, ops).items()]
    return issues


def ref_problems(plan: dict, ops: list[Operation]) -> dict:
    """step id -> reason: a {{sN.field}} used in a request names a field sN's response does not have."""
    step_ops = step_operations(plan, ops)
    out = {}
    for step in plan.get("steps", []):
        for ref in _refs([step.get("path"), step.get("body"), step.get("query")]):
            head, _, rest = ref.partition(".")
            op = step_ops.get(head)
            if op is None or not rest or rest.startswith("req") or rest == "status":
                continue
            fields = schema_fields(response_schema(op))
            if fields and normalize_field(rest) not in fields:
                leaf = rest.split(".")[-1]
                hint = [f for f in fields if f.split(".")[-1] == leaf and "[]" not in f]
                out[step.get("id")] = (f"{{{{{ref}}}}}: {op.op_id} returns no field '{rest}'"
                                       + (f"; use {{{{{head}.{hint[0]}}}}}" if hint else ""))
    return out


def drop_impossible(plan: dict, ops: list[Operation]) -> tuple[dict, list[dict]]:
    """Remove checks the spec shows can never pass, and format status checks whose code the spec
    does not document for that operation. Spec-only: no service output is used."""
    problems = field_problems(plan, ops)
    step_ops = step_operations(plan, ops)
    kept, events = [], []
    for c in plan.get("checks", []):
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid in problems:
            events.append({"type": "impossible_check_dropped", "check": cid, "reason": problems[cid]})
            continue
        op = step_ops.get(c.get("step"))
        if (c.get("kind") == "format" and c.get("expect") in ("status", "status_in") and op is not None
                and op.responses):
            vals = c.get("value") if isinstance(c.get("value"), list) else [c.get("value")]
            if not all(str(v) in op.responses for v in vals):
                events.append({"type": "undocumented_status_format_check_dropped", "check": cid,
                               "documented": sorted(op.responses)})
                continue
        kept.append(c)
    return dict(plan, checks=kept), events


def autofill_path_params(plan: dict, ops: list[Operation]) -> tuple[dict, list[dict]]:
    """Fill missing path parameters from the latest earlier step whose documented response has a
    matching field (param name, or "id" for id-like params). Spec-only and deterministic."""
    step_ops = step_operations(plan, ops)
    events, earlier = [], []
    for step in plan.get("steps", []):
        op = step_ops.get(step.get("id"))
        if op is not None:
            needed = template_params(step.get("call", ""))
            given = step.get("path") if isinstance(step.get("path"), dict) else {}
            for p in needed:
                if given.get(p) not in (None, ""):
                    continue
                names = [p] + (["id"] if p.lower().endswith("id") or p.lower() == "id" else [])
                # latest creating step first, then any other earlier step
                order = sorted(reversed(earlier), key=lambda st: str(st.get("call", "")).upper()[:4] != "POST")
                for prev in order:
                    pop = step_ops.get(prev.get("id"))
                    fields = schema_fields(response_schema(pop)) if pop else {}
                    match = next((f for n in names for f in sorted(fields, key=len)
                                  if "[]" not in f and f.split(".")[-1] == n), None)
                    if match:
                        given[p] = "{{%s.%s}}" % (prev["id"], match)
                        events.append({"type": "path_param_autofilled", "step": step.get("id"),
                                       "param": p, "value": given[p]})
                        break
            if given:
                step["path"] = given
        earlier.append(step)
    return plan, events


REPAIR_PROMPT = """The test plan below has problems found by checking it against the API specification.
Fix ONLY these problems. Keep every check's expectation and expected value unchanged.

Problems:
{issues}

API operations you may call:
{operations}

Plan:
{plan}

Write the corrected plan JSON only."""


def define_missing_actors(plan: dict, requirement: str = "") -> tuple[dict, list[dict]]:
    """An actor used by a step but never defined becomes a fresh registered user ("new_user"):
    plans name a second user B/other to mean exactly that. An "existing" actor whose email the
    requirement does not quote has made-up credentials, so it also becomes a new user (the plan
    format allows existing accounts only when the requirement gives them). No service output used."""
    actors = plan.get("actors") if isinstance(plan.get("actors"), dict) else {}
    events = []
    for name, spec in list(actors.items()):
        if isinstance(spec, dict) and spec.get("type") == "existing":
            email = str(spec.get("email") or "")
            if not email or email.lower() not in requirement.lower():
                actors[name] = {"type": "new_user"}
                events.append({"type": "invented_credentials_replaced", "actor": name, "email": email})
    for step in plan.get("steps", []):
        a = step.get("actor")
        if a and a not in actors:
            actors[a] = {"type": "new_user"}
            events.append({"type": "actor_defined_as_new_user", "actor": a})
    plan["actors"] = actors
    return plan, events


def fix_misdirected_refs(plan: dict, ops: list[Operation]) -> tuple[dict, list[dict]]:
    """{{sN.req.f}} where sN sent no field f but its documented response has f -> {{sN.f}}, and
    {{sN.f}} where the response has no f but sN's request body sent f -> {{sN.req.f}}.
    Uses only the spec and the plan's own request bodies."""
    step_ops = step_operations(plan, ops)
    bodies = {s.get("id"): (s.get("body") if isinstance(s.get("body"), dict) else {}) for s in plan.get("steps", [])}
    events = []

    def fix(ref: str) -> str:
        head, _, rest = ref.partition(".")
        op = step_ops.get(head)
        if op is None or not rest or rest == "status":
            return ref
        fields = schema_fields(response_schema(op))
        body = bodies.get(head) or {}
        flat = next(iter(body.values())) if len(body) == 1 and isinstance(next(iter(body.values())), dict) else body
        if rest.startswith("req."):
            key = rest[4:]
            if key.split(".")[0] not in body and key.split(".")[0] not in flat and normalize_field(key) in fields:
                return f"{head}.{key}"
        elif fields and normalize_field(rest) not in fields and rest.split(".")[0] in flat:
            return f"{head}.req.{rest}"
        elif fields and normalize_field(rest) not in fields:
            # "{{s1.resp.id}}", "{{s1.response.id}}", "{{s1.body.id}}": wrapper words the runtime does not use
            first, _, remainder = rest.partition(".")
            if first in ("resp", "response", "body", "json", "result") and normalize_field(remainder) in fields:
                return f"{head}.{remainder}"
        return ref

    def walk(obj, where):
        if isinstance(obj, str):
            def sub(m):
                new = fix(m.group(1))
                if new != m.group(1):
                    events.append({"type": "reference_redirected", "where": where, "from": m.group(1), "to": new})
                return "{{" + new + "}}"
            return _REF.sub(sub, obj)
        if isinstance(obj, dict):
            return {k: walk(v, where) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, where) for v in obj]
        return obj

    for step in plan.get("steps", []):
        for k in ("path", "body", "query"):
            if k in step:
                step[k] = walk(step[k], step.get("id"))
    for c in plan.get("checks", []):
        if isinstance(c, dict) and "value" in c:
            c["value"] = walk(c["value"], c.get("id"))
    return plan, events


_SIGNUP_WORDS = re.compile(r"sign(ing)?[ -]?up|registration|register(ing)?\s+(a|an|new|with)\b|"
                           r"creat\w*\s+(a|an)\s+(new\s+)?(user|account)|new account|log(ging)?[ -]?in\b|login",
                           re.I)


def remove_redundant_signup(plan: dict, requirement: str, auth: dict | None) -> tuple[dict, list[dict]]:
    """Actors are registered and logged in automatically (plan format). A register step in a plan
    whose requirement is not about signing up or logging in only creates an extra, unrelated user;
    the plan then means its actor. Remove the step and point its references at the actor."""
    reg = (auth or {}).get("register_path")
    if not reg or _SIGNUP_WORDS.search(requirement):
        return plan, []
    events, keep, removed = [], [], {}
    for step in plan.get("steps", []):
        method, _, path = str(step.get("call", "")).partition(" ")
        if method.upper() == "POST" and path.rstrip("/") == reg.rstrip("/") and step.get("actor"):
            removed[step.get("id")] = step["actor"]
            events.append({"type": "redundant_signup_removed", "step": step.get("id"), "actor": step["actor"]})
        else:
            keep.append(step)
    if not removed:
        return plan, []
    actor_fields = {"id": "id", "email": "email", "password": "password", "username": "username",
                    "req.email": "email", "req.password": "password", "req.username": "username"}

    def rewrite(obj):
        if isinstance(obj, str):
            def sub(m):
                head, _, rest = m.group(1).partition(".")
                if head in removed and rest in actor_fields:
                    return "{{%s.%s}}" % (removed[head], actor_fields[rest])
                return m.group(0)
            return _REF.sub(sub, obj)
        if isinstance(obj, dict):
            return {k: rewrite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rewrite(v) for v in obj]
        return obj

    for step in keep:
        for k in ("path", "body", "query"):
            if k in step:
                step[k] = rewrite(step[k])
    checks = []
    for c in plan.get("checks", []):
        if isinstance(c, dict) and c.get("step") in removed:
            events.append({"type": "check_on_removed_signup_dropped", "check": c.get("id")})
            continue
        if isinstance(c, dict) and "value" in c:
            c["value"] = rewrite(c["value"])
        checks.append(c)
    return dict(plan, steps=keep, checks=checks), events


def generalize_unstated_status(plan: dict, requirement: str) -> tuple[dict, list[dict]]:
    """A meaning check may fix an exact status code only if the requirement states that code;
    otherwise the meaning is the status class (the catalogue's rule: exact codes are format)."""
    events = []
    for c in plan.get("checks", []):
        if not isinstance(c, dict) or c.get("kind", "meaning") != "meaning" or c.get("expect") != "status":
            continue
        code = str(c.get("value", ""))
        if code.isdigit() and not re.search(rf"\b{code}\b", requirement):
            c["expect"], c["value"] = "status_class", f"{code[0]}xx"
            events.append({"type": "unstated_status_generalized", "check": c.get("id"), "from": int(code),
                           "to": c["value"]})
    return plan, events
