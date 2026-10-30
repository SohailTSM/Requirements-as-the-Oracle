"""Compile a structured test plan into a self-contained pytest file.

Deterministic, spec-driven mechanics (the part small models get wrong):
  - authentication recipe inferred from the OpenAPI security scheme and login/registration operations
  - request content type and envelopes ({"user": {...}}) taken from the operation's schema
  - missing required fields filled for setup steps
The generated test embeds agent/test_runtime.py, so it needs only `requests` and `pytest`.
"""
from __future__ import annotations

import json
import pprint
import re
from pathlib import Path

from .spec import Operation, load_operations, spec_path

RUNTIME_SOURCE = (Path(__file__).parent / "test_runtime.py").read_text()
TOKEN_KEYS = ["access_token", "accessToken", "token"]


# ---------------------------------------------------------------- auth recipe --
def _body_schema(op: Operation | None) -> tuple[str | None, dict]:
    if not op or not op.request_body:
        return None, {}
    for ctype in ("application/json", "application/x-www-form-urlencoded"):
        if ctype in op.request_body:
            return ctype, op.request_body[ctype] or {}
    ctype, schema = next(iter(op.request_body.items()))
    return ctype, schema or {}


def envelope_of(schema: dict) -> str | None:
    """{"user": {...}} style wrapper: exactly one required property, which is an object."""
    props, req = schema.get("properties", {}), schema.get("required", [])
    if len(req) == 1 and len(props) == 1:
        inner = props.get(req[0], {})
        if inner.get("type") == "object" or "properties" in inner:
            return req[0]
    return None


def _inner(schema: dict) -> dict:
    env = envelope_of(schema)
    return schema["properties"][env] if env else schema


def infer_auth(service: str, ops: list[Operation] | None = None) -> dict:
    ops = ops or load_operations(service)
    spec = json.loads(spec_path(service).read_text())
    schemes = (spec.get("components") or {}).get("securitySchemes") or {}
    scheme = next(iter(schemes.values()), {})
    by_path = {(o.method, o.url_path): o for o in ops}
    base = ops[0].url_path[: len(ops[0].url_path) - len(ops[0].path)] if ops else ""

    prefix, login = "Bearer", None
    if scheme.get("type") == "oauth2":
        token_url = scheme.get("flows", {}).get("password", {}).get("tokenUrl", "")
        login = by_path.get(("POST", base + token_url)) or by_path.get(("POST", token_url))
    elif scheme.get("type") == "apiKey":
        m = re.search(r"^\s*(\w+)\s+x{3,}", scheme.get("description", ""), re.M)
        prefix = m.group(1) if m else ""
    if login is None:
        login = next((o for o in ops if o.method == "POST" and "login" in o.url_path.lower()
                      and "test" not in o.url_path.lower()), None)

    def has_password(o):
        _, s = _body_schema(o)
        return "password" in json.dumps(_inner(s).get("properties", {})).lower()

    reg_candidates = [o for o in ops if o.method == "POST" and not o.auth_required and has_password(o)
                      and o is not login and "login" not in o.url_path.lower()
                      and "reset" not in o.url_path.lower() and "private" not in o.url_path.lower()]
    reg_candidates.sort(key=lambda o: (not any(k in o.url_path.lower() for k in ("signup", "register")), len(o.url_path)))
    register = reg_candidates[0] if reg_candidates else None

    login_ct, login_schema = _body_schema(login)
    reg_ct, reg_schema = _body_schema(register)
    login_fields = [f for f in _inner(login_schema).get("required", []) or list(_inner(login_schema).get("properties", {}))
                    if f != "grant_type"]
    reg_inner = _inner(reg_schema)
    reg_fields = list(dict.fromkeys(reg_inner.get("required", []) + [f for f in reg_inner.get("properties", {})
                                                                      if f in ("username", "email", "password")]))
    me = next((o for o in ops if o.method == "GET" and o.auth_required and
               (o.url_path.rstrip("/").endswith("/me") or o.url_path.rstrip("/").endswith("/user"))), None)
    return {
        "header_prefix": prefix,
        "login_path": login.url_path if login else None,
        "login_content_type": login_ct or "application/json",
        "login_envelope": envelope_of(login_schema),
        "login_fields": login_fields,
        "login_username_is_email": "username" in login_fields and "username" not in reg_fields,
        "register_path": register.url_path if register else None,
        "register_envelope": envelope_of(reg_schema),
        "register_fields": reg_fields,
        "token_keys": TOKEN_KEYS,
        "me_path": me.url_path if me else None,
    }


# ------------------------------------------------------------- plan completion --
def _placeholder_for(name: str, prop: dict) -> object:
    fmt, typ, low = prop.get("format"), prop.get("type"), name.lower()
    if fmt == "email" or "email" in low:
        return "{{rand.email}}"
    if "password" in low:
        return "{{rand.password}}"
    if fmt == "uuid":
        return "{{rand.uuid}}"
    if typ == "integer" or typ == "number":
        return 1
    if typ == "boolean":
        return False
    if typ == "array":
        return []
    return "test {{rand.str}}"


def complete_plan(plan: dict, ops: list[Operation]) -> dict:
    """Fill mechanics from the spec without touching what the plan chose to test."""
    plan = json.loads(json.dumps(plan))
    by_id = {o.op_id: o for o in ops}
    for step in plan.get("steps", []):
        method, _, path = step.get("call", "").partition(" ")
        op = by_id.get(f"{method.upper()} {path}")
        if op is None:  # tolerate trailing-slash / case differences
            op = next((o for o in ops if o.method == method.upper() and o.url_path.rstrip("/") == path.rstrip("/")), None)
        if op is None and "{{" in path:  # "GET /items/{{s1.id}}" -> template + path parameter
            concrete = re.sub(r"\{\{[^}]+\}\}", "X", path)
            for o in ops:
                if o.method == method.upper() and o.matches(o.method, concrete):
                    names = re.findall(r"\{([^}]+)\}", o.url_path)
                    values = re.findall(r"\{\{[^}]+\}\}", path)
                    if len(names) == len(values):
                        step.setdefault("path", {}).update(dict(zip(names, values)))
                        op = o
                    break
        if op is None:
            step["unknown_operation"] = True
            continue
        step["call"] = op.op_id
        ctype, schema = _body_schema(op)
        if ctype:
            step.setdefault("content_type", ctype)
        env = envelope_of(schema) if schema else None
        body = step.get("body")
        if env and isinstance(body, dict) and env not in body:
            step["body"] = {env: body}
        if step.get("role") == "setup" and schema:
            inner = step["body"][env] if env and isinstance(step.get("body"), dict) else step.get("body")
            if inner is None:
                inner = {}
                step["body"] = {env: inner} if env else inner
            target = _inner(schema)
            for name in target.get("required", []):
                if name not in inner:
                    inner[name] = _placeholder_for(name, target.get("properties", {}).get(name, {}))
    return plan


# ----------------------------------------------------------------- code emit --
TEST_TEMPLATE = '''"""
Generated by the PreTrained agent.
Requirement: {requirement}
"""
{runtime}

BASE_URL = {base_url!r}
AUTH = {auth}
PLAN = {plan}


def test_requirement():
    run_plan(PLAN, AUTH, BASE_URL)
'''


JUDGED_TEMPLATE = '''"""
Generated by the PreTrained B4 baseline (LLM response judge, LogiAgent-style oracle).
Requirement: {requirement}
"""
{runtime}

BASE_URL = {base_url!r}
AUTH = {auth}
PLAN = {plan}
JUDGE = {judge}


def test_requirement():
    run_plan_judged(PLAN, AUTH, BASE_URL, JUDGE)
'''


def compile_judged_plan(plan: dict, service: str, base_url: str, requirement: str, ops_text: str,
                        judge_url: str, judge_model: str, ops: list[Operation] | None = None,
                        auth: dict | None = None) -> str:
    """B4: the plan's steps plus an LLM verdict at test time instead of fixed checks."""
    ops = ops or load_operations(service)
    auth = auth or infer_auth(service, ops)
    full = complete_plan(dict(plan, checks=[]), ops)
    judge = {"url": judge_url, "model": judge_model, "requirement": requirement, "operations": ops_text}
    return JUDGED_TEMPLATE.format(
        requirement=requirement.replace('"""', "'''"), runtime=RUNTIME_SOURCE, base_url=base_url,
        auth=pprint.pformat(auth, indent=1, width=100, sort_dicts=False),
        plan=pprint.pformat(full, indent=1, width=100, sort_dicts=False),
        judge=pprint.pformat(judge, indent=1, width=100, sort_dicts=False))


def compile_plan(plan: dict, service: str, base_url: str, requirement: str = "",
                 ops: list[Operation] | None = None, auth: dict | None = None) -> str:
    ops = ops or load_operations(service)
    auth = auth or infer_auth(service, ops)
    full = complete_plan(plan, ops)
    return TEST_TEMPLATE.format(
        requirement=requirement.replace('"""', "'''"),
        runtime=RUNTIME_SOURCE,
        base_url=base_url,
        auth=pprint.pformat(auth, indent=1, width=100, sort_dicts=False),
        plan=pprint.pformat(full, indent=1, width=100, sort_dicts=False),
    )
