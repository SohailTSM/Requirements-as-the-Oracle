# ---- PreTrained test runtime (embedded verbatim into every generated test) ----
# Executes a structured test plan against the service and evaluates its checks.
# Needs only `requests`, so generated tests stay self-contained for the harness.
#
# Plan format (written by the planner LLM, completed by the compiler):
#   actors: {name: {"type": "new_user"} | {"type": "existing", "email": ..., "password": ...}
#                  | {"type": "anonymous"}}
#   vars:   {name: "rand.email" | "rand.uuid" | "rand.str" | "rand.int" | literal}
#   steps:  [{"id": "s1", "actor": "A", "call": "POST /path/{id}", "path": {...}, "query": {...},
#             "body": {...}, "content_type": "application/json", "role": "setup|action|verify"}]
#   checks: [{"id": "C1", "step": "s2", "expect": "<kind>", "field": "a.b", "key": ..., "value": ...,
#             "kind": "meaning|format", "description": ...}]
# Values may contain placeholders: {{s1.<body path>}}, {{s1.req.<path>}}, {{s1.status}},
# {{A.email}} / {{A.password}} / {{A.username}} / {{A.id}} / {{A.token}}, {{var.<name>}}.
import json as _json
import os as _os
import random as _random
import re as _re
import string as _string
import uuid as _uuid

import requests as _requests

_PH = _re.compile(r"\{\{\s*([A-Za-z0-9_\.\-\[\]]+)\s*\}\}")


class SetupError(Exception):
    """The test could not run as planned (auth failed, a setup call failed, a value was missing)."""


def _rand(kind):
    tag = _uuid.uuid4().hex[:10]
    if kind == "email":
        return f"user_{tag}@example.com"
    if kind == "uuid":
        return str(_uuid.uuid4())
    if kind == "int":
        return _random.randint(100000, 999999)
    if kind == "username":
        return f"user_{tag}"
    if kind == "password":
        return "Pw_" + tag + "9a"
    return "".join(_random.choices(_string.ascii_lowercase, k=12))


def _get_path(obj, path):
    cur = obj
    path = _re.sub(r"\[(\-?\d+)\]", r".\1", str(path))          # data[0].id -> data.0.id
    for part in [p for p in path.split(".") if p != ""]:
        if isinstance(cur, list):
            if part.lstrip("-").isdigit() and -len(cur) <= int(part) < len(cur):
                cur = cur[int(part)]
            else:
                raise KeyError(path)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _find_key(obj, keys):
    """Depth-first search for the first value under any of `keys` (e.g. a token)."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] not in (None, ""):
                return obj[k]
        for v in obj.values():
            found = _find_key(v, keys)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_key(v, keys)
            if found is not None:
                return found
    return None


class _Run:
    def __init__(self, plan, auth, base_url):
        self.plan, self.auth, self.base = plan, auth, base_url.rstrip("/")
        self.vars = {}
        for name, spec in (plan.get("vars") or {}).items():
            self.vars[name] = _rand(spec[5:]) if isinstance(spec, str) and spec.startswith("rand.") else spec
        self.actors = {}
        self.steps = {}
        self.report = {"steps": [], "checks": [], "setup_errors": [], "actors": {}}

    # -- placeholders ---------------------------------------------------------------
    def _lookup(self, ref):
        head, _, rest = ref.partition(".")
        if head == "var":
            if rest not in self.vars:
                raise SetupError(f"unknown variable {ref}")
            return self.vars[rest]
        if head == "rand":
            return _rand(rest)
        if head in self.actors:
            if rest not in self.actors[head]:
                raise SetupError(f"actor {head} has no '{rest}'")
            return self.actors[head][rest]
        if head in self.steps:
            st = self.steps[head]
            if rest == "status":
                return st["status"]
            if rest.startswith("req.") or rest == "req":
                try:
                    return _get_path(st["request_body"], rest[4:])
                except (KeyError, TypeError):
                    raise SetupError(f"request body of {head} has no field '{rest[4:]}'")
            try:
                return _get_path(st["body"], rest)
            except (KeyError, TypeError):
                raise SetupError(f"response of {head} has no field '{rest}'")
        raise SetupError(f"cannot resolve {{{{{ref}}}}}")

    def resolve(self, value):
        if isinstance(value, str):
            m = _PH.fullmatch(value.strip())
            if m:
                return self._lookup(m.group(1))
            return _PH.sub(lambda mm: str(self._lookup(mm.group(1))), value)
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        return value

    # -- HTTP ----------------------------------------------------------------------
    def http(self, method, path, actor=None, body=None, query=None, content_type="application/json"):
        headers = {}
        if actor and self.actors.get(actor, {}).get("token"):
            headers["Authorization"] = f"{self.auth['header_prefix']} {self.actors[actor]['token']}".strip()
        kwargs = {"headers": headers, "params": query or None, "timeout": 30}
        if body is not None:
            if content_type == "application/x-www-form-urlencoded":
                kwargs["data"] = body
            else:
                kwargs["json"] = body
        resp = _requests.request(method, self.base + path, **kwargs)
        try:
            parsed = resp.json()
        except ValueError:
            parsed = resp.text
        return resp.status_code, parsed

    # -- actors --------------------------------------------------------------------
    def _wrap(self, body, envelope):
        return {envelope: body} if envelope else body

    def _login(self, name, email, username, password):
        a = self.auth
        fields = {}
        for f in a["login_fields"]:
            low = f.lower()
            fields[f] = password if "pass" in low else (email if ("email" in low or low == "username" and a.get("login_username_is_email")) else username or email)
        status, body = self.http("POST", a["login_path"], body=self._wrap(fields, a.get("login_envelope")),
                                 content_type=a.get("login_content_type", "application/json"))
        token = _find_key(body, a["token_keys"])
        if not (200 <= status < 300) or not token:
            raise SetupError(f"login for actor {name} failed: {status} {str(body)[:200]}")
        return token

    def setup_actor(self, name, spec):
        kind = (spec or {}).get("type", "new_user")
        info = {"email": None, "password": None, "username": None, "id": None, "token": None}
        if kind == "anonymous":
            self.actors[name] = info
            return
        if kind == "existing":
            info["email"], info["password"] = spec.get("email"), spec.get("password")
            info["username"] = spec.get("username")
        else:
            info["email"] = self.resolve(spec.get("email")) if spec.get("email") else _rand("email")
            info["password"] = self.resolve(spec.get("password")) if spec.get("password") else _rand("password")
            info["username"] = self.resolve(spec.get("username")) if spec.get("username") else _rand("username")
            reg = {}
            for f in self.auth["register_fields"]:
                low = f.lower()
                reg[f] = (info["password"] if "pass" in low else info["email"] if "email" in low
                          else info["username"] if ("user" in low or low == "name") else None)
            reg = {k: v for k, v in reg.items() if v is not None}
            for k, v in (spec.get("extra") or {}).items():
                reg[k] = self.resolve(v)
            status, body = self.http("POST", self.auth["register_path"], body=self._wrap(reg, self.auth.get("register_envelope")))
            if not (200 <= status < 300):
                raise SetupError(f"registration of actor {name} failed: {status} {str(body)[:200]}")
            info["id"] = _find_key(body, ["id"])
            info["token"] = _find_key(body, self.auth["token_keys"])
        if not info["token"]:
            info["token"] = self._login(name, info["email"], info["username"], info["password"])
        if info["id"] is None and self.auth.get("me_path"):
            self.actors[name] = info
            st, me = self.http("GET", self.auth["me_path"], actor=name)
            if 200 <= st < 300:
                info["id"] = _find_key(me, ["id"])
                info["username"] = info["username"] or _find_key(me, ["username"])
        self.actors[name] = info
        self.report["actors"][name] = {k: v for k, v in info.items() if k != "token"}

    # -- steps ---------------------------------------------------------------------
    def run_step(self, step):
        method, _, template = step["call"].partition(" ")
        path_params = self.resolve(step.get("path") or {})
        path = self.resolve(template)               # placeholders written directly in the call
        for k, v in path_params.items():
            path = path.replace("{" + k + "}", str(v))
        if "{" in path:
            raise SetupError(f"step {step['id']}: unfilled path parameter in {path}")
        body = self.resolve(step["body"]) if "body" in step and step["body"] is not None else None
        query = self.resolve(step.get("query") or {})
        actor = step.get("actor")
        if actor and actor not in self.actors:
            raise SetupError(f"step {step['id']}: unknown actor {actor}")
        status, resp = self.http(method, path, actor=actor, body=body, query=query,
                                 content_type=step.get("content_type", "application/json"))
        self.steps[step["id"]] = {"status": status, "body": resp, "request_body": body}
        self.report["steps"].append({"id": step["id"], "call": f"{method} {path}", "actor": actor,
                                     "role": step.get("role", "action"), "status": status,
                                     "request_body": body, "response": resp})
        if step.get("role") == "setup" and not (200 <= status < 300):
            raise SetupError(f"setup step {step['id']} ({method} {path}) returned {status}: {str(resp)[:200]}")

    # -- checks --------------------------------------------------------------------
    def evaluate(self, check):
        st = self.steps.get(check["step"])
        if st is None:
            return False, f"step {check['step']} did not run"
        exp = check["expect"]
        value = self.resolve(check.get("value")) if "value" in check else None
        status, body = st["status"], st["body"]

        def field():
            return _get_path(body, check.get("field", ""))

        if exp in ("list_contains_item", "list_not_contains_item") and not check.get("key"):
            return False, "list check has no 'key'"
        try:
            if exp == "status":
                return status == int(value), f"status {status}"
            if exp == "status_in":
                return status in [int(v) for v in value], f"status {status}"
            if exp == "status_class":
                return str(status)[0] == str(value)[0], f"status {status}"
            if exp == "field_equals":
                got = field()
                return got == value or str(got) == str(value), f"{check.get('field')} = {str(got)[:120]}"
            if exp == "field_not_equals":
                got = field()
                return got != value and str(got) != str(value), f"{check.get('field')} = {str(got)[:120]}"
            if exp == "field_exists":
                got = field()
                return got not in (None, "", [], {}), f"{check.get('field')} present"
            if exp == "field_absent":
                try:
                    field()
                    return False, f"{check.get('field')} present"
                except KeyError:
                    return True, "absent"
            if exp == "field_contains":
                got = field()
                return (value in got) if isinstance(got, (list, str)) else False, f"{check.get('field')} = {str(got)[:120]}"
            if exp == "field_not_contains":
                got = field()
                return (value not in got) if isinstance(got, (list, str)) else True, f"{check.get('field')} = {str(got)[:120]}"
            if exp in ("list_contains_item", "list_not_contains_item"):
                items = field() if check.get("field") else body
                found = any(isinstance(i, dict) and str(i.get(check["key"])) == str(value) for i in items or [])
                return (found if exp == "list_contains_item" else not found), f"{len(items or [])} items"
            if exp == "count_equals":
                got = field() if check.get("field") else body
                n = len(got) if isinstance(got, (list, dict, str)) else got
                return int(n) == int(value), f"count {n}"
            if exp == "field_startswith":
                got = field()
                return str(got).startswith(str(value)), f"{check.get('field')} = {str(got)[:120]}"
        except (KeyError, TypeError) as e:
            return False, f"field {e} missing in response"
        except (ValueError, IndexError) as e:
            return False, f"cannot evaluate: {e}"
        return False, f"unknown expectation {exp}"

    def run(self):
        try:
            for name, spec in (self.plan.get("actors") or {}).items():
                self.setup_actor(name, spec)
            for step in self.plan["steps"]:
                self.run_step(step)
        except SetupError as e:
            self.report["setup_errors"].append(str(e))
        except _requests.RequestException as e:
            self.report["setup_errors"].append(f"connection error: {e}")
        except Exception as e:                      # malformed plan: report it, never crash
            self.report["setup_errors"].append(f"{type(e).__name__}: {e}")
        failed = []
        if not self.report["setup_errors"]:
            for check in self.plan.get("checks", []):
                try:
                    ok, observed = self.evaluate(check)
                except SetupError as e:
                    ok, observed = False, f"setup: {e}"
                    self.report["setup_errors"].append(str(e))
                self.report["checks"].append({"id": check["id"], "passed": bool(ok), "observed": observed,
                                              "kind": check.get("kind", "meaning")})
                if not ok:
                    failed.append(f"{check['id']} ({check.get('description', check['expect'])}): {observed}")
        out = _os.environ.get("PRETRAINED_REPORT")
        if out:
            with open(out, "w") as f:
                _json.dump(self.report, f, default=str)
        if self.report["setup_errors"]:
            raise AssertionError("setup error: " + "; ".join(self.report["setup_errors"]))
        if failed:
            raise AssertionError("failed checks: " + " | ".join(failed))


def run_plan(plan, auth, base_url):
    _Run(plan, auth, base_url).run()


# -- B4 baseline: LLM response judge (LogiAgent-style oracle) ----------------------------------
_JUDGE_PROMPT = """You validate whether a REST API behaves as its requirement says.

Requirement: "{requirement}"

API operations (the specification):
{operations}

The test made these calls (request and full response):
{trace}

Does the observed behaviour satisfy the requirement? Judge the service, not the test.
Answer in exactly one line: PASS: <short reason>  or  FAIL: <which part of the requirement is violated>"""


def _judge_trace(report):
    lines = []
    for st in report["steps"]:
        req = _json.dumps(st.get("request_body"), default=str)[:400]
        resp = _json.dumps(st.get("response"), default=str)[:1200]
        lines.append(f"{st['id']} {st['call']} as {st.get('actor')}\n  request: {req}\n"
                     f"  -> status {st['status']}, response: {resp}")
    return "\n".join(lines)


def run_plan_judged(plan, auth, base_url, judge):
    """Run the plan's steps (setup must succeed), then ask the LLM judge for a verdict.
    The verdict is computed at test time, so every run (correct service, each mutant) is judged."""
    r = _Run(dict(plan, checks=[]), auth, base_url)
    r.run()                                    # raises on setup errors (no checks to evaluate)
    prompt = _JUDGE_PROMPT.format(requirement=judge["requirement"], operations=judge["operations"],
                                  trace=_judge_trace(r.report))
    resp = _requests.post(judge["url"].rstrip("/") + "/chat/completions", timeout=300, json={
        "model": judge["model"], "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 0, "max_tokens": 200})
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    r.report["judge"] = answer
    out = _os.environ.get("PRETRAINED_REPORT")
    if out:
        with open(out, "w") as f:
            _json.dump(r.report, f, default=str)
    head = answer.upper().lstrip("*# ")
    if not head.startswith("PASS"):
        raise AssertionError("response judge: " + answer[:300])
# ---- end of runtime ----
