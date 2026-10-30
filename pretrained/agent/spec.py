"""Load a service's OpenAPI spec into a list of operations.

The spec used everywhere is `harness/services/<service>/spec.json`, which ships with
RESTestBench and describes the *correct* implementation; it never changes during
mutation runs (the "frozen spec").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import jsonref

from common.paths import HARNESS_ROOT

HTTP_METHODS = ("get", "post", "put", "patch", "delete")


@dataclass
class Operation:
    service: str
    method: str                 # "POST"
    path: str                   # path as written in the spec, e.g. "/users/{user_id}"
    url_path: str               # path including the server's base path, e.g. "/api/users/{user_id}"
    summary: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    parameters: list = field(default_factory=list)     # [{name, in, required, schema}]
    request_body: dict | None = None                  # {content_type: schema}
    responses: dict = field(default_factory=dict)     # {status: {description, schema}}
    auth_required: bool = False

    @property
    def op_id(self) -> str:
        return f"{self.method} {self.url_path}"

    def matches(self, method: str, url_path: str) -> bool:
        if method.upper() != self.method:
            return False
        pattern = "^" + re.sub(r"\\\{[^/]+?\\\}", r"[^/]+", re.escape(self.url_path.rstrip("/"))) + "/?$"
        return re.match(pattern, url_path.split("?")[0].rstrip("/") or "/") is not None

    def to_text(self, with_schemas: bool = True) -> str:
        """Compact, human-readable rendering used for retrieval documents and prompts."""
        lines = [f"{self.op_id}"]
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.description and self.description != self.summary:
            lines.append(f"Description: {self.description.strip()}")
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        lines.append(f"Auth required: {'yes' if self.auth_required else 'no'}")
        for p in self.parameters:
            req = "required" if p.get("required") else "optional"
            lines.append(f"Parameter ({p.get('in')}, {req}): {p.get('name')} {_short(p.get('schema'))}")
        if self.request_body:
            for ctype, schema in self.request_body.items():
                lines.append(f"Request body ({ctype}): {_short(schema) if with_schemas else ''}")
        for status, r in self.responses.items():
            s = f"Response {status}: {r.get('description', '')}"
            if with_schemas and r.get("schema"):
                s += f" {_short(r['schema'])}"
            lines.append(s)
        return "\n".join(lines)


def _short(schema, depth: int = 0) -> str:
    """Compact one-line schema rendering (field names and types)."""
    if not isinstance(schema, dict):
        return ""
    if "anyOf" in schema or "oneOf" in schema:
        alts = [_short(s, depth) for s in schema.get("anyOf", schema.get("oneOf", []))]
        return " | ".join(a for a in alts if a)
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        if depth > 2:
            return "object"
        req = set(schema.get("required", []))
        props = ", ".join(f"{k}{'*' if k in req else ''}: {_short(v, depth + 1)}"
                          for k, v in schema.get("properties", {}).items())
        return "{" + props + "}"
    if t == "array":
        return f"[{_short(schema.get('items', {}), depth + 1)}]"
    fmt = schema.get("format")
    enum = schema.get("enum")
    out = t or ""
    if fmt:
        out += f"({fmt})"
    if enum:
        out += f" enum{list(enum)}"
    return out


def _plain(obj):
    """jsonref proxies -> plain dicts/lists (so results are JSON-serialisable)."""
    return json.loads(json.dumps(obj, default=lambda o: dict(o)))


def spec_path(service: str) -> Path:
    return HARNESS_ROOT / "services" / service / "spec.json"


def load_operations(service: str) -> list[Operation]:
    raw = json.loads(spec_path(service).read_text())
    spec = jsonref.replace_refs(raw, lazy_load=False, proxies=False)
    servers = spec.get("servers") or [{"url": ""}]
    base_path = urlparse(servers[0]["url"]).path.rstrip("/")
    global_security = bool(spec.get("security"))
    ops = []
    for path, item in spec["paths"].items():
        shared_params = item.get("parameters", [])
        for method in HTTP_METHODS:
            if method not in item:
                continue
            o = item[method]
            params = [
                {"name": p.get("name"), "in": p.get("in"), "required": p.get("required", False),
                 "schema": _plain(p.get("schema", {}))}
                for p in shared_params + o.get("parameters", [])
            ]
            body = None
            if "requestBody" in o:
                body = {ct: _plain(v.get("schema", {})) for ct, v in o["requestBody"].get("content", {}).items()}
            responses = {}
            for status, r in o.get("responses", {}).items():
                schema = None
                for v in (r.get("content") or {}).values():
                    schema = _plain(v.get("schema", {}))
                    break
                responses[str(status)] = {"description": r.get("description", ""), "schema": schema}
            security = o.get("security", None)
            auth = bool(security) if security is not None else global_security
            ops.append(Operation(
                service=service, method=method.upper(), path=path, url_path=base_path + path,
                summary=o.get("summary", "") or "", description=o.get("description", "") or "",
                tags=o.get("tags", []), parameters=params, request_body=body,
                responses=responses, auth_required=auth))
    return ops


def find_operation(ops: list[Operation], method: str, url_path: str) -> Operation | None:
    matches = [op for op in ops if op.matches(method, url_path)]
    if not matches:
        return None
    # Prefer the most specific template (fewest path parameters), e.g. /users/me over /users/{id}.
    return min(matches, key=lambda op: op.url_path.count("{"))


# ------------------------------------------------------------ response schemas --
def _type_of(schema) -> str:
    if not isinstance(schema, dict):
        return "any"
    alts = [s for s in schema.get("anyOf", []) + schema.get("oneOf", []) if s.get("type") != "null"]
    if alts:
        return _type_of(alts[0])
    if "properties" in schema:
        return "object"
    return schema.get("type") or "any"


def schema_fields(schema, prefix: str = "", depth: int = 0, out: dict | None = None) -> dict:
    """Dotted field paths a response can contain -> type. Array items appear as "data[].id"."""
    out = {} if out is None else out
    if not isinstance(schema, dict) or depth > 5:
        return out
    for alt in schema.get("anyOf", []) + schema.get("oneOf", []) + schema.get("allOf", []):
        schema_fields(alt, prefix, depth, out)
    if schema.get("type") == "array" and "items" in schema:
        schema_fields(schema["items"], prefix + "[]", depth + 1, out)
    for k, v in (schema.get("properties") or {}).items():
        path = f"{prefix}.{k}" if prefix else k
        out[path] = _type_of(v)
        schema_fields(v, path, depth + 1, out)
    return out


def normalize_field(path: str) -> str:
    """data[0].id / data.0.id -> data[].id (the form used by schema_fields)."""
    path = re.sub(r"\[-?\d+\]", "[]", str(path))
    return re.sub(r"\.(-?\d+)(?=\.|$)", "[]", path)


def response_schema(op: "Operation", status=None):
    """Schema documented for `status` (exact code), or for the first 2xx response. None if unknown."""
    if status is not None and str(status) in op.responses:
        return op.responses[str(status)].get("schema")
    if status is not None and str(status)[0] != "2":
        return None
    for code in sorted(op.responses):
        if code.startswith("2"):
            return op.responses[code].get("schema")
    return None
