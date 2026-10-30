#!/usr/bin/env python3
"""Gold operations: which API operations each requirement's reference test calls.

Parses RESTestBench's reference tests (harness/data/tests/<service>/test_<id>.py) with the
Python AST, reconstructs every requests.<method>(url, ...) URL template, and matches it to
the frozen spec. Output: gold/operations.json and a list of calls that could not be matched.
Used as ground truth for retrieval metrics (Recall@k, MRR, nDCG).
"""
import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from agent.spec import load_operations, find_operation  # noqa: E402
from common.paths import HARNESS_ROOT, SERVICES  # noqa: E402

METHODS = {"get", "post", "put", "patch", "delete"}


def _template(node, env: dict) -> str | None:
    """Turn a URL expression into a template string; unknown parts become '{x}'."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = ""
        for v in node.values:
            if isinstance(v, ast.Constant):
                out += str(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = v.value
                if isinstance(inner, ast.Name) and inner.id in env and env[inner.id] is not None:
                    out += env[inner.id]
                else:
                    out += "{x}"
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template(node.left, env), _template(node.right, env)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _strip_base(url: str) -> str:
    url = re.sub(r"^https?://[^/]+", "", url)       # literal host
    url = re.sub(r"^\{x\}", "", url)                 # {BASE_URL} or similar
    return url if url.startswith("/") else "/" + url


def extract_calls(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    env: dict[str, str | None] = {}
    # Module- and function-level string assignments (BASE_URL, url = f"...").
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.upper() in ("BASE_URL", "API_URL", "BASE", "HOST"):
                env[name] = "{x}"
            else:
                t = _template(node.value, env)
                if t is not None:
                    env[name] = t
    calls = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method = node.func.attr.lower()
        if method not in METHODS:
            continue
        url_node = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "url"), None)
        if url_node is None:
            continue
        t = _template(url_node, env)
        if t is None:
            continue
        calls.append((method.upper(), _strip_base(t)))
    # ast.walk is breadth-first; sort by source position to keep call order.
    return calls


def ordered_calls(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    positions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.lower() in METHODS:
            positions.setdefault(node.func.attr.lower(), []).append((node.lineno, node.col_offset))
    calls = extract_calls(source)
    return calls  # order is approximate; gold metrics use the set of operations


def main() -> int:
    gold, unmatched = {}, []
    for service in SERVICES:
        ops = load_operations(service)
        gold[service] = {}
        for test_file in sorted((HARNESS_ROOT / "data/tests" / service).glob("test_*.py"),
                                key=lambda p: int(p.stem.split("_")[1])):
            req_id = int(test_file.stem.split("_")[1])
            found = []
            for method, url in extract_calls(test_file.read_text()):
                op = find_operation(ops, method, url)
                if op is None:
                    unmatched.append({"service": service, "requirement": req_id, "method": method, "url": url})
                elif op.op_id not in found:
                    found.append(op.op_id)
            gold[service][str(req_id)] = found
    out = PROJECT_ROOT / "gold"
    out.mkdir(exist_ok=True)
    (out / "operations.json").write_text(json.dumps(gold, indent=1))
    (out / "operations_unmatched.json").write_text(json.dumps(unmatched, indent=1))
    for s, reqs in gold.items():
        empty = [r for r, o in reqs.items() if not o]
        sizes = [len(o) for o in reqs.values()]
        print(f"{s}: {len(reqs)} requirements, ops per requirement avg {sum(sizes)/len(sizes):.1f} "
              f"(max {max(sizes)}), empty: {empty}")
    print(f"unmatched calls: {len(unmatched)}")
    for u in unmatched[:15]:
        print("  ", u)
    return 0


if __name__ == "__main__":
    sys.exit(main())
