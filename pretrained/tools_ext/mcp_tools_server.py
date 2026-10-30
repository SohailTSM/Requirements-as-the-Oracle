"""MCP server exposing the agent's non-LLM tools (communication variant "mcp", Decision Log D-65).

    python tools_ext/mcp_tools_server.py        # stdio transport; started by tools_ext/mcp_client.py

Tools: search_operations (retrieval), check_plan (spec lint), run_test (execute a compiled test).
The same functions the pipeline calls in-process; only the transport differs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from agent.lint import lint_plan  # noqa: E402
from agent.retrieval import Retriever  # noqa: E402
from agent.spec import load_operations  # noqa: E402
from common.harness import run_pytest  # noqa: E402

server = MCPServer("pretrained-tools", instructions="Spec search, plan lint and test execution for API tests.")
_retrievers, _ops = {}, {}


def _svc(service: str):
    if service not in _ops:
        _ops[service] = load_operations(service)
        _retrievers[service] = Retriever(service)
    return _ops[service], _retrievers[service]


@server.tool()
def search_operations(service: str, query: str, variant: str = "hybrid_rerank", k: int = 5) -> str:
    """Rank the service's API operations for a query; returns a JSON list of operation ids."""
    _, r = _svc(service)
    ranked = r.rank(query, variant)
    return json.dumps([o.op_id for o in (ranked if variant == "full" else ranked[:k])])


@server.tool()
def check_plan(service: str, plan: str) -> str:
    """Lint a test plan (JSON) against the service's specification; returns a JSON list of issues."""
    ops, _ = _svc(service)
    return json.dumps(lint_plan(json.loads(plan), ops))


@server.tool()
def run_test(test_file: str) -> str:
    """Run one compiled test file; returns JSON {passed, output, report}."""
    passed, output, report = run_pytest(Path(test_file))
    return json.dumps({"passed": passed, "output": output[-4000:], "report": report}, default=str)


if __name__ == "__main__":
    server.run("stdio")
