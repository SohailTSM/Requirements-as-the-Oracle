"""A2A agents for the communication variant "a2a" (Decision Log D-67).

    python tools_ext/a2a_agents.py --role reviewer --port 9101
    python tools_ext/a2a_agents.py --role debugger --port 9102

The Reviewer (evidence judge) and the Debugger (restricted refinement) run as separate A2A agents
(JSON-RPC over HTTP, local). The orchestrator sends one JSON payload per request as a text message
and receives one JSON text message back. The agents call the same functions as the in-process
pipeline, and return their LLM call records so token accounting stays complete.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402

from a2a.helpers.proto_helpers import get_message_text, new_text_message  # noqa: E402
from a2a.server.agent_execution import AgentExecutor  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill  # noqa: E402

from agent.llm import LLM  # noqa: E402
from agent.lint import field_problems  # noqa: E402
from agent.refine import judge, refine  # noqa: E402
from agent.spec import load_operations  # noqa: E402

_ops = {}


def _work(role: str, payload: dict) -> dict:
    llm = LLM()
    if role == "reviewer":
        out = judge(payload["judge"], llm, payload["requirement"], payload["ops_text"], payload["plan"],
                    payload["report"], payload["hide_values"])
    else:
        svc = payload["service"]
        if svc not in _ops:
            _ops[svc] = load_operations(svc)
        new_plan, info = refine(payload["refinement"], llm, payload["requirement"], payload["ops_text"],
                                payload["plan"], payload["report"], review=payload.get("review", ""),
                                spec_codes=set(payload.get("spec_codes", [])),
                                spec_problems=(lambda p: field_problems(p, _ops[svc])) if payload.get("spec_cleanup", True) else None)
        out = {"plan": new_plan, "info": info}
    return {"result": out, "llm_calls": llm.calls}


class Executor(AgentExecutor):
    def __init__(self, role):
        self.role = role

    async def execute(self, context, event_queue):
        import asyncio
        payload = json.loads(get_message_text(context.message))
        result = await asyncio.to_thread(_work, self.role, payload)
        await event_queue.enqueue_event(new_text_message(json.dumps(result, default=str),
                                                         context_id=context.context_id, task_id=context.task_id))

    async def cancel(self, context, event_queue):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["reviewer", "debugger"], required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/"
    desc = {"reviewer": "Decides whether a failed API test found a bug (API_BUG) or is wrong (TEST_BUG).",
            "debugger": "Repairs how an API test plan runs without changing what it expects."}[a.role]
    card = AgentCard(name=f"pretrained-{a.role}", description=desc, version="1.0",
                     supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")],
                     capabilities=AgentCapabilities(streaming=False),
                     default_input_modes=["text/plain"], default_output_modes=["text/plain"],
                     skills=[AgentSkill(id=a.role, name=a.role, description=desc, tags=["api-testing"])])
    handler = DefaultRequestHandler(agent_executor=Executor(a.role), task_store=InMemoryTaskStore(), agent_card=card)
    app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/"))
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
