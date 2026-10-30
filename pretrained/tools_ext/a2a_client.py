"""Orchestrator side of the "a2a" variant: starts the two agent processes and talks to them."""
from __future__ import annotations

import asyncio
import atexit
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

AGENTS = Path(__file__).resolve().parent / "a2a_agents.py"
PORTS = {"reviewer": 9101, "debugger": 9102}


class A2ATeam:
    def __init__(self):
        self.timings = []
        log = open(AGENTS.parent.parent / "results" / "a2a_agents.log", "a")
        self.procs = [subprocess.Popen([sys.executable, str(AGENTS), "--role", r, "--port", str(p)],
                                       stdout=log, stderr=log, cwd=str(AGENTS.parent.parent))
                      for r, p in PORTS.items()]
        for r, p in PORTS.items():
            deadline = time.time() + 180
            while True:
                try:
                    if requests.get(f"http://127.0.0.1:{p}/.well-known/agent-card.json", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                if time.time() > deadline:
                    raise RuntimeError(f"A2A agent {r} did not start")
                time.sleep(1)
        atexit.register(self.close)              # agents must not outlive the run (they hold ports)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self.clients = {}

    async def _send(self, role: str, payload: dict) -> dict:
        from a2a.client.client_factory import create_client
        from a2a.helpers.proto_helpers import get_stream_response_text, new_text_message
        from a2a.types import SendMessageRequest
        if role not in self.clients:
            import httpx
            from a2a.client.client import ClientConfig
            # one LLM call can take minutes while the GPU is shared; the default HTTP timeout is far shorter
            cfg = ClientConfig(streaming=False, httpx_client=httpx.AsyncClient(timeout=900))
            self.clients[role] = await create_client(f"http://127.0.0.1:{PORTS[role]}", client_config=cfg)
        text = ""
        req = SendMessageRequest(message=new_text_message(json.dumps(payload, default=str), role=1))
        async for resp in self.clients[role].send_message(req):
            text += get_stream_response_text(resp)
        return json.loads(text)

    def ask(self, role: str, payload: dict) -> dict:
        t0 = time.time()
        out = asyncio.run_coroutine_threadsafe(self._send(role, payload), self._loop).result(timeout=900)
        self.timings.append({"agent": role, "seconds": round(time.time() - t0, 3)})
        return out

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
