"""Synchronous client for tools_ext/mcp_tools_server.py (one server process per tool instance).

The MCP session lives in a background thread with its own event loop; calls from the (synchronous)
pipeline are submitted to it. Every call's wall time is recorded for the overhead comparison.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent / "mcp_tools_server.py"


class MCPTools:
    def __init__(self):
        self.timings = []
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop = None
        self._error = None
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._open(), self._loop)
        if not self._ready.wait(180):
            raise RuntimeError(f"MCP server did not start: {self._error}")
        if self._error:
            raise RuntimeError(f"MCP server failed: {self._error}")

    async def _open(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(SERVER.parent.parent))
        self._stop = asyncio.Event()
        try:
            errlog = open(SERVER.parent.parent / "results" / "mcp_server.log", "a")   # keep the server's logs out of the run log
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as e:                       # surface start-up errors to the constructor
            self._error = f"{type(e).__name__}: {e}"
            self._ready.set()

    def _call(self, name: str, args: dict):
        t0 = time.time()
        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, args), self._loop)
        res = fut.result(timeout=900)
        self.timings.append({"tool": name, "seconds": round(time.time() - t0, 3)})
        text = "".join(getattr(c, "text", "") for c in (getattr(res, "content", None) or []))
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise RuntimeError(f"MCP tool {name} failed: {text[:500]}")
        return json.loads(text)

    def search_operations(self, service, query, variant, k):
        return self._call("search_operations", {"service": service, "query": query, "variant": variant, "k": k})

    def check_plan(self, service, plan: dict):
        return self._call("check_plan", {"service": service, "plan": json.dumps(plan)})

    def run_test(self, test_file):
        out = self._call("run_test", {"test_file": str(test_file)})
        return out["passed"], out["output"], out["report"]

    def close(self):
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
