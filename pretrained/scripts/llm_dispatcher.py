#!/usr/bin/env python3
"""Local LLM dispatcher: one request queue in front of every model server (Decision Log D-101).

    .venv/bin/python scripts/llm_dispatcher.py [--port 11500]
    # every run then uses PRETRAINED_LLM_URL=http://127.0.0.1:11500/v1

All runs send their OpenAI-compatible requests here. Requests wait in one FIFO queue; whenever a
model server has a free slot, it takes the oldest request (a conversation returns to the server it
last used when that server is free, to reuse its prompt cache). If a server fails mid-request
(connection lost, Cloudflare 52x/530/1016), the request goes back to the front of the queue and
another server takes it; the failed server is checked again every 30 s.

Servers come from secrets/llm_backends.json (re-read every 30 s; add or remove links any time):
    {"backends": [{"name": "sohail-A1", "url": "https://.../<secret>/v1", "slots": 2,
                   "valid_until": "2026-10-26T19:05:00Z"}, ...]}
`name` appears in logs; the URL (with its secret) never does.

GET /status : queue length, waiting ages, per-server slots/busy/served/errors/alive (JSON)
GET /health : 200 if at least one server is usable
Per-request log: results/dispatcher_requests.jsonl (time, server name, wait, latency, status, tokens)
"""
import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

ROOT = Path(__file__).resolve().parent.parent
BACKENDS_FILE = Path(os.environ.get("PRETRAINED_BACKENDS_FILE", ROOT.parent / "secrets" / "llm_backends.json"))
LOG_FILE = Path(os.environ.get("PRETRAINED_DISPATCH_LOG", ROOT / "results" / "dispatcher_requests.jsonl"))
MAX_WAIT_S = float(os.environ.get("DISPATCH_MAX_WAIT_S", "540"))      # below the clients' 600 s timeout
REQUEST_TIMEOUT_S = float(os.environ.get("DISPATCH_REQUEST_TIMEOUT_S", "900"))
MAX_ATTEMPTS = 4                     # a request is tried on at most this many servers
END_MARGIN_S = 120                   # no new request to a server in its last 2 minutes
RETRY_STATUS = {502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 530}


class Backend:
    def __init__(self, name, url, slots, valid_until):
        self.name, self.url, self.slots = name, url.rstrip("/"), max(1, int(slots))
        self.valid_until = valid_until
        self.busy = 0
        self.alive = True
        self.next_check = 0.0
        self.served = self.errors = 0
        self.busy_s = 0.0

    def usable(self, now):
        if not self.alive:
            return False
        return self.valid_until is None or now < self.valid_until - END_MARGIN_S

    def free(self, now):
        return self.usable(now) and self.busy < self.slots


class Item:
    __slots__ = ("method", "path", "body", "ctype", "key", "t0", "attempts", "fut", "tried")

    def __init__(self, method, path, body, ctype, key):
        self.method, self.path, self.body, self.ctype, self.key = method, path, body, ctype, key
        self.t0 = time.time()
        self.attempts = 0
        self.tried = []
        self.fut = asyncio.get_running_loop().create_future()


class Dispatcher:
    def __init__(self):
        self.backends = {}            # name -> Backend
        self.queue = deque()
        self.affinity = {}            # conversation key -> backend name
        self.wake = asyncio.Event()
        self.session = None
        self.started = time.time()
        self.totals = {"requests": 0, "ok": 0, "failed": 0, "timeouts": 0, "retries": 0}

    # ---------------------------------------------------------------- servers
    def load_backends(self):
        try:
            cfg = json.loads(BACKENDS_FILE.read_text())
        except Exception:
            return
        seen = set()
        for b in cfg.get("backends", []):
            name = b["name"]
            seen.add(name)
            vu = b.get("valid_until")
            vu = datetime.fromisoformat(vu.replace("Z", "+00:00")).timestamp() if vu else None
            if name in self.backends and self.backends[name].url == b["url"].rstrip("/"):
                be = self.backends[name]
                be.slots, be.valid_until = max(1, int(b.get("slots", 1))), vu
            else:
                self.backends[name] = Backend(name, b["url"], b.get("slots", 1), vu)
        for name in list(self.backends):
            if name not in seen and self.backends[name].busy == 0:
                del self.backends[name]
        self.wake.set()

    async def health_loop(self):
        while True:
            self.load_backends()
            now = time.time()
            for be in list(self.backends.values()):
                if not be.alive and now >= be.next_check:
                    be.next_check = now + 30
                    if await self.ping(be):
                        be.alive = True
                        self.wake.set()
            await asyncio.sleep(10)

    async def ping(self, be) -> bool:
        base = be.url[:-3] if be.url.endswith("/v1") else be.url
        try:
            async with self.session.get(base + "/", timeout=ClientTimeout(total=15)) as r:
                return r.status == 200
        except Exception:
            return False

    # ---------------------------------------------------------------- scheduling
    def pick(self, item, now):
        """The conversation's last server if free, else the free server with the most free slots."""
        last = self.affinity.get(item.key)
        if last in self.backends and self.backends[last].free(now) and last not in item.tried[-1:]:
            return self.backends[last]
        free = [b for b in self.backends.values() if b.free(now) and b.name not in item.tried[-1:]]
        if not free:
            free = [b for b in self.backends.values() if b.free(now)]
        if not free:
            return None
        return max(free, key=lambda b: (b.slots - b.busy, -b.busy_s))

    async def scheduler(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            now = time.time()
            while self.queue:
                item = self.queue[0]
                if item.fut.done():
                    self.queue.popleft()
                    continue
                be = self.pick(item, now)
                if be is None:
                    break
                self.queue.popleft()
                be.busy += 1
                asyncio.create_task(self.run(item, be))

    async def expiry_loop(self):
        while True:
            now = time.time()
            for item in list(self.queue):
                if not item.fut.done() and now - item.t0 > MAX_WAIT_S:
                    self.queue.remove(item)
                    self.totals["timeouts"] += 1
                    item.fut.set_result((503, json.dumps({"error": {"message": f"no model server free within {MAX_WAIT_S:.0f} s",
                                                                    "type": "dispatcher_timeout"}}).encode(), "application/json"))
                    self.log(item, None, 503, 0)
            self.wake.set()
            await asyncio.sleep(5)

    async def run(self, item, be):
        item.attempts += 1
        item.tried.append(be.name)
        t = time.time()
        status, body, ctype, retry = None, b"", "application/json", False
        try:
            async with self.session.request(item.method, be.url + item.path, data=item.body,
                                            headers={"Content-Type": item.ctype},
                                            timeout=ClientTimeout(total=REQUEST_TIMEOUT_S)) as r:
                body = await r.read()
                status, ctype = r.status, r.headers.get("Content-Type", "application/json")
                if status in RETRY_STATUS or (status == 200 and not body.strip()):
                    retry = True
        except Exception as e:
            retry, body = True, json.dumps({"error": {"message": f"{type(e).__name__}: {e}"[:300]}}).encode()
        finally:
            be.busy -= 1
            be.busy_s += time.time() - t
        if retry:
            be.errors += 1
            be.alive, be.next_check = False, time.time() + 30
            if item.attempts < MAX_ATTEMPTS and not item.fut.done():
                self.totals["retries"] += 1
                self.queue.appendleft(item)               # keep its place at the front
            elif not item.fut.done():
                self.totals["failed"] += 1
                item.fut.set_result((502, body or b'{"error": {"message": "all servers failed"}}', "application/json"))
                self.log(item, be, 502, time.time() - t)
        else:
            be.served += 1
            self.affinity[item.key] = be.name
            if len(self.affinity) > 5000:
                self.affinity.clear()
            if not item.fut.done():
                self.totals["ok" if status == 200 else "failed"] += 1
                item.fut.set_result((status, body, ctype))
                self.log(item, be, status, time.time() - t, body)
        self.wake.set()

    def log(self, item, be, status, latency, body=b""):
        rec = {"t": round(time.time(), 1), "server": be.name if be else None, "status": status,
               "wait_s": round(time.time() - item.t0 - latency, 2), "latency_s": round(latency, 2),
               "attempts": item.attempts}
        try:
            u = json.loads(body.decode().strip()).get("usage") or {}
            rec.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))
        except Exception:
            pass
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- HTTP
    async def handle(self, request):
        path = request.path
        if not path.startswith("/v1"):
            return web.json_response({"error": "use /v1/..."}, status=404)
        body = await request.read()
        key = "-"
        try:
            msgs = json.loads(body).get("messages") or []
            key = hashlib.sha1(json.dumps(msgs[:2], sort_keys=True).encode()).hexdigest()[:16]
        except Exception:
            pass
        item = Item(request.method, path[3:], body, request.headers.get("Content-Type", "application/json"), key)
        self.totals["requests"] += 1
        self.queue.append(item)
        self.wake.set()
        status, out, ctype = await item.fut
        return web.Response(status=status, body=out, content_type=ctype.split(";")[0])

    async def status(self, request):
        now = time.time()
        return web.json_response({
            "uptime_s": round(now - self.started), "queue": len(self.queue),
            "oldest_wait_s": round(now - self.queue[0].t0, 1) if self.queue else 0,
            "slots_total": sum(b.slots for b in self.backends.values() if b.usable(now)),
            "busy_total": sum(b.busy for b in self.backends.values()), "totals": self.totals,
            "servers": [{"name": b.name, "alive": b.alive, "usable": b.usable(now), "slots": b.slots, "busy": b.busy,
                         "served": b.served, "errors": b.errors, "busy_s": round(b.busy_s),
                         "ends_in_h": round((b.valid_until - now) / 3600, 2) if b.valid_until else None}
                        for b in self.backends.values()]})

    async def health(self, request):
        ok = any(b.usable(time.time()) for b in self.backends.values())
        return web.json_response({"ok": ok}, status=200 if ok else 503)

    async def start(self, app):
        self.session = ClientSession()
        self.load_backends()
        for coro in (self.scheduler(), self.health_loop(), self.expiry_loop()):
            asyncio.create_task(coro)

    async def stop(self, app):
        await self.session.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11500)
    a = ap.parse_args()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    d = Dispatcher()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/status", d.status)
    app.router.add_get("/health", d.health)
    app.router.add_route("*", "/v1/{tail:.*}", d.handle)
    app.on_startup.append(d.start)
    app.on_cleanup.append(d.stop)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
