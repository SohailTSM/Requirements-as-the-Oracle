#!/usr/bin/env python3
"""Integration tests for scripts/llm_dispatcher.py with fake model servers (Decision Log D-101).

    .venv/bin/python scripts/test_dispatcher.py

Fake servers mimic the Kaggle proxy (URL /<secret>/v1/..., ping on /<secret>/) and can answer
normally, fail with Cloudflare 530, drop the connection mid-answer, or send keep-alive spaces
before the JSON. Each records its peak concurrency. Every scenario prints PASS/FAIL.
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from aiohttp import web
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="dispatch_test_"))
BACKENDS = TMP / "backends.json"
DPORT = 11599
DURL = f"http://127.0.0.1:{DPORT}/v1"
RESULTS = []


# ------------------------------------------------------------------ fake servers
class Fake:
    def __init__(self, port, secret="s3cret"):
        self.port, self.secret = port, secret
        self.mode, self.latency = "ok", 0.3
        self.active = self.peak = self.served = 0
        self.prefixes = []

    async def chat(self, request):
        data = await request.json()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.mode == "fail530":
                return web.Response(status=530, text="cloudflare: origin unreachable")
            if self.mode == "drop":
                await asyncio.sleep(0.5)
                request.transport.close()
                return web.Response(status=200)
            content = f"from-{self.port}:{data['messages'][-1]['content']}"
            payload = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": data.get("model", "m"),
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": content}}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}).encode()
            if self.mode == "spaces":                        # like the Kaggle proxy for slow answers
                resp = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
                await resp.prepare(request)
                for _ in range(3):
                    await resp.write(b" ")
                    await asyncio.sleep(1)
                await resp.write(payload)
                await resp.write_eof()
                self.served += 1
                return resp
            await asyncio.sleep(self.latency)
            self.served += 1
            self.prefixes.append(data["messages"][0]["content"])
            return web.Response(body=payload, content_type="application/json")
        finally:
            self.active -= 1

    async def ping(self, request):
        return web.Response(status=530 if self.mode in ("fail530", "down") else 200, text="Ollama is running")

    def start(self):
        app = web.Application()
        app.router.add_post(f"/{self.secret}/v1/chat/completions", self.chat)
        app.router.add_get(f"/{self.secret}/", self.ping)
        loop = asyncio.new_event_loop()
        runner = web.AppRunner(app)

        def run():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", self.port).start())
            loop.run_forever()
        threading.Thread(target=run, daemon=True).start()
        time.sleep(0.3)
        return self

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/{self.secret}/v1"


def write_backends(entries):
    BACKENDS.write_text(json.dumps({"backends": entries}))


def status():
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DPORT}/status", timeout=5).read())


def client(timeout=60):
    return OpenAI(base_url=DURL, api_key="x", timeout=timeout, max_retries=0)


def ask(c, text, first="sys"):
    r = c.chat.completions.create(model="m", messages=[{"role": "system", "content": first},
                                                       {"role": "user", "content": text}])
    return r.choices[0].message.content


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def concurrent(n, workers, fn):
    out, lock, idx = [], threading.Lock(), [0]

    def w():
        while True:
            with lock:
                if idx[0] >= n:
                    return
                i = idx[0]
                idx[0] += 1
            try:
                out.append(fn(i))
            except Exception as e:
                out.append(e)
    ts = [threading.Thread(target=w) for _ in range(workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


# ------------------------------------------------------------------ scenarios
def main() -> int:
    A, B, C = Fake(18601).start(), Fake(18602).start(), Fake(18603).start()
    write_backends([{"name": "A", "url": A.url, "slots": 1}, {"name": "B", "url": B.url, "slots": 2},
                    {"name": "C", "url": C.url, "slots": 2}])
    env = dict(os.environ, PRETRAINED_BACKENDS_FILE=str(BACKENDS), PRETRAINED_DISPATCH_LOG=str(TMP / "req.jsonl"),
               DISPATCH_MAX_WAIT_S="8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/llm_dispatcher.py"), "--port", str(DPORT)],
                            env=env, stdout=open(TMP / "disp.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            try:
                status()
                break
            except Exception:
                time.sleep(0.2)
        c = client()

        # 1 basic round trip, usage passes through
        r = c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hello"}])
        check("1 basic round trip", r.choices[0].message.content.endswith(":hello") and r.usage.completion_tokens == 3)

        # 2 concurrency limits and spread: 25 requests, 8 client threads, 5 slots of 1.0 s
        for f in (A, B, C):
            f.latency, f.peak = 1.0, 0
        t0 = time.time()
        out = concurrent(25, 8, lambda i: ask(c, f"q{i}"))
        wall = time.time() - t0
        errs = [o for o in out if isinstance(o, Exception)]
        check("2 slot limits never exceeded", A.peak <= 1 and B.peak <= 2 and C.peak <= 2,
              f"peaks A={A.peak} B={B.peak} C={C.peak}")
        check("2 all 25 answered, all slots used", not errs and len(out) == 25 and 4.5 <= wall <= 8.0,
              f"wall {wall:.1f}s (ideal 5.0s)")

        # 3 failover on Cloudflare 530: A fails, its requests go to B/C
        for f in (A, B, C):
            f.latency = 0.2
        A.mode = "fail530"
        out = concurrent(10, 5, lambda i: ask(c, f"f{i}"))
        st = {s["name"]: s for s in status()["servers"]}
        check("3 failover after 530", all(isinstance(o, str) and "from-1860" in o and "18601" not in o for o in out)
              and not st["A"]["alive"], f"A alive={st['A']['alive']} errors={st['A']['errors']}")

        # 4 server dropping the connection mid-answer: request retried elsewhere
        A.mode = "ok"
        time.sleep(41)                                         # health check revives A (≤ 40 s)
        st = {s["name"]: s for s in status()["servers"]}
        check("5 revival of a recovered server", st["A"]["alive"], "")
        B.mode = "drop"
        out = concurrent(10, 5, lambda i: ask(c, f"d{i}"))
        check("4 dropped connection -> retried elsewhere", all(isinstance(o, str) and "18602" not in o for o in out),
              f"errors {[str(o)[:60] for o in out if not isinstance(o, str)]}")
        B.mode = "ok"

        # 6 keep-alive spaces before the JSON (slow answers via the Kaggle proxy) parse fine
        write_backends([{"name": "C", "url": C.url, "slots": 2}])
        time.sleep(12)
        C.mode = "spaces"
        txt = ask(c, "slow")
        check("6 keep-alive spaces pass through", txt.endswith(":slow"), txt)
        C.mode = "ok"

        # 7 live reload: add B back, it gets work
        write_backends([{"name": "B", "url": B.url, "slots": 2}, {"name": "C", "url": C.url, "slots": 2}])
        time.sleep(12)
        b0 = B.served
        concurrent(8, 4, lambda i: ask(c, f"r{i}"))
        check("7 live reload adds a server", B.served > b0, f"B served {B.served - b0} of 8")

        # 8 conversation affinity: same first two messages -> same server when it is free
        C.prefixes.clear(); B.prefixes.clear()
        for i in range(6):
            ask(c, f"turn{i}", first="conversation-XYZ")
        check("8 affinity keeps a conversation on one server",
              (C.prefixes.count("conversation-XYZ") == 6) or (B.prefixes.count("conversation-XYZ") == 6),
              f"B={B.prefixes.count('conversation-XYZ')} C={C.prefixes.count('conversation-XYZ')}")

        # 9 server in its last 2 minutes gets no new work
        soon = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 60))
        write_backends([{"name": "B", "url": B.url, "slots": 2, "valid_until": soon},
                        {"name": "C", "url": C.url, "slots": 2}])
        time.sleep(12)
        b0 = B.served
        concurrent(6, 3, lambda i: ask(c, f"e{i}"))
        check("9 server about to end gets no new work", B.served == b0, f"B served {B.served - b0}")

        # 10 no server at all: request waits, then a clear 503 (max wait 8 s in this test)
        write_backends([])
        time.sleep(12)
        t0 = time.time()
        try:
            ask(client(), "nobody")
            ok, msg = False, "answered?"
        except Exception as e:
            ok, msg = "503" in str(e) or "dispatcher_timeout" in str(e), str(e)[:80]
        check("10 no server -> 503 after max wait", ok and 7 <= time.time() - t0 <= 16, f"{time.time() - t0:.1f}s {msg}")

        # 11 load: 300 requests, 40 client threads, 3 servers (5 slots), 0.1 s latency
        write_backends([{"name": "A", "url": A.url, "slots": 1}, {"name": "B", "url": B.url, "slots": 2},
                        {"name": "C", "url": C.url, "slots": 2}])
        time.sleep(12)
        for f in (A, B, C):
            f.latency, f.peak = 0.1, 0
        t0 = time.time()
        out = concurrent(300, 40, lambda i: ask(c, f"L{i}"))
        errs = [o for o in out if not isinstance(o, str)]
        st = status()
        check("11 load 300 requests, no errors", not errs and len(out) == 300,
              f"{time.time() - t0:.1f}s, peaks A={A.peak} B={B.peak} C={C.peak}, queue now {st['queue']}")

        # 12 our agent's LLM class through the dispatcher
        sys.path.insert(0, str(ROOT))
        from agent.llm import LLM
        llm = LLM(base_url=DURL)
        out = llm.complete("agent-call", max_tokens=10, tag="test")
        check("12 agent LLM class works through the dispatcher", out.endswith(":agent-call"), out)

        lines = (TMP / "req.jsonl").read_text().splitlines()
        secret_leak = any("s3cret" in l for l in lines)
        check("13 request log has no secrets and records every answer", not secret_leak and len(lines) >= 300,
              f"{len(lines)} log lines")

        # 14 per-model routing (fine-tuned servers, D-117): each request goes only to a server of its
        # model, and a busy model does not hold up requests for another one
        write_backends([{"name": "A", "url": A.url, "slots": 1, "models": ["m1"]},
                        {"name": "B", "url": B.url, "slots": 2, "models": ["m2"]}])
        time.sleep(11)
        A.latency, B.latency = 1.0, 0.1
        c = client()

        def ask_model(i):
            model = "m1" if i < 6 else "m2"
            t = time.time()
            r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": f"{model}-{i}"}])
            return model, r.choices[0].message.content, time.time() - t
        out = concurrent(8, 8, ask_model)
        wrong = [o for o in out if isinstance(o, Exception) or not o[1].startswith(
            f"from-{A.port if o[0] == 'm1' else B.port}:")]
        m2_wait = max(o[2] for o in out if not isinstance(o, Exception) and o[0] == "m2")
        check("14 per-model routing, no head-of-line blocking", not wrong and m2_wait < 2.0,
              f"wrong {len(wrong)}, slowest m2 {m2_wait:.1f}s (m1 queue ~6s)")
        A.latency = B.latency = 0.3
    finally:
        proc.terminate()
    passed = sum(ok for _, ok in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
