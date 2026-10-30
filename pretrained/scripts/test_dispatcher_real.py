#!/usr/bin/env python3
"""Real-chain test of the dispatcher (Decision Log D-101): real Llama 3.1 8B on the laptop CPU
(no CUDA, no Vulkan; the laptop GPU stays off, D-81) behind the Kaggle proxy code, the dispatcher
in front, then (1) real-prompt replay at concurrency 1 and 2 and (2) one real RESTestBench job
(B1, TodoApp requirement 5) through the harness with all its LLM calls going through the dispatcher.

    .venv/bin/python scripts/test_dispatcher_real.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle_llm_server"))
import serve_llm  # noqa: E402

SCR = Path(os.environ.get("SCRATCH", "/tmp")) / "dispatch_real"
SCR.mkdir(parents=True, exist_ok=True)
OLLAMA = str(Path.home() / ".local/ollama/bin/ollama")
OPORT, PPORT, SHIM, DPORT = 11534, 18521, 18522, 11598
CPU_MODEL = "pt-cpu-dispatch-test"


def wait(url, s=90):
    for _ in range(s):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(1)
    return False


def shim():
    """Maps the agent's model name to the CPU test model (the real alias stays untouched)."""
    import asyncio
    from aiohttp import ClientSession

    async def fwd(request):
        body = await request.read()
        try:
            d = json.loads(body)
            if "model" in d:
                d["model"] = CPU_MODEL
            body = json.dumps(d).encode()
        except Exception:
            pass
        async with ClientSession() as s:
            async with s.request(request.method, f"http://127.0.0.1:{OPORT}{request.path}", data=body,
                                 headers={"Content-Type": "application/json"}) as r:
                return web.Response(status=r.status, body=await r.read(), content_type="application/json")
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", fwd)
    web.run_app(app, host="127.0.0.1", port=SHIM, print=None)


def main() -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OLLAMA_VULKAN="0", GGML_VK_VISIBLE_DEVICES="",
               OLLAMA_LLM_LIBRARY="cpu", OLLAMA_HOST=f"127.0.0.1:{OPORT}", OLLAMA_NUM_PARALLEL="2",
               OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_KEEP_ALIVE="10m",
               LLAMA_ARG_CACHE_RAM="512")
    procs = [subprocess.Popen([OLLAMA, "serve"], env=env, stdout=open(SCR / "ollama.log", "w"), stderr=subprocess.STDOUT)]
    try:
        wait(f"http://127.0.0.1:{OPORT}/")
        (SCR / "Modelfile").write_text("FROM llama3.1:8b\nPARAMETER num_ctx 8192\n")
        subprocess.run([OLLAMA, "create", CPU_MODEL, "-f", str(SCR / "Modelfile")], env=env, check=True, capture_output=True)
        import multiprocessing
        sp = multiprocessing.Process(target=shim, daemon=True)
        sp.start()
        wait(f"http://127.0.0.1:{SHIM}/")
        # the Kaggle notebook's proxy (token check, keep-alive spaces) in front of the shim
        import threading
        threading.Thread(target=lambda: serve_llm.make_proxy(PPORT, SHIM, "realtest"), daemon=True).start()
        time.sleep(1)
        (SCR / "backends.json").write_text(json.dumps({"backends": [
            {"name": "cpu-test", "url": f"http://127.0.0.1:{PPORT}/realtest/v1", "slots": 2}]}))
        denv = dict(os.environ, PRETRAINED_BACKENDS_FILE=str(SCR / "backends.json"),
                    PRETRAINED_DISPATCH_LOG=str(SCR / "requests.jsonl"))
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/llm_dispatcher.py"), "--port", str(DPORT)],
                                      env=denv, stdout=open(SCR / "dispatcher.log", "w"), stderr=subprocess.STDOUT))
        wait(f"http://127.0.0.1:{DPORT}/status")
        durl = f"http://127.0.0.1:{DPORT}/v1"
        print("== (1) real prompts through the dispatcher, concurrency 1 vs 2", flush=True)
        subprocess.run([sys.executable, str(ROOT / "kaggle_llm_server/throughput_test.py"), durl, "--n", "4"])
        print("== (2) real RESTestBench job (B1, TodoApp requirement 5) through the dispatcher", flush=True)
        jenv = dict(os.environ, PRETRAINED_LLM_URL=durl, AZURE_AI_FOUNDRY_ENDPOINT_OPENAI=durl)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/run_benchmark.py"), "--tool", "b1_single_step",
                            "--services", "todoapp", "--variant", "vague", "--rep", "1", "--group", "smoke",
                            "--exp-id", "dispatcher_real_test_b1", "--requirements", "5", "--no-wandb"],
                           env=jenv, cwd=ROOT, capture_output=True, text=True, timeout=3600)
        ev = ROOT / "results/smoke/dispatcher_real_test_b1/todoapp/evaluation_results.json"
        meta = json.loads((ROOT / "results/smoke/dispatcher_real_test_b1/run_meta.json").read_text())
        print("job exit", r.returncode, "| status", meta.get("status"), "| model_server", meta.get("model_server"),
              "| evaluated", ev.exists(), flush=True)
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DPORT}/status", timeout=5).read())
        print("dispatcher totals:", st["totals"], "| servers:", [(s["name"], s["served"], s["errors"]) for s in st["servers"]])
        reqs = [json.loads(l) for l in (SCR / "requests.jsonl").read_text().splitlines()]
        print("request log lines:", len(reqs), "| statuses:", sorted({q["status"] for q in reqs}),
              "| secret in log:", any("realtest" in json.dumps(q) for q in reqs))
    finally:
        for p in procs:
            p.terminate()
        subprocess.run([OLLAMA, "rm", CPU_MODEL], env=env, capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
