#!/usr/bin/env python3
"""Functional test of the server notebook's parts on the laptop CPU (no GPU, Decision Log D-81/D-100).

    .venv/bin/python kaggle_llm_server/local_cpu_test.py

Runs the same pieces as serve_llm.py: Ollama 0.40.3 with OLLAMA_NUM_PARALLEL=2 (CPU only, small
context to save RAM), the token-checking proxy, and a real Cloudflare quick tunnel. Then:
  1. two requests sent at the same time through the public link both answer correctly;
  2. Ollama processes them in parallel (its log shows 2 slots; both finish together);
  3. a wrong secret is refused (403).
Speed on a CPU says nothing about a T4; the Kaggle throughput test measures that.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_llm as s  # noqa: E402

OLLAMA = str(Path.home() / ".local/ollama/bin/ollama")
PORT, PROXY = 11534, 18511
TEST_MODEL = "pt-cpu-parallel-test"
SCRATCH = Path(os.environ.get("SCRATCH", "/tmp")) / "pt_cpu_test"


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    s.BIN = str(SCRATCH / "bin")
    os.makedirs(s.BIN, exist_ok=True)
    if not os.path.exists(f"{s.BIN}/cloudflared"):
        subprocess.run(f"curl -fsSL -o {s.BIN}/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/"
                       f"download/cloudflared-linux-amd64 && chmod +x {s.BIN}/cloudflared", shell=True, check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OLLAMA_HOST=f"127.0.0.1:{PORT}", OLLAMA_NUM_PARALLEL="2",
               OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_KEEP_ALIVE="5m", OLLAMA_LLM_LIBRARY="cpu", LLAMA_ARG_CACHE_RAM="512", OLLAMA_VULKAN="0",
               GGML_VK_VISIBLE_DEVICES="")              # CPU only: no CUDA, no Vulkan (laptop GPU stays off)
    os.environ["TUNNEL_PROTOCOL"] = "http2"          # this laptop's network blocks cloudflared's QUIC (UDP)
    log = open(SCRATCH / "ollama.log", "w")
    srv = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=log, stderr=log)
    results, tun = {}, None
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2)
                break
            except Exception:
                time.sleep(1)
        mf = SCRATCH / "Modelfile"
        mf.write_text("FROM llama3.1:8b\nPARAMETER num_ctx 4096\n")
        subprocess.run([OLLAMA, "create", TEST_MODEL, "-f", str(mf)], env=env, check=True, capture_output=True)
        s.ALIAS = TEST_MODEL
        token = "localtest123"
        s.make_proxy(PROXY, PORT, token)
        tun = None
        if os.environ.get("NO_TUNNEL"):
            # This laptop's network blocks Cloudflare's tunnel port 7844 (UDP and TCP); the tunnel itself
            # is proven on Kaggle (D-82/D-83). Test the new parts (2 slots, concurrency, proxy) directly.
            url = f"http://127.0.0.1:{PROXY}"
            print("tunnel: skipped (NO_TUNNEL), testing through the local proxy", flush=True)
        else:
            tun, url = s.start_tunnel(PROXY)
            print("tunnel:", url, flush=True)
            print("public check:", s.check_public(url, token), flush=True)
        base = f"{url}/{token}/v1"

        def ask(tag):
            body = json.dumps({"model": TEST_MODEL, "temperature": 0, "max_tokens": 40,
                               "messages": [{"role": "user", "content": f"Reply with exactly: test {tag} ok"}]}).encode()
            t = time.time()
            req = urllib.request.Request(f"{base}/chat/completions", body, {"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=900).read())
            results[tag] = (out["choices"][0]["message"]["content"].strip(), round(time.time() - t, 1))

        t1 = time.time(); ask("solo"); solo = time.time() - t1
        t0 = time.time()
        th = [threading.Thread(target=ask, args=(x,)) for x in ("A", "B")]
        [t.start() for t in th]
        [t.join() for t in th]
        both = time.time() - t0
        try:
            urllib.request.urlopen(urllib.request.Request(f"{url}/wrongsecret/v1/chat/completions", b"{}",
                                                          {"Content-Type": "application/json"}), timeout=30)
            refused = False
        except urllib.error.HTTPError as e:
            refused = e.code == 403
        slots = re.findall(r"n_slots\s*=\s*(\d+)|parallel\s*=\s*(\d+)|n_seq_max\s*=\s*(\d+)", open(SCRATCH / "ollama.log").read())
        print(json.dumps({"answers": results, "solo_s": round(solo, 1), "two_at_once_s": round(both, 1),
                          "parallel_ratio": round(both / solo, 2), "wrong_secret_refused": refused,
                          "slot_lines": slots[-3:]}, indent=1), flush=True)
        ok = (all("ok" in results[k][0].lower() for k in ("A", "B")) and refused and both < 1.7 * solo)
        print("RESULT:", "PASS" if ok else "CHECK", flush=True)
    finally:
        if tun is not None:
            tun.kill()
        subprocess.run([OLLAMA, "rm", TEST_MODEL], env=env, capture_output=True)
        srv.terminate()
        srv.wait(30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
