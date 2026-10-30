"""PreTrained LLM server for Kaggle (2 x T4): two Ollama servers (one per GPU), each reachable
through its own Cloudflare quick tunnel. Run with "Save Version -> Save & Run All" (background).

The links to share appear in the log between the ===== lines ("SHARE THESE LINKS"). Each link
already contains a random secret; requests without it are refused. A link stops working when the
session ends (Kaggle's 12-hour limit) or is stopped. If a tunnel restarts, a NEW link is printed.

Same software, model and settings as the project's local server (Decision Log D-82):
Ollama 0.40.3, llama3.1:8b (Q4_K_M, digest checked), alias Meta-Llama-3.1-8B-Instruct with a
24576-token context, flash attention, q8_0 KV cache, one request at a time per GPU.
"""
import http.client
import http.server
import json
import os
import re
import secrets
import socketserver
import subprocess
import threading
import time
import urllib.request

OLLAMA_VERSION = "v0.40.3"
MODEL_TAG = "llama3.1:8b"
EXPECTED_MODEL_LAYER = "sha256:667b0c1932bc"      # the local project copy (prefix)
ALIAS = "Meta-Llama-3.1-8B-Instruct"
NUM_CTX = 24576
RUN_HOURS = float(os.environ.get("RUN_HOURS", "11.8"))   # Kaggle stops GPU sessions at 12 h
# Requests each GPU answers at the same time. 2 fills the GPU while a job runs its tests between
# LLM calls (Decision Log D-100). Each request keeps the full NUM_CTX context; the KV cache doubles
# (~3 GB at q8_0), which fits a 15 GB T4 next to the 4.9 GB model.
PARALLEL = int(os.environ.get("PARALLEL", "2"))
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp"
BIN = "/tmp/pt_bin"
START = time.time()


def log(msg):
    print(time.strftime("%H:%M:%S UTC ", time.gmtime()) + msg, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, **kw)


# ----------------------------------------------------------------------------- install
def install():
    os.makedirs(BIN, exist_ok=True)
    if not os.path.exists(f"{BIN}/ollama/bin/ollama"):
        log(f"installing Ollama {OLLAMA_VERSION}")
        sh("apt-get -qq install -y zstd > /dev/null 2>&1 || true")
        sh(f"curl -fsSL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/download/"
           f"{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst")
        sh(f"mkdir -p {BIN}/ollama && tar --use-compress-program=unzstd -xf /tmp/ollama.tar.zst -C {BIN}/ollama")
    if not os.path.exists(f"{BIN}/cloudflared"):
        log("installing cloudflared")
        sh(f"curl -fsSL -o {BIN}/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/"
           f"download/cloudflared-linux-amd64 && chmod +x {BIN}/cloudflared")


# ----------------------------------------------------------------------------- ollama
def start_ollama(gpu, port):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST=f"127.0.0.1:{port}",
               OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_NUM_PARALLEL=str(PARALLEL),
               OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="-1", OLLAMA_MODELS="/tmp/pt_models",
               LLAMA_ARG_CACHE_RAM="2048")
    out = open(f"/tmp/ollama_gpu{gpu}.log", "a")
    subprocess.Popen([f"{BIN}/ollama/bin/ollama", "serve"], env=env, stdout=out, stderr=out)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama on GPU {gpu} did not start")


def ollama_cli(port, args):
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS="/tmp/pt_models")
    return subprocess.run([f"{BIN}/ollama/bin/ollama"] + args, env=env, capture_output=True, text=True)


def prepare_model(port):
    log(f"pulling {MODEL_TAG}")
    r = ollama_cli(port, ["pull", MODEL_TAG])
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    man = json.load(open("/tmp/pt_models/manifests/registry.ollama.ai/library/llama3.1/8b"))
    layer = next(l["digest"] for l in man["layers"] if l["mediaType"].endswith(".model"))
    same = layer.startswith(EXPECTED_MODEL_LAYER)
    log(f"model layer {layer[:19]} -> {'SAME as the project copy' if same else 'DIFFERENT from the project copy!'}")
    with open("/tmp/Modelfile", "w") as f:
        f.write(f"FROM {MODEL_TAG}\nPARAMETER num_ctx {NUM_CTX}\n")
    r = ollama_cli(port, ["create", ALIAS, "-f", "/tmp/Modelfile"])
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return same


def warm(port):
    body = json.dumps({"model": ALIAS, "messages": [{"role": "user", "content": "Say OK."}],
                       "max_tokens": 5, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()


# ----------------------------------------------------------------------------- proxy
STATS = {}


def make_proxy(listen_port, target_port, token):
    prefix = f"/{token}"

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _forward(self):
            if not self.path.startswith(prefix):
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            path = self.path[len(prefix):] or "/"
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            STATS[listen_port] = STATS.get(listen_port, 0) + 1
            result = {}

            def call():
                try:
                    c = http.client.HTTPConnection("127.0.0.1", target_port, timeout=3600)
                    c.request(self.command, path, body=body,
                              headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
                    r = c.getresponse()
                    result.update(status=r.status, body=r.read(), ctype=r.getheader("Content-Type", "application/json"))
                except Exception as e:
                    result.update(status=502, body=json.dumps({"error": str(e)}).encode(), ctype="application/json")

            t = threading.Thread(target=call, daemon=True)
            t.start()
            t.join(20)
            if not t.is_alive():                       # fast answer: send it as is
                self.send_response(result["status"])
                self.send_header("Content-Type", result["ctype"])
                self.send_header("Content-Length", str(len(result["body"])))
                self.end_headers()
                self.wfile.write(result["body"])
                return
            # Slow answer: send headers now and a space every 15 s, so Cloudflare (100 s limit) keeps
            # the request open. Leading spaces are valid JSON, so clients parse the body unchanged.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(data):
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            while t.is_alive():
                chunk(b" ")
                t.join(15)
            chunk(result["body"])
            self.wfile.write(b"0\r\n\r\n")

        do_GET = do_POST = do_DELETE = _forward

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = Server(("127.0.0.1", listen_port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


# ----------------------------------------------------------------------------- tunnels
def start_tunnel(port):
    logf = f"/tmp/tunnel_{port}.log"
    open(logf, "w").close()
    proto = os.environ.get("TUNNEL_PROTOCOL", "")           # e.g. "http2" where UDP/QUIC is blocked
    extra = ["--protocol", proto] if proto else []
    p = subprocess.Popen([f"{BIN}/cloudflared", "tunnel", "--no-autoupdate", *extra, "--url", f"http://127.0.0.1:{port}"],
                         stdout=open(logf, "a"), stderr=subprocess.STDOUT)
    for _ in range(90):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(logf).read())
        if m:
            return p, m.group(0)
        time.sleep(1)
    p.kill()
    raise RuntimeError("tunnel did not start")


def check_public(url, token):
    body = json.dumps({"model": ALIAS, "messages": [{"role": "user", "content": "Reply with the word ready."}],
                       "max_tokens": 5, "temperature": 0}).encode()
    for _ in range(12):                                 # DNS for a new quick tunnel can take a moment
        try:
            req = urllib.request.Request(f"{url}/{token}/v1/chat/completions", body, {"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = e
            time.sleep(10)
    return f"public check failed: {err}"


def announce(links, note=""):
    until = time.strftime("%H:%M UTC", time.gmtime(START + RUN_HOURS * 3600))
    print("\n" + "=" * 78, flush=True)
    print(f"  PRETRAINED LLM SERVER {note}- SHARE THESE LINKS WITH SOHAIL (valid until about {until})", flush=True)
    print(f"  ({PARALLEL} parallel requests per GPU)", flush=True)
    for name, link in links.items():
        print(f"  {name} = {link}", flush=True)
    print("=" * 78 + "\n", flush=True)


# ----------------------------------------------------------------------------- main
def preflight():
    """Stop early with a clear message if the notebook settings are wrong."""
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split("\n")
        gpus = [g for g in gpus if g.strip()]
    except FileNotFoundError:
        gpus = []
    if len(gpus) < 2:
        raise SystemExit("\n>>> SETTINGS PROBLEM: this notebook needs 'GPU T4 x2' (found %d GPU). "
                         "Open Settings -> Accelerator -> GPU T4 x2, then Save Version again.\n" % len(gpus))
    try:
        urllib.request.urlopen("https://github.com", timeout=15)
    except Exception:
        raise SystemExit("\n>>> SETTINGS PROBLEM: Internet is off. Open Settings -> Internet -> On "
                         "(needs a phone-verified Kaggle account), then Save Version again.\n")
    log(f"settings OK: {len(gpus)} x {gpus[0].strip()}, internet on")


def main():
    preflight()
    install()
    servers = [(0, 11434, 8001), (1, 11435, 8002)]
    for gpu, port, _ in servers:
        start_ollama(gpu, port)
    same = prepare_model(servers[0][1])
    for gpu, port, _ in servers:
        log(f"loading the model on GPU {gpu}")
        warm(port)
    token = secrets.token_urlsafe(12)
    links, tunnels = {}, {}
    for gpu, port, pport in servers:
        make_proxy(pport, port, token)
        p, url = start_tunnel(pport)
        tunnels[gpu] = (p, url, pport)
        links[f"PRETRAINED_LLM_URL_{gpu + 1}"] = f"{url}/{token}/v1"
    for gpu, (p, url, pport) in tunnels.items():
        log(f"GPU {gpu} public check: {check_public(url, token)!r}")
    if not same:
        log("WARNING: the model differs from the project copy; tell Sohail.")
    announce(links)

    while time.time() - START < RUN_HOURS * 3600:
        time.sleep(600)
        for gpu, (p, url, pport) in list(tunnels.items()):
            if p.poll() is not None:                       # tunnel died: start a new one, print the new link
                log(f"tunnel for GPU {gpu} stopped; starting a new one")
                p, url = start_tunnel(pport)
                tunnels[gpu] = (p, url, pport)
                links[f"PRETRAINED_LLM_URL_{gpu + 1}"] = f"{url}/{token}/v1"
                announce(links, note="(NEW LINK) ")
        left = (START + RUN_HOURS * 3600 - time.time()) / 3600
        log(f"alive: requests served {dict(STATS)}; {left:.1f} h left")
    log("time is up; stopping (Kaggle's 12-hour session limit)")


if __name__ == "__main__":
    main()
