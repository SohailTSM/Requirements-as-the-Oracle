#!/usr/bin/env python3
"""Print the share links (and status lines) from a running PreTrained LLM server notebook's log.

    .venv/bin/python kaggle_llm_server/read_links.py sohailmemon/pretrained-llm-server-a [--wait]

With --wait it polls every 30 s until the links appear. Teammates' runs: they send the links instead.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.secrets import export_env  # noqa: E402


def read_log(kernel: str, seconds: int = 25) -> str:
    """Live log of a running session (`kaggle kernels logs -f`, which streams; we read for a while).
    The session-output API returns an empty log until the run has finished."""
    import subprocess
    p = subprocess.run(["timeout", str(seconds), str(ROOT / ".venv/bin/kaggle"), "kernels", "logs", "-f", kernel],
                       capture_output=True, text=True)
    return p.stdout


def links(text: str) -> dict:
    out = {}
    for name, url in re.findall(r"(PRETRAINED_LLM_URL_\d) = (https://\S+)", text):
        out[name] = url                                   # later (NEW LINK) blocks overwrite earlier ones
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    export_env()
    while True:
        try:
            text = read_log(a.kernel)
        except Exception as e:
            text = f"(log not readable yet: {type(e).__name__}: {e})"
        found = links(text)
        if found:                                          # keep the full links (with secret) out of logs
            store = ROOT.parent / "secrets" / "llm_endpoints.json"
            data = json.loads(store.read_text()) if store.exists() else {}
            data[a.kernel] = {"links": found, "read_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
            store.write_text(json.dumps(data, indent=1))
            os.chmod(store, 0o600)
        if found or not a.wait:
            status = [l for l in text.splitlines() if re.match(r"\d\d:\d\d:\d\d UTC ", l)]
            print("\n".join(status[-8:]))
            from common.paths import public_llm_url
            for k, v in found.items():
                print(f"{k} = {public_llm_url(v)}   (full link saved in secrets/llm_endpoints.json)")
            return 0 if found else 1
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
