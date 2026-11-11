#!/usr/bin/env python3
"""Add model servers from a pasted notebook log block to the dispatcher (Decision Log D-101).

    .venv/bin/python kaggle_llm_server/add_links.py NAME < pasted.txt
    # or: ... add_links.py NAME "<the pasted block>"

Reads the PRETRAINED_LLM_URL_n lines, the "valid until about HH:MM UTC" time and the
"(N parallel requests per GPU)" line, and adds NAME-1, NAME-2, ... to secrets/llm_backends.json
(mode 600). The running dispatcher picks them up within 30 s. Prints only names, never links.
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent.parent / "secrets" / "llm_backends.json"


def main() -> int:
    name = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
    urls = re.findall(r"PRETRAINED_LLM_URL_(\d) = (https://\S+/v1)", text)
    if not urls:
        print("no PRETRAINED_LLM_URL lines found")
        return 1
    m = re.search(r"valid until about (\d\d):(\d\d) UTC", text)
    until = None
    if m:
        now = datetime.now(timezone.utc)
        until = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if until < now:                                          # a fresh session ends ~12 h ahead: next day
            until += timedelta(days=1)
    models = dict(re.findall(r"PRETRAINED_LLM_MODEL_(\d) = (\S+)", text))   # fine-tuned servers (D-117)
    p = re.search(r"\((\d+) parallel requests per GPU\)", text)
    slots = int(p.group(1)) if p else 1
    data = json.loads(STORE.read_text()) if STORE.exists() else {"backends": []}
    data["backends"] = [b for b in data["backends"] if not b["name"].startswith(name + "-")]
    for i, url in urls:
        data["backends"].append({"name": f"{name}-{i}", "url": url, "slots": slots,
                                 "valid_until": until.strftime("%Y-%m-%dT%H:%M:%SZ") if until else None})
        if i in models:
            data["backends"][-1]["models"] = [models[i]]
    STORE.write_text(json.dumps(data, indent=1))
    os.chmod(STORE, 0o600)
    ist = (until + timedelta(hours=5, minutes=30)).strftime("%H:%M IST") if until else "unknown"
    print(f"added {', '.join(f'{name}-{i}' + (f' [{models[i]}]' if i in models else '') for i, _ in urls)} "
          f"({slots} slot(s) each, until {ist}); "
          f"{len(data['backends'])} server(s) in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
