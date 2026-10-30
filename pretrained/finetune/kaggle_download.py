#!/usr/bin/env python3
"""Stream a Kaggle kernel's output files to disk (constant memory, resumable).

The Kaggle CLI's `kernels output` keeps each whole file in memory before writing it
(`out.write(response.content)`), which for 4.9 GB models risks exhausting RAM (Decision Log D-80).
This lists the same files through the Kaggle API and streams them in 8 MB chunks.

    .venv/bin/python finetune/kaggle_download.py sohailmemon/pretrained-sft-export-gguf DEST [--pattern REGEX]
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.secrets import export_env  # noqa: E402

CHUNK = 8 * 1024 * 1024


def list_files(kernel: str):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug, _ = api.parse_kernel_string(kernel)
    files, token = [], None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            files += [(f.file_name, f.url) for f in (resp.files or [])]
            token = resp.next_page_token
            if not token:
                return files


def stream(url: str, dest: Path, log_every: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if have and r.status_code != 206:          # server ignored the range: start over
            have = 0
            part.unlink(missing_ok=True)
        r.raise_for_status()
        total = have + int(r.headers.get("Content-Length", 0))
        done, t0, last = have, time.time(), 0.0
        with part.open("ab" if have else "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > log_every:
                    rate = (done - have) / max(1e-6, time.time() - t0) / 2**20
                    print(f"  {dest.name}: {done / 2**30:.2f} / {total / 2**30:.2f} GB, {rate:.1f} MB/s", flush=True)
                    last = time.time()
    os.replace(part, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("dest")
    ap.add_argument("--pattern", default=None)
    a = ap.parse_args()
    export_env()
    pat = re.compile(a.pattern) if a.pattern else None
    for name, url in list_files(a.kernel):
        if pat and not pat.search(name):
            continue
        dest = Path(a.dest) / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have {name}", flush=True)
            continue
        print(f"downloading {name}", flush=True)
        for attempt in range(5):                     # resume after network errors
            try:
                stream(url, dest)
                break
            except (requests.RequestException, OSError) as e:
                print(f"  retry {attempt + 1}: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print(f"FAILED {name}", flush=True)
            return 1
        print(f"done {name} ({dest.stat().st_size / 2**30:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
