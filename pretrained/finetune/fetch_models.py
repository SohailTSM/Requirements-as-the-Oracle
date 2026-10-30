#!/usr/bin/env python3
"""Wait for a Kaggle training kernel, run the GGUF export kernel on its output, then install the
merged models locally (Decision Log D-57, D-79). Network errors while polling are retried.

    .venv/bin/python finetune/fetch_models.py --train sohailmemon/pretrained-precision-sft \
        --export sohailmemon/pretrained-sft-export-gguf --export-dir finetune/kaggle_export \
        --prefix pt-precision-sft --hf-repo sohailtsm/pretrained-precision-sft
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.secrets import export_env  # noqa: E402

K = str(ROOT / ".venv/bin/kaggle")
DONE = ("COMPLETE", "ERROR", "CANCEL")


def status(kernel: str) -> str:
    for attempt in range(10):
        try:
            p = subprocess.run([K, "kernels", "status", kernel], capture_output=True, text=True, timeout=120)
            out = (p.stdout + p.stderr).strip()
            if "status" in out:
                return out
        except subprocess.TimeoutExpired:
            pass
        time.sleep(30 * (attempt + 1))
    return "unknown (status polling failed 10 times)"


def wait(kernel: str) -> str:
    last = None
    while True:
        s = status(kernel)
        if s != last:
            print(time.strftime("%H:%M"), s[-90:], flush=True)
            last = s
        if any(d in s for d in DONE):
            return s
        time.sleep(180)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--skip-train-wait", action="store_true")
    a = ap.parse_args()
    export_env()
    if not a.skip_train_wait and "COMPLETE" not in wait(a.train):
        print("training did not complete", flush=True)
        return 1
    p = subprocess.run([K, "kernels", "push", "-p", a.export_dir], capture_output=True, text=True, timeout=300)
    print(p.stdout.strip()[-200:], p.stderr.strip()[-200:], flush=True)
    time.sleep(90)
    if "COMPLETE" not in wait(a.export):
        print("export did not complete", flush=True)
        return 1
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "finetune/install_adapters.py"), "--kernel", a.train,
           "--export-kernel", a.export, "--prefix", a.prefix] + (["--hf-repo", a.hf_repo] if a.hf_repo else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
