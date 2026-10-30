"""Kaggle job: merge each fold's LoRA adapter into Llama 3.1 8B Instruct and export Q4_K_M GGUF.

Why: the local Ollama (0.40) no longer accepts LoRA adapters on top of a base model, so each
fine-tuned fold is served as its own merged model, quantized exactly like the base we use
(Q4_K_M). Merging needs ~16 GB of RAM for the 16-bit weights, too much for the local PC.

Input : output of the training kernel (adapter_<fold>/), mounted under /kaggle/input/
Output: /kaggle/working/<fold>.Q4_K_M.gguf  (+ export_summary.json)
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
from unsloth import FastLanguageModel  # noqa: E402

FOLDS = ["fastapi", "nestjs-realworld", "todoapp"]
OUT = "/kaggle/working"
TMP = "/tmp/export"
found = {f: glob.glob(f"/kaggle/input/**/adapter_{f}", recursive=True) for f in FOLDS}
print("adapters:", found, flush=True)
summary = {}
for fold in FOLDS:
    if not found[fold]:
        summary[fold] = "adapter missing"
        continue
    t0 = time.time()
    # 4-bit on the GPU (16-bit does not fit a T4); save_pretrained_gguf merges into 16-bit on the CPU
    model, tok = FastLanguageModel.from_pretrained(found[fold][0], max_seq_length=4096, load_in_4bit=True,
                                                   dtype=None)
    work = f"{TMP}/{fold}"
    os.makedirs(TMP, exist_ok=True)
    os.chdir(TMP)        # the ~17 GB f16 intermediate is written to the cwd; /kaggle/working holds only 20 GB
    model.save_pretrained_gguf(work, tok, quantization_method="q4_k_m")
    os.chdir(OUT)
    ggufs = [g for g in glob.glob(f"{work}*/**/*.gguf", recursive=True) + glob.glob(f"{work}*.gguf")
             if "q4_k_m" in g.lower()]
    print(fold, "gguf files:", ggufs, flush=True)
    if ggufs:
        shutil.move(ggufs[0], f"{OUT}/{fold}.Q4_K_M.gguf")
    summary[fold] = {"gguf": bool(ggufs), "minutes": round((time.time() - t0) / 60, 1)}
    shutil.rmtree(TMP, ignore_errors=True)
    del model, tok
json.dump(summary, open(f"{OUT}/export_summary.json", "w"), indent=1)
print("done", summary, flush=True)
