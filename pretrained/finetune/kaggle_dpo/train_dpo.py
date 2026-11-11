"""Kaggle job: DPO of Llama 3.1 8B Instruct for the refinement role, 3 leave-one-service-out folds,
then merge each fold and export it as Q4_K_M GGUF (Ollama 0.40 serves no adapters, D-57).

Input : /kaggle/input/<dataset>/dpo_fold_<held_out>.jsonl  ({"prompt": [messages], "chosen", "rejected"})
Output: /kaggle/working/adapter_dpo_<fold>/, <fold>.dpo.Q4_K_M.gguf, train_log_dpo_<fold>.json, summary.json
Pairs: "chosen" keeps every locked expectation (the guarded plan), "rejected" changed one or copied
a value the service returned (Decision Log D-54). Loss is computed on the answers only (DPO).
"""
import dataclasses
import gc
import glob
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: E402

PatchDPOTrainer()
from trl import DPOConfig, DPOTrainer  # noqa: E402

BASE = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_LEN, MAX_PROMPT = 6144, 5120
MIN_PAIRS = 40                    # below this a fold is not trained (reported in summary.json)
found = glob.glob("/kaggle/input/**/dpo_fold_*.jsonl", recursive=True)
FOLDS = sorted(os.path.basename(p)[len("dpo_fold_"):-len(".jsonl")] for p in found)   # only the folds in the dataset (D-116)
DATA_DIR = os.path.dirname(found[0]) if found else "/kaggle/input/pretrained-dpo-data"
OUT, TMP = "/kaggle/working", "/tmp/export"
print("data:", found, flush=True)

summary = {}
for fold in FOLDS:
    path = f"{DATA_DIR}/dpo_fold_{fold}.jsonl"
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    if len(rows) < MIN_PAIRS:
        summary[fold] = {"pairs": len(rows), "trained": False, "reason": f"fewer than {MIN_PAIRS} pairs"}
        continue
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    ds = Dataset.from_dict({
        "prompt": [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows],
        "chosen": [r["chosen"] + tok.eos_token for r in rows],
        "rejected": [r["rejected"] + tok.eos_token for r in rows]})
    cfg = dict(per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=2,
               learning_rate=5e-6, beta=0.1, warmup_ratio=0.1, lr_scheduler_type="linear",
               optim="adamw_8bit", max_length=MAX_LEN, max_prompt_length=MAX_PROMPT,
               fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
               logging_steps=5, seed=3407, output_dir=f"{OUT}/ckpt_dpo_{fold}", save_strategy="no",
               report_to="none")
    known = {f.name for f in dataclasses.fields(DPOConfig)}   # TRL renames/drops options between versions
    print("DPOConfig ignores:", [k for k in cfg if k not in known], flush=True)
    tok_kw = "processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else "tokenizer"
    trainer = DPOTrainer(model=model, ref_model=None, train_dataset=ds, **{tok_kw: tok},
                         args=DPOConfig(**{k: v for k, v in cfg.items() if k in known}))
    stats = trainer.train()
    model.save_pretrained(f"{OUT}/adapter_dpo_{fold}")
    tok.save_pretrained(f"{OUT}/adapter_dpo_{fold}")
    json.dump(trainer.state.log_history, open(f"{OUT}/train_log_dpo_{fold}.json", "w"), indent=1)
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    # merge + quantize like the base model we serve (Q4_K_M)
    work = f"{TMP}/{fold}"
    os.makedirs(TMP, exist_ok=True)
    os.chdir(TMP)        # the ~17 GB f16 intermediate is written to the cwd; /kaggle/working holds only 20 GB
    model.save_pretrained_gguf(work, tok, quantization_method="q4_k_m")
    os.chdir(OUT)
    ggufs = [g for g in glob.glob(f"{work}*/**/*.gguf", recursive=True) + glob.glob(f"{work}*.gguf") if "q4_k_m" in g.lower()]
    if ggufs:
        shutil.move(ggufs[0], f"{OUT}/{fold}.dpo.Q4_K_M.gguf")
    shutil.rmtree(TMP, ignore_errors=True)
    summary[fold] = {"pairs": len(rows), "trained": True, "train_loss": stats.training_loss,
                     "gguf": bool(ggufs), "minutes": round((time.time() - t0) / 60, 1)}
    print(fold, summary[fold], flush=True)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print("done", summary, flush=True)
