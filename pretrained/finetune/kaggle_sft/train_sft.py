"""Kaggle job: QLoRA SFT of Llama 3.1 8B Instruct for the precision layer, 3 leave-one-service-out folds.

Input : /kaggle/input/pretrained-sft-data/sft_fold_<held_out>.jsonl  (chat-format examples)
Output: /kaggle/working/adapter_<held_out>/  (LoRA adapter)  +  train_log_<held_out>.json
Runs unattended (kernel_type script). Loss is computed on the assistant answer only.
"""
import gc
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

BASE = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_LEN = 4096
FOLDS = ["fastapi", "nestjs-realworld", "todoapp"]
_found = glob.glob("/kaggle/input/**/sft_fold_*.jsonl", recursive=True)
DATA_DIR = os.path.dirname(_found[0]) if _found else "/kaggle/input/pretrained-sft-data"
print("data dir:", DATA_DIR, "files:", _found, flush=True)
OUT = "/kaggle/working"

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
summary = {}
for fold in FOLDS:
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    rows = [json.loads(line) for line in open(f"{DATA_DIR}/sft_fold_{fold}.jsonl")]
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(dataset_text_field="text", max_seq_length=MAX_LEN, per_device_train_batch_size=1,
                       gradient_accumulation_steps=8, num_train_epochs=3, learning_rate=2e-4,
                       warmup_steps=5, lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.01,
                       fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
                       logging_steps=5, seed=3407, output_dir=f"{OUT}/ckpt_{fold}", save_strategy="no",
                       report_to="none"))
    trainer = train_on_responses_only(trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                                      response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")
    stats = trainer.train()
    model.save_pretrained(f"{OUT}/adapter_{fold}")
    tok.save_pretrained(f"{OUT}/adapter_{fold}")
    json.dump(trainer.state.log_history, open(f"{OUT}/train_log_{fold}.json", "w"), indent=1)
    summary[fold] = {"examples": len(rows), "train_loss": stats.training_loss,
                     "minutes": round((time.time() - t0) / 60, 1)}
    print(fold, summary[fold], flush=True)
    del model, tok, trainer
    gc.collect()
    torch.cuda.empty_cache()
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print("done", summary, flush=True)
