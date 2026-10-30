#!/usr/bin/env python3
"""Fetch fine-tuned models from Kaggle, register them in Ollama, back up adapters to HF Hub.

    .venv/bin/python finetune/install_adapters.py --kernel sohailmemon/pretrained-precision-sft \
        --export-kernel sohailmemon/pretrained-sft-export-gguf \
        --prefix pt-precision-sft [--hf-repo sohailtsm/pretrained-precision-sft]

Ollama 0.40 no longer accepts LoRA adapters (Decision Log D-57), so each fold is registered from
the merged Q4_K_M GGUF made by the export kernel, with the base model's chat template and stop
tokens. Without --export-kernel the old adapter route is tried.

Creates Ollama models <prefix>-<held_out_service> (Llama 3.1 8B Q4_K_M + adapter, 24k context).
The fold for service S was trained WITHOUT S, so S's requirements must use <prefix>-S.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import SERVICES  # noqa: E402
from common.secrets import export_env, hf_token  # noqa: E402

PY = str(PROJECT_ROOT / ".venv/bin/python")
OLLAMA = str(Path.home() / ".local/ollama/bin/ollama")
CONVERTER = PROJECT_ROOT / "third_party/llama.cpp/convert_lora_to_gguf.py"
BASE_REPO = "unsloth/Meta-Llama-3.1-8B-Instruct"   # config + tokenizer only (already cached)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--export-kernel", default=None, help="Kaggle kernel with <fold>.Q4_K_M.gguf files")
    args = ap.parse_args()
    export_env()
    out = PROJECT_ROOT / "finetune" / "artifacts" / args.prefix
    out.mkdir(parents=True, exist_ok=True)
    if not all((out / f"adapter_{svc}").exists() for svc in SERVICES):
        subprocess.run([str(PROJECT_ROOT / ".venv/bin/kaggle"), "kernels", "output", args.kernel, "-p", str(out), "-o"],
                       check=True)
    from huggingface_hub import HfApi, snapshot_download
    if args.export_kernel:
        gdir = out / "gguf"
        gdir.mkdir(exist_ok=True)
        if not all((gdir / f"{svc}.Q4_K_M.gguf").exists() for svc in SERVICES):
            # streamed in chunks: the Kaggle CLI holds whole files in memory (D-80)
            subprocess.run([PY, str(PROJECT_ROOT / "finetune/kaggle_download.py"), args.export_kernel, str(gdir),
                            "--pattern", r"\.(gguf|json)$"], check=True)
        base_mf = subprocess.run([OLLAMA, "show", "llama3.1:8b", "--modelfile"], capture_output=True,
                                 text=True, check=True).stdout
        template = base_mf[base_mf.index("TEMPLATE"):]
        template = template[:template.index('"""', template.index('"""') + 3) + 3]
        stops = [l for l in base_mf.splitlines() if l.startswith("PARAMETER stop")]
        for svc in SERVICES:
            gguf = gdir / f"{svc}.Q4_K_M.gguf"
            if not gguf.exists():
                print(f"missing merged model for {svc}; skipping")
                continue
            modelfile = out / f"Modelfile.{svc}"
            modelfile.write_text(f"FROM {gguf}\n{template}\n" + "\n".join(stops) + "\nPARAMETER num_ctx 24576\n")
            name = f"{args.prefix}-{svc}"
            subprocess.run([OLLAMA, "create", name, "-f", str(modelfile)], check=True)
            print(f"registered Ollama model {name} (merged Q4_K_M)")
        if args.hf_repo:
            api = HfApi(token=hf_token())
            api.create_repo(args.hf_repo, private=True, exist_ok=True)
            api.upload_folder(folder_path=str(out), repo_id=args.hf_repo, path_in_repo=args.prefix,
                              ignore_patterns=["ckpt_*", "*.gguf", "gguf/*", "unsloth_compiled_cache/*",
                                               "huggingface_tokenizers_cache/*"],
                              commit_message=f"{args.prefix} adapters")
            print(f"uploaded adapters to https://huggingface.co/{args.hf_repo} (private)")
        return 0
    # Config and tokenizer only. Not "*.json": that also fetches model.safetensors.index.json,
    # and the converter then tries to open the weight shards.
    base_dir = snapshot_download(BASE_REPO, local_dir=str(PROJECT_ROOT / "finetune/artifacts/base_config"),
                                 allow_patterns=["config.json", "generation_config.json", "tokenizer*",
                                                 "special_tokens_map.json"])
    for svc in SERVICES:
        adapter = out / f"adapter_{svc}"
        if not adapter.exists():
            print(f"missing adapter for {svc}; skipping")
            continue
        gguf = out / f"{args.prefix}-{svc}.gguf"
        subprocess.run([PY, str(CONVERTER), str(adapter), "--base", base_dir, "--outfile", str(gguf),
                        "--outtype", "f16"], check=True)
        modelfile = out / f"Modelfile.{svc}"
        modelfile.write_text(f"FROM llama3.1:8b\nADAPTER {gguf}\nPARAMETER num_ctx 24576\n")
        name = f"{args.prefix}-{svc}"
        subprocess.run([OLLAMA, "create", name, "-f", str(modelfile)], check=True)
        print(f"registered Ollama model {name}")
    if args.hf_repo:
        api = HfApi(token=hf_token())
        api.create_repo(args.hf_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.hf_repo,
                          ignore_patterns=["ckpt_*", "*.gguf"], commit_message=f"{args.prefix} adapters")
        print(f"uploaded adapters to https://huggingface.co/{args.hf_repo} (private)")
    summary = out / "summary.json"
    if summary.exists():
        print(json.loads(summary.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
