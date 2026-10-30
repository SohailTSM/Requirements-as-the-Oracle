#!/bin/bash
# Start the local Ollama server with the settings used for every experiment.
# One request at a time and one loaded model keep VRAM/RAM use predictable.
export OLLAMA_FLASH_ATTENTION=1        # needed for the quantised KV cache
export OLLAMA_KV_CACHE_TYPE=q8_0       # halves KV-cache memory vs fp16
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_HOST=127.0.0.1:11434
export LLAMA_ARG_CACHE_RAM=2048       # cap llama.cpp host prompt cache at 2 GB (default 8 GB) to protect RAM
exec "$HOME/.local/ollama/bin/ollama" serve >> "$HOME/.local/ollama/serve.log" 2>&1
