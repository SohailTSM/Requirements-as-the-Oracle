import os
"""Project-wide paths and local model-server settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # pretrained/
WORKSPACE_ROOT = PROJECT_ROOT.parent                             # "LMA Major Project/"
# RESTestBench submodule; PRETRAINED_HARNESS_ROOT selects a service copy (D-84) for one run
HARNESS_ROOT = Path(os.environ.get("PRETRAINED_HARNESS_ROOT", PROJECT_ROOT / "harness")).resolve()
RESULTS_ROOT = PROJECT_ROOT / "results"
SECRETS_DIR = WORKSPACE_ROOT / "secrets"

SERVICES = ["fastapi", "nestjs-realworld", "todoapp"]

# Local Ollama server (OpenAI-compatible API). The model name matches
# RESTestBench's llama8b config so their tools run unchanged.
# LLM server: the local Ollama by default; PRETRAINED_LLM_URL points a job at a remote server
# (e.g. a Kaggle T4 session, Decision Log D-82). The link contains a secret: use public_llm_url()
# for anything that is written to results or logs.
OLLAMA_BASE_URL = os.environ.get("PRETRAINED_LLM_URL", "http://127.0.0.1:11434/v1")


def public_llm_url(url: str = OLLAMA_BASE_URL) -> str:
    """The server address with its secret path segment hidden."""
    import re
    return re.sub(r"^(https?://[^/]+)/[^/]+/v1/?$", r"\1/<secret>/v1", url)
LLAMA_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
