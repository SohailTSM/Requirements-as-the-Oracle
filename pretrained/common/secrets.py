"""Read API keys from the workspace `secrets/` folder.

Values are returned to callers only; nothing here prints or logs them.
"""
import json
import os

from .paths import SECRETS_DIR


def _read_text(name: str) -> str:
    path = SECRETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing secret file: {path}")
    return path.read_text(encoding="utf-8").strip()


def wandb_api_key() -> str:
    return _read_text("wandb_api_key.txt")


def hf_token() -> str:
    return _read_text("hf_token.txt")


def kaggle_credentials() -> dict:
    return json.loads(_read_text("kaggle.json"))


def export_env() -> None:
    """Expose keys to libraries that read environment variables (W&B, HF, Kaggle)."""
    os.environ.setdefault("WANDB_API_KEY", wandb_api_key())
    os.environ.setdefault("HF_TOKEN", hf_token())
    creds = kaggle_credentials()
    if str(creds.get("key", "")).startswith("KGAT"):      # new-style API token
        os.environ.setdefault("KAGGLE_API_TOKEN", creds["key"])
        os.environ.setdefault("KAGGLE_USERNAME", creds.get("username", ""))
    else:                                               # legacy kaggle.json key
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(SECRETS_DIR))
