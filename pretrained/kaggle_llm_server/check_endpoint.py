#!/usr/bin/env python3
"""Check an LLM server link: the model answers, and how fast (prompt + generation), through the tunnel.

    .venv/bin/python kaggle_llm_server/check_endpoint.py URL [URL ...]

Uses a real planner prompt from a smoke run, so the timing reflects our workload.
"""
import glob
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.paths import LLAMA_MODEL_NAME, public_llm_url  # noqa: E402


def sample_prompt() -> str:
    for f in sorted(glob.glob(str(ROOT / "results/smoke/ours_default_smoke_v3/fastapi/generated_tests/*/agent_history.json"))):
        for c in json.load(open(f)).get("llm_calls", []):
            if str(c.get("tag", "")).startswith("planner"):
                return c["messages"][-1]["content"]
    return "List three HTTP status codes and their meaning."


def check(url: str, prompt: str) -> None:
    client = OpenAI(base_url=url, api_key="ollama", timeout=600, max_retries=0)
    t0 = time.time()
    r = client.chat.completions.create(model=LLAMA_MODEL_NAME, temperature=0.2, max_tokens=600,
                                       messages=[{"role": "user", "content": prompt}])
    dt = time.time() - t0
    u = r.usage
    print(f"{public_llm_url(url)}: {dt:.1f} s, prompt {u.prompt_tokens} tok, answer {u.completion_tokens} tok "
          f"(~{u.completion_tokens / dt:.1f} tok/s end to end); answer starts: {r.choices[0].message.content[:60]!r}")


def main() -> int:
    prompt = sample_prompt()
    for url in sys.argv[1:]:
        try:
            check(url, prompt)
        except Exception as e:
            print(f"{public_llm_url(url)}: FAILED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
