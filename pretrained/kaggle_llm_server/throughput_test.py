#!/usr/bin/env python3
"""Throughput of one LLM server link with 1 vs 2 concurrent requests (Decision Log D-100).

    .venv/bin/python kaggle_llm_server/throughput_test.py URL [--n 12] [--concurrency 1 2]

Replays a fixed sample of real prompts from our runs (planner, precision layer, judge, refinement;
same order, temperature 0.2, same max_tokens as the agent) at each concurrency level and reports
total output tokens per second, mean and max latency, errors, and how many planner answers still
parse as plans. The gain of concurrency 2 over 1 is what the queue's SHARED_SLOWDOWN is set from.
"""
import argparse
import glob
import json
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.planner import parse_plan  # noqa: E402
from common.paths import LLAMA_MODEL_NAME, public_llm_url  # noqa: E402

MAX_TOKENS = {"planner": 1800, "precision": 1200, "judge": 300, "refine": 1800}


def sample_prompts(n: int) -> list:
    picks, kinds = [], ["planner", "precision", "judge", "refine"]
    files = sorted(glob.glob(str(ROOT / "results/sweep/ours__default*/fastapi/generated_tests/*/agent_history.json")))
    for f in files:
        for c in json.load(open(f)).get("llm_calls", []):
            kind = next((k for k in kinds if str(c.get("tag", "")).startswith(k)), None)
            if kind and sum(1 for p in picks if p[0] == kind) < -(-n // 4):
                picks.append((kind, c["messages"]))
        if len(picks) >= n:
            break
    return picks[:n]


def run(client, prompts, concurrency: int) -> dict:
    lock, idx, out = threading.Lock(), [0], []

    def worker():
        while True:
            with lock:
                if idx[0] >= len(prompts):
                    return
                i = idx[0]
                idx[0] += 1
            kind, msgs = prompts[i]
            t = time.time()
            try:
                r = client.chat.completions.create(model=LLAMA_MODEL_NAME, messages=msgs, temperature=0.2,
                                                   max_tokens=MAX_TOKENS[kind])
                text = r.choices[0].message.content or ""
                ok_plan = parse_plan(text) is not None if kind == "planner" else None
                out.append((kind, time.time() - t, r.usage.completion_tokens, None, ok_plan))
            except Exception as e:
                out.append((kind, time.time() - t, 0, f"{type(e).__name__}: {e}"[:120], None))

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(concurrency)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    toks = sum(o[2] for o in out)
    plans = [o[4] for o in out if o[4] is not None]
    return {"concurrency": concurrency, "requests": len(out), "wall_s": round(wall, 1),
            "out_tok_per_s": round(toks / wall, 1), "mean_latency_s": round(sum(o[1] for o in out) / len(out), 1),
            "max_latency_s": round(max(o[1] for o in out), 1), "errors": [o[3] for o in out if o[3]],
            "planner_parsed": f"{sum(plans)}/{len(plans)}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    client = OpenAI(base_url=a.url, api_key="ollama", timeout=900, max_retries=2)
    prompts = sample_prompts(a.n)
    print(f"{public_llm_url(a.url)}: {len(prompts)} real prompts "
          f"({', '.join(f'{k} {sum(1 for p in prompts if p[0] == k)}' for k in MAX_TOKENS)})")
    results = [run(client, prompts, c) for c in a.concurrency]
    for r in results:
        print(json.dumps(r))
    if len(results) == 2 and results[0]["out_tok_per_s"]:
        gain = results[1]["out_tok_per_s"] / results[0]["out_tok_per_s"]
        slow = results[1]["mean_latency_s"] / results[0]["mean_latency_s"]
        print(f"throughput gain x{gain:.2f}; each request takes x{slow:.2f} as long (-> SHARED_SLOWDOWN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
