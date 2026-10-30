#!/bin/bash
# Start the LLM dispatcher (D-101) detached; logs to results/dispatcher.log. Idempotent.
cd "$(dirname "$0")/.."
if curl -s -o /dev/null --max-time 3 http://127.0.0.1:11500/status; then echo "dispatcher already running"; exit 0; fi
setsid nohup .venv/bin/python scripts/llm_dispatcher.py --port 11500 >> results/dispatcher.log 2>&1 < /dev/null &
sleep 2; curl -s --max-time 3 http://127.0.0.1:11500/status >/dev/null && echo "dispatcher started on 127.0.0.1:11500" || echo "dispatcher did not start (see results/dispatcher.log)"
