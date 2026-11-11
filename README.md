# Requirements as the Oracle

**Generating REST API tests from vague requirements, without learning the bugs they should catch.**
Team PreTrained (Sohail Memon, Mohd Ahmad, Kathan Patel, Md Taufique Hussain), LMA Group Project 2026.

An LLM agent that writes executable REST API tests from natural-language requirements and an OpenAPI specification, using one small open model (Llama 3.1 8B, 4-bit). It is evaluated on [RESTestBench](https://github.com/casablancahotelsoftware/RESTestBench): 3 services, 106 requirements, 228 planted bugs.

## Results in brief

Planted bugs caught (%), mean ± sd over 3 runs, vague requirements, same model for every tool:

| Tool | Caught | Refined on the buggy service (mutant-gen) | Adaptation gap |
|---|---|---|---|
| B1 single-step (RESTestBench) | 2.6 ± 0.9 | – | – |
| B2 refinement loop (RESTestBench) | 11.3 ± 1.4 | 4.3 ± 2.0 | **7.0** (p = 0.004) |
| B3 prompt-fixed refinement | 2.8 ± 1.8 | 3.1 ± 0.9 | −0.3 |
| B4 response judge | 6.7 ± 1.1 | – | – |
| E Schemathesis fuzzer | 9.5 ± 4.8 | – | – |
| **Ours (final)** | **12.7 ± 2.7** | **14.8 ± 1.2** | −2.1 (n.s.) |

**Comparisons**
- Against single-step generation: +10.1 points (95% CI [5.2, 15.2], p < 0.001).
- Against B2 on the correct service: a draw (+1.5, not significant).
- When tests are refined against the buggy service, the realistic case: 3.4× B2 (+10.5 points, p < 0.001).

**Ablations**
- Two components carry the gain: the plan-and-compile executor (−11.8 points without it) and specification clean-up (−7.5).
- No measurable effect beyond run-to-run noise: derived checks, retrieval, the judge and fine-tuning.

**Human validation**
- Two labellers agreed strongly with each other (κ = 0.84 over 181 labels).
- The Llama judge used for offline check scoring agreed poorly with them (κ 0.03–0.27), so the report gives the human-labelled rates instead.

Full write-up: [`Reports/Final/PreTrained_Final.pdf`](Reports/Final/PreTrained_Final.pdf). Every decision and its reason: [`Plan/Decision_Log.md`](Plan/Decision_Log.md).

## Repository layout

| Path | Content |
|---|---|
| `Proposal/` | Approved proposal (frozen) |
| `Reports/Mid/`, `Reports/Final/` | Mid-term and final reports (LaTeX + PDF) |
| `Plan/` | Decision log, flowchart, remaining-work schedule |
| `pretrained/agent/` | The agent: retrieval, precision layer, planner, lint and spec clean-up, compiler, test runtime, refinement and judges |
| `pretrained/tools_ext/` | Tool adapters for the harness, baselines B3/B4/E, ReAct, multi-agent, MCP and A2A variants |
| `pretrained/configs/` | Agent configurations (`agent/*.yaml`) and experiment queues (`queue_*.yaml`) |
| `pretrained/scripts/` | Benchmark runner, queue runner, LLM dispatcher, tables, statistics, figures, human agreement |
| `pretrained/finetune/` | SFT/DPO data builders and Kaggle training + GGUF export jobs |
| `pretrained/kaggle_llm_server/` | Notebook that serves Llama 3.1 8B (Ollama) from a Kaggle T4 session through a tunnel |
| `pretrained/gold/` | Gold operations, gold checks, reference plans, human-validation sample and labels |
| `pretrained/results/` | Per-run scores (`evaluation_results.json`, `run_meta.json`), report tables (`results/tables/`) |
| `pretrained/harness/` | RESTestBench (git submodule, unmodified) |

## Reproducing

```bash
git clone --recurse-submodules https://github.com/SohailTSM/Requirements-as-the-Oracle
cd Requirements-as-the-Oracle/pretrained
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r harness/requirements.txt
# Docker is also needed: the three services run in containers (see harness/README.md)
```

1. **Serve the model.** You need an OpenAI-compatible endpoint for Llama 3.1 8B Q4_K_M with a 24k context. There are two ways:
   - Run Ollama locally (`scripts/start_ollama.sh`).
   - Use the Kaggle notebook in `kaggle_llm_server/` behind the local dispatcher (`scripts/start_dispatcher.sh`). Server links go in `secrets/llm_backends.json`, which is never committed.
2. **Run one job:**
   ```bash
   .venv/bin/python scripts/run_benchmark.py --tool ours --agent-config configs/agent/final.yaml \
       --services fastapi --variant vague --rep 1 --group finals
   ```
   You can also run a whole queue: `scripts/run_queue.py configs/queue_finals.yaml`. A queue can be restarted safely, because finished jobs are skipped.
3. **Rebuild the tables, statistics and figures from the result files:**
   ```bash
   .venv/bin/python scripts/make_tables.py
   .venv/bin/python scripts/final_stats.py
   .venv/bin/python scripts/make_figures.py
   ```

All reported runs used free Kaggle T4 GPUs. That came to about 105k LLM requests and 251M tokens; no paid APIs were used.
