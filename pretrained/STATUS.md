# Status — overnight work (started Tue 20 Oct, ~01:00)

_Updated by Claude as work progresses. Last update: Tue 27 Oct ~06:00 IST._
**Every decision and its reason is logged in [`Plan/Decision_Log.md`](../Plan/Decision_Log.md)** (D-01 … D-74 so far; D-40 onward are tonight's changes).

**Laptop GPU off (D-81). Next batch uses the LLM dispatcher (D-101): one local queue in front of all Kaggle servers, any free model slot takes the next request, runs are started while the queue is short. Steps: `scripts/start_dispatcher.sh`, then `kaggle_llm_server/add_links.py <name>` for each pasted link block. Notebook: `PreTrained_LLM_Server.ipynb` (2 requests per GPU, D-100).** All times in IST.

## Latest: batch 5 (teammate's 2 sessions, 26 Oct 17:55 → 27 Oct 05:40 IST), D-105 … D-111
- 4 T4 servers × 2 slots through the dispatcher: **86% of slot-time busy**, 13,158 requests answered, 6 retried, 0 failed. The 4 timeouts after 05:40 were the two cut-off parts below.
- FastAPI instances: 10 (original + copies b…j, each golden-checked 100%, D-103/D-106/D-107).
- **Sweep: every variant done** except the 3 fine-tuned rows, which need a server with the fine-tuned models (D-95). Next: analysis of the sweep (T5/T10).
- FastAPI baselines started after the sweep (D-108/D-109). B3 vague rep2 and B2 precise rep2 are now mostly done.
- 85 parts finished, every one with a test for each requirement. 2 one-requirement parts cut off at the end are in `results/_cutoff/baselines/` (B3 precise rep2 part25 and part27) and will rerun.
- **Left in the queues:** baselines 36 parts ≈ 44 est-h (≈ 66 T4-slot-h; B2 16 parts, B3 8, B4 12), about one 2-session batch. Sweep: 0 (fine-tuned rows pending).
- Queue changes this batch: parts are sized to the time left near a session end (D-108), and the B2 FastAPI estimate is 10.5 h (D-110).

## What is running
| What | Where | Notes |
|---|---|---|
| Ollama (Llama 3.1 8B Q4_K_M, 24k context) | local RTX 5050 | RAM prompt cache capped at 2 GB |
| Baseline queue (`configs/queue_baselines.yaml`) | local, detached | up to 2 jobs at once on different services (D-51); **FastAPI reserved for development** (`results/queue.reserve`, D-60); now: B2 FastAPI vague rep 1, B2 TodoApp precise rep 1 |
| NestJS fix (D-56) | verified | a 2-requirement B1 smoke run now evaluates NestJS |
| Fine-tuned models (final, prompt v2) | **installed 12:31** | Ollama `pt-precision-sft-<service>`, backed up to HF; not yet test-run (GPU paused) |

## Results so far
**Baselines so far** (local Llama 3.1 8B 4-bit, % of planted bugs caught by tests generated on the correct service; NestJS missing everywhere, see the correction below):
| run | FastAPI | TodoApp | notes |
|---|---|---|---|
| B1 single-step, precise (3 runs) | 40.4 / 36.9 / 41.8 | 0 / 0 / 4.5 | ~70% of tests fail on the correct API |
| B1 single-step, vague (3 runs) | 2.8 / 4.3 / 5.7 | 0 / 0 / 0 | ~96% fail on the correct API |
| B2 refinement, vague, run 1 | **14.9** (tests generated on mutants: 7.9) | 4.5 | adaptation gap on FastAPI: 7.0 points |
| B3 prompt-fixed, vague, run 1 | — | 0 | |

**Correction (D-56):** last night's "B1 = 35.0%" covered FastAPI + TodoApp only. The harness skipped NestJS in every run so far ("service unreachable" at a single health check right after a reset) and still reported success. Fixed (longer health check; such runs now count as failures), and the 9 affected NestJS jobs were re-queued.
**Finding:** TodoApp baselines fail almost everything because the model registers users with weak passwords ("password123"), which TodoApp rejects (ASP.NET password policy, not documented in the spec). Our runtime creates strong random passwords for actors, so our agent does not hit this.

**Our agent, smoke runs (FastAPI vague, 5 requirements, 14 mutants):**
| run | caught (valid tests) | what changed |
|---|---|---|
| v1 | 0% (not re-scored) | first version |
| v2 | **14.3%** (was shown as 0%, see the bug below) | format relaxation, lint rules, guard, dependency closure |
| v3 | **14.3%**; on tests generated against mutants **14.3%** (v2: 4.8%); 7/19 tests pass on the correct API (v2: 1/19) | spec-schema lint + clean-up, prompt v2, new relaxation rule, parser fixes (D-40 … D-48) |
| v4 | next, when FastAPI is free | unclosed plans, `resp.` references, unstated exact status → class, redundant signup removed (D-52, D-59) |

**Bug found tonight (D-45):** our tool gave the harness a bare `pytest <path>` command. The workspace path has spaces and pytest is not on the harness shell's PATH, so *every* evaluation of our agent failed before the test ran. Fixed. The baselines were not affected (they use their own tools' command). Added `--phases execute evaluate` to re-score generated tests without LLM calls (D-46).

**Retrieval (offline, 106 requirements, non-auth operations):**
| vague | R@5 | all needed ops in top 5 | MRR |
|---|---|---|---|
| BM25 | 0.79 | 0.69 | 0.70 |
| dense (bge-small) | 0.88 | 0.75 | 0.89 |
| hybrid | 0.85 | 0.73 | 0.83 |
| hybrid + reranker | 0.89 | 0.81 | 0.87 |
| **adaptive (rerank only ≤ 30 words; default, D-78)** | **0.89** | **0.81** | 0.87 |
| + LLM query rewriting | 0.84 | 0.76 | 0.73 |

**Precision layer (offline, vague → gold checks, Llama judge; to be validated with human labels):**
| variant (prompt v2) | check recall | soft recall | contradicts gold | unsupported | checks/req |
|---|---|---|---|---|---|
| catalogue (default) | 0.39 | 0.47 | 0.27 | 0.06 | 3.6 |
| direct rewrite | 0.40 | 0.47 | 0.21 | 0.09 | 5.7 |
| self-consistency | 0.37 | 0.47 | 0.26 | 0.07 | 3.1 |
| catalogue + support filter | 0.32 | 0.42 | 0.27 | 0.04 | 2.5 → not adopted (D-58) |

**Fine-tuning (SFT, pipeline test on prompt v1):** trained fine on Kaggle T4: 20 / 32 / 41 min per fold, final loss 0.35 / 0.21 / 0.22. Not usable yet: Ollama 0.40 refuses LoRA adapters, so the folds are being merged and exported as Q4_K_M models on Kaggle (D-57). Final adapters will be retrained on the frozen prompt (D-44).

**Executor check:** 10 hand-written reference plans pass 10/10 on the correct FastAPI service and catch 26/28 planted bugs.

## Built tonight
- `common/`, `scripts/run_benchmark.py` (now with `--phases` re-scoring), `scripts/run_queue.py`, `scripts/analyze_results.py`
- Baselines: B1/B2 = RESTestBench tools unchanged, B3 prompt-fixed refinement, **B4 response judge** (D-49; smoke on TodoApp works, needs a 600 s test timeout, D-74), **E fuzzer** (Schemathesis, D-50; works, but the correct NestJS service already returns 500s under fuzzing, so the report gives its false-alarm rate too, D-65; 9 runs queued)
- Agent variants for the sweep, all 27 configs present: **ReAct** single agent with tools (D-62), **multi-persona** and **blackboard** (D-63), content full/summary/verdict (D-64), **MCP** tool server (D-66), **A2A** reviewer + debugger agents (D-67), model sft / sft_dpo (D-68). MCP and A2A round trips tested; the rest wait for FastAPI to be free
- DPO: pair builder (`finetune/build_dpo_data.py`) and Kaggle train-and-export job (`finetune/kaggle_dpo/`, D-73); pairs collect from guarded runs (D-54)
- `scripts/make_tables.py`: report tables (Markdown + LaTeX) straight from the result files; `scripts/human_agreement.py`: Cohen's κ from the labelling page
- Gold data: `gold/operations.json`, `gold/checks.json` (268 checks, Claude-drafted, please spot-check), `gold/reference_plans/fastapi.json`
- Agent (`agent/`): retrieval ×6 + dependency closure, precision layer ×5 (prompt v2), planner, lint + static repair, **spec clean-up** (impossible-check drop, path autofill, undefined/invented actors, misdirected references), compiler + runtime, restricted refinement + guard (now also restricted to fields in the spec) + value-copy detector, judges ×3
- Fine-tuning: data builder, Kaggle job, adapter installer (Kaggle → GGUF → Ollama, backup to HF)

## Problems found and fixed
- Docker socket permissions (you ran `setfacl`); TodoApp root-owned files → chown before each reset
- Ollama prompt cache used 8.6 GB RAM → capped at 2 GB
- Smoke 1 → 2: format checks failing on the correct API counted as API bugs; impossible checks; unrepairable references
- Smoke 2 → 3: the precision prompt's examples were being copied; checks reading fields the spec does not have; missing path parameters never repaired; invented login credentials; `{{s1.req.x}}` vs `{{s1.x}}` mix-ups; the plan parser broke on one bad placeholder; `{{s1.data[0].id}}` was never resolved; the test command bug above

## Needs you
- **Human labels (replaces the gold spot-check):** open https://claude.ai/artifact/51FenN3D8LqZBbgCgHvgEq. It has 45 requirements and 181 labels: whether the gold checks are correct, and how well the precision layer covers them. It is private: share it with teammates as **Contributor** so they can save labels; each person sees only their own answers. Two labellers give a human–human κ too (D-71).
- **Compute (measured, no action needed now):** our agent ≈ 1 min per generated test; interactive runs need 204 tests on FastAPI, 94 on NestJS, 36 on TodoApp (one per requirement + one per mutant). Remaining work ≈ 380 run-hours (sweep 95 h, our finals + ablations 145 h, B2/B3 133 h, B4 9 h). Two services can run side by side, so it **fits on this PC alone if the queue runs day and night**: baselines until ~27 Oct, the sweep by 4 Nov (it is FastAPI-only, so it runs one at a time while other-service jobs fill the second slot), and finals by ~10 Nov. The 4060 laptop is only a buffer if something slips.
