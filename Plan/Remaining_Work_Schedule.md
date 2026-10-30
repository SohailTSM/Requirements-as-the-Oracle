# Remaining work: time and dependencies

_As of Sat 24 Oct 2026, 18:30 IST. GPU-hours are estimates: baselines measured on Kaggle T4s (D-90); our agent scaled from smoke runs (~1 min per generated test; 204 / 94 / 36 tests per FastAPI / NestJS / TodoApp run). Updated as runs finish._

## Tasks

| # | Task | Where (service copies) | GPU-hours | Calendar time (if its copies run nonstop) | Must wait for | Can run alongside |
|---|---|---|---|---|---|---|
| T1 | Remaining baselines on NestJS (B2/B3 repeats, fuzzer) | NestJS ×2 | ~47 h | ~24 h | nothing (running now) | everything |
| T2 | Remaining baselines on FastAPI (B2/B3 repeats + requirement-15 parts) | FastAPI copies | ~45 h | ~23 h on 2 copies | nothing; **better run after T5**, so the sweep goes first | T1, T4, T6 |
| T3 | Freeze the default configuration and planner (last fixes, check smokes) | FastAPI (1 copy) | ~2 h | ~0.5 day of development | smoke results (in progress) | T1, T2 |
| T4 | B4 response-judge baseline (6 runs) | FastAPI 9 h, NestJS 4.8 h, TodoApp 2.4 h | ~16 h | ~5 h | **T3** (B4 uses our planner) | T1, T5 |
| T5 | Component sweep, 26 runs (all rows except the 3 fine-tuned-model rows; includes default ×3) | FastAPI only | ~91 h | ~46 h on 2 copies, ~23 h on 4 | **T3** | T1, T4, T6 |
| T6 | DPO training data runs (agent with visible values + guard) | NestJS 1.6 h, TodoApp 0.6 h | ~2 h | ~2 h | **T3** | T5 |
| T7 | DPO training + export (Kaggle, training quota) | Kaggle T4 | ~3 h (separate quota) | ~3 h | **T6** and the guarded sweep runs in **T5** | T5, T8 |
| T8 | Sweep rows with the SFT model: `precision: finetuned`, `model: sft` (2 runs) | FastAPI | ~7 h | ~3.5 h on 2 copies | **T3** (SFT models installed ✓) | T5 |
| T9 | Sweep row `model: sft_dpo` (1 run) | FastAPI | ~3.5 h | ~3.5 h | **T7** | T5 |
| T10 | Choose the final configuration + 2 runner-ups (sweep spread from default ×3) | – | – | ~0.5 day | **T5, T8, T9** | – |
| T11 | Our finals: 3 configurations × 2 wordings × 3 runs, all services (18 runs) | FastAPI 63 h, NestJS 28.8 h, TodoApp 10.8 h | ~103 h | ~32 h with 2/2/1 copies (FastAPI-bound) | **T10** | T12 |
| T12 | Ablations: final configuration minus one component (8 runs, vague) | FastAPI 28 h, NestJS 12.8 h, TodoApp 4.8 h | ~46 h | ~14 h | **T10** | T11 |
| T13 | fp16 vs 4-bit comparison (final configuration, vague, 1 run) | all services; needs an fp16 server (8B in fp16 ≈ 16 GB: both T4s of one session) | ~8 h | ~8 h | **T10** + an fp16 server notebook | T11, T12 |
| T14 | Human labels on the labelling page (≥ 100 labels) | teammates, no GPU | – | – | nothing | everything |
| T15 | Statistics, agreement (κ), final report and slides | – | – | ~2 days | **T11, T12, T13, T14** | – |
| T11a | *Optional:* default configuration's finals on NestJS + TodoApp during the sweep | NestJS 9.6 h, TodoApp 3.6 h | ~13 h | ~5 h | **T3** | T5 (wasted only if the default is not among the final three) |

**Total ≈ 368 GPU-hours** (FastAPI ~251 h, NestJS ~97 h, TodoApp ~20 h).

## Critical path

```
T3 (freeze) ──► T5 sweep (FastAPI) ─────────────────────┐
           ├──► T6 DPO data ──► T7 DPO training ──► T9 ─┤
           └──► T8 SFT rows ────────────────────────────┴──► T10 choose ──► T11 finals + T12 ablations + T13 fp16 ──► T15 report
T1, T2, T4, T14 run alongside (no effect on the critical path if T2 waits until after T5)
```

The longest chain is **T3 → T5 → T10 → T11 → T15**. The sweep (T5) and our finals on FastAPI (T11) are the two biggest blocks, so extra FastAPI copies shorten the project most. With 4 FastAPI copies the sweep takes about 1 day of server time instead of 2.
