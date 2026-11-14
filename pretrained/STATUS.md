# Status: experiments frozen, final report written

_Last update: Wed 11 Nov 2026, IST. Every decision and its reason is logged in [`Plan/Decision_Log.md`](../Plan/Decision_Log.md) (D-01 … D-128)._

## Done
- **Experiments frozen (11 Nov).**
  - All baselines are complete (B1–B4, E), 3 runs each.
  - Our finals are complete: final configuration, 2 runner-ups, unrestricted follow-up; 3 runs each, both wordings, all services.
  - Also done: 7 ablations, the 29-variant component sweep (including the 3 fine-tuned rows) and one fp16 run.
  - Nothing is running: no Kaggle sessions, no queue runners, the dispatcher has no servers.
- **Human validation (D-127).**
  - Sohail and Kathan labelled 181 items each. Human–human κ = 0.84, and 125 of 126 gold checks were judged correct.
  - The Llama judge agreed poorly with them (κ 0.03 / 0.27), so the report gives the human-labelled precision rates.
- **Statistics (D-128).**
  - `scripts/final_stats.py` writes `results/tables/stats.md`: paired Wilcoxon per requirement with Holm correction, bootstrap CIs, adaptation gaps, value copying, step-level metrics, failure taxonomy and human-labelled precision.
  - `scripts/make_figures.py` writes the report figures.
- **Final report:** `Reports/Final/PreTrained_Final.tex` / `.pdf` (9 pages including references).
- **Final slides:** `Presentation/PreTrained_Final.pptx` (16 slides, with speaker notes; kept out of git like the proposal deck). Generator: `Presentation/build_final_deck.js`.
- **Repository:** root `README.md` and `pretrained/requirements.txt`.

## Headline results (vague, mean ± sd over 3 runs)
| | Caught % | Mutant-gen % | Adaptation gap |
|---|---|---|---|
| B1 single-step | 2.6 ± 0.9 | – | – |
| B2 refinement | 11.3 ± 1.4 | 4.3 ± 2.0 | 7.0 (p = 0.004) |
| Ours (final) | 12.7 ± 2.7 | 14.8 ± 1.2 | −2.1 (n.s.) |

- Ours vs B1: +10.1 points (p < 0.001).
- Ours vs B2: a draw on the correct service; +10.5 points on the buggy service (p < 0.001).
- Precise wording: B2 is better (31.3 vs 19.2).
- Ablations: the executor (−11.8) and spec clean-up (−7.5) carry the gain. The rest is within noise.

## Left for the team
- **Review** the report and slides before Sun 15 Nov.
- **Oral exam prep:** an AI examiner will question each of you on the submission.
- **Commit and push:** only when Sohail says so.
