<!-- Ablations (vague, one run each): the final configuration minus one component; Δ against its 3-run mean -->
| Removed | FastAPI | NestJS | TodoApp | All | Δ vs final | Mutant-gen | Fail-OK |
|---|---|---|---|---|---|---|---|
| no_executor | 0.7 | 0.0 | 4.5 | 0.9 | -11.8 | – | 98.1 |
| no_format_relaxation | 14.9 | 4.6 | 27.3 | 13.2 | +0.4 | 15.0 | 74.5 |
| no_judge | 17.0 | 6.2 | 22.7 | 14.5 | +1.8 | 10.4 | 65.1 |
| no_precision | 10.6 | 9.2 | 36.4 | 12.7 | +0.0 | 16.5 | 75.5 |
| no_retrieval | 13.5 | 6.2 | 27.3 | 12.7 | +0.0 | 13.8 | 71.7 |
| no_spec_cleanup | 4.3 | 0.0 | 27.3 | 5.3 | -7.5 | 6.9 | 85.8 |
| unrestricted_refinement | 19.1 | 12.3 | 54.5 | 20.6 | +7.9 | 16.4 | 60.4 |
