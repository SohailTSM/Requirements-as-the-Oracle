<!-- Planted bugs caught (%), local Llama 3.1 8B (Q4_K_M); mean ± sd over runs -->
| Tool | Wording | FastAPI | NestJS | TodoApp | All | Mutant-gen | Fail-OK | Runs |
|---|---|---|---|---|---|---|---|---|
| B1 single-step | precise | 39.7 ± 2.6 | 5.1 ± 1.8 | 1.5 ± 2.6 | 26.2 ± 1.8 | – | 77.4 ± 2.8 | 3 |
| B2 refinement | precise | 47.8 ± 2.2 | 6.2 ± 2.7 | 0.0 ± 0.0 | 31.3 ± 0.7 | 24.8 ± 0.6 | 68.6 ± 2.0 | 3 |
| B3 prompt-fixed | precise | 37.6 ± 2.5 | 5.6 ± 4.9 | 0.0 ± 0.0 | 24.9 ± 0.3 | 25.4 ± 3.0 | 78.0 ± 1.1 | 3 |
| B4 response judge | precise | 13.5 ± 2.5 | 10.3 ± 5.8 | 4.5 ± 0.0 | 11.7 ± 2.9 | – | 56.0 ± 3.3 | 3 |
| Ours (final) | precise | 20.6 ± 5.1 | 22.1 ± 3.9 | 1.5 ± 2.6 | 19.2 ± 2.2 | 16.0 ± 3.0 | 77.7 ± 2.9 | 3 |
| Ours − self-consistency | precise | 18.9 ± 3.6 | 25.6 ± 0.9 | 1.5 ± 2.6 | 19.2 ± 2.1 | 16.7 ± 0.5 | 77.7 ± 2.0 | 3 |
| Ours − full-trace content | precise | 17.0 ± 3.7 | 12.3 ± 1.5 | 10.6 ± 9.5 | 15.1 ± 1.8 | 14.8 ± 1.3 | 79.6 ± 1.1 | 3 |
| Ours, unrestricted refinement | precise | 19.6 ± 4.7 | 17.4 ± 0.9 | 4.5 ± 7.9 | 17.5 ± 2.3 | 18.7 ± 3.1 | 79.2 ± 1.9 | 3 |
| B1 single-step | vague | 4.3 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.6 ± 0.9 | – | 97.2 ± 0.9 | 3 |
| B2 refinement | vague | 17.5 ± 2.3 | 0.0 ± 0.0 | 4.5 ± 0.0 | 11.3 ± 1.4 | 4.3 ± 2.0 | 85.2 ± 0.5 | 3 |
| B3 prompt-fixed | vague | 4.5 ± 2.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.8 ± 1.8 | 3.1 ± 0.9 | 96.9 ± 1.1 | 3 |
| B4 response judge | vague | 6.6 ± 0.4 | 3.1 ± 4.1 | 18.2 ± 0.0 | 6.7 ± 1.1 | – | 40.6 ± 1.9 | 3 |
| E fuzzer | vague | 13.9 ± 7.8 | 0.0 ± 0.0 | 9.1 ± 0.0 | 9.5 ± 4.8 | – | 70.8 ± 10.6 | 3 |
| Ours (final) | vague | 12.5 ± 3.2 | 2.6 ± 3.2 | 43.9 ± 6.9 | 12.7 ± 2.7 | 14.8 ± 1.2 | 70.4 ± 3.6 | 3 |
| Ours, fp16 weights | vague | 12.8 | 7.7 | 36.4 | 13.6 | 14.3 | 70.8 | 1 |
| Ours − self-consistency | vague | 12.8 ± 0.0 | 7.2 ± 3.6 | 37.9 ± 2.6 | 13.6 ± 0.8 | 12.9 ± 0.4 | 74.8 ± 2.4 | 3 |
| Ours − full-trace content | vague | 18.0 ± 1.6 | 6.2 ± 4.1 | 36.4 ± 7.9 | 16.4 ± 1.7 | 14.5 ± 1.1 | 67.0 ± 1.6 | 3 |
| Ours, unrestricted refinement | vague | 15.8 ± 6.0 | 6.7 ± 1.8 | 39.4 ± 2.6 | 15.5 ± 3.3 | 14.9 ± 1.1 | 68.2 ± 2.0 | 3 |
