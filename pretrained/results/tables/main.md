<!-- Planted bugs caught (%), local Llama 3.1 8B (Q4_K_M); mean ± sd over runs -->
| Tool | Wording | FastAPI | NestJS | TodoApp | All | Mutant-gen | Fail-OK | Runs |
|---|---|---|---|---|---|---|---|---|
| B1 single-step | precise | 39.7 ± 2.6 | 5.1 ± 1.8 | 1.5 ± 2.6 | 26.2 ± 1.8 | – | 77.4 ± 2.8 | 3 |
| B2 refinement | precise | 49.6 | 6.2 ± 2.7 | 0.0 ± 0.0 | 32.0 | 24.3 | 67.0 | 3 |
| B3 prompt-fixed | precise | 36.2 | 5.6 ± 4.9 | 0.0 ± 0.0 | 25.0 | 28.9 | 79.2 | 3 |
| B4 response judge | precise | – | 10.3 ± 5.8 | 4.5 ± 0.0 | – | – | – | 3 |
| B1 single-step | vague | 4.3 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.6 ± 0.9 | – | 97.2 ± 0.9 | 3 |
| B2 refinement | vague | 17.0 ± 3.0 | 0.0 ± 0.0 | 4.5 ± 0.0 | 11.0 ± 1.9 | 5.4 ± 0.1 | 85.4 ± 0.7 | 3 |
| B3 prompt-fixed | vague | 4.3 ± 4.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.6 ± 2.5 | 3.4 ± 1.0 | 97.2 ± 1.3 | 3 |
| B4 response judge | vague | – | 3.1 ± 4.1 | 18.2 ± 0.0 | – | – | – | 3 |
| E fuzzer | vague | 13.9 ± 7.8 | 0.0 ± 0.0 | 9.1 ± 0.0 | 9.5 ± 4.8 | – | 70.8 ± 10.6 | 3 |
