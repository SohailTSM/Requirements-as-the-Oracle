# Final statistics (scripts/final_stats.py)

## Ours (final) vs others, vague wording (n = requirements; Δ = ours − other, pooled caught %)

| Other | Runs | Δ caught % | 95% CI | ours better / worse (req.) | Wilcoxon p (Holm) | r |
|---|---|---|---|---|---|---|
| B1 single-step | 3 | +10.1 | [+5.2, +15.2] | 35 / 4 | <0.001 (<0.001) | +0.75 |
| B2 refinement | 3 | +1.5 | [-5.3, +7.9] | 30 / 22 | 0.378 (1.000) | +0.09 |
| B3 prompt-fixed | 3 | +9.9 | [+5.5, +14.6] | 33 / 5 | <0.001 (<0.001) | +0.85 |
| B4 response judge | 3 | +6.0 | [+1.0, +11.1] | 32 / 11 | 0.005 (0.032) | +0.45 |
| E fuzzer | 3 | +3.2 | [-2.5, +9.1] | 26 / 19 | 0.303 (1.000) | +0.20 |
| Ours, fp16 weights | 1 | -0.9 | [-5.9, +4.1] | 22 / 16 | 0.543 (1.000) | -0.05 |
| Ours − self-consistency | 3 | -0.9 | [-5.0, +3.3] | 18 / 20 | 0.778 (1.000) | -0.05 |
| Ours − full-trace content | 3 | -3.7 | [-7.5, +0.0] | 13 / 21 | 0.204 (1.000) | -0.33 |
| Ours, unrestricted refinement | 3 | -2.8 | [-6.3, +0.6] | 16 / 23 | 0.268 (1.000) | -0.27 |

## Mutant-gen (tests generated against the buggy service), ours (final) vs others, vague

| Other | Δ mutant-gen % | 95% CI | ours better / worse (req.) | Wilcoxon p (Holm) | r |
|---|---|---|---|---|---|
| B2 refinement | +10.5 | [+5.6, +15.7] | 42 / 10 | <0.001 (<0.001) | +0.69 |
| B3 prompt-fixed | +11.7 | [+6.8, +16.8] | 45 / 5 | <0.001 (<0.001) | +0.80 |
| Ours, fp16 weights | +0.5 | [-3.6, +4.6] | 30 / 19 | 0.172 (0.690) | +0.18 |
| Ours − self-consistency | +1.9 | [-1.0, +4.9] | 26 / 19 | 0.252 (0.757) | +0.27 |
| Ours − full-trace content | +0.3 | [-2.4, +2.9] | 27 / 21 | 0.540 (1.000) | +0.03 |
| Ours, unrestricted refinement | -0.1 | [-2.6, +2.5] | 25 / 22 | 0.693 (1.000) | +0.06 |

## Ours (final) vs others, precise wording (n = requirements; Δ = ours − other, pooled caught %)

| Other | Runs | Δ caught % | 95% CI | ours better / worse (req.) | Wilcoxon p (Holm) | r |
|---|---|---|---|---|---|---|
| B1 single-step | 3 | -7.0 | [-16.2, +2.3] | 26 / 28 | 0.540 (1.000) | -0.19 |
| B2 refinement | 3 | -12.1 | [-20.6, -3.4] | 20 / 37 | 0.009 (0.062) | -0.49 |
| B3 prompt-fixed | 3 | -5.7 | [-14.5, +3.5] | 23 / 27 | 0.483 (1.000) | -0.17 |
| B4 response judge | 3 | +7.5 | [+2.0, +12.8] | 30 / 14 | 0.042 (0.251) | +0.27 |
| Ours − self-consistency | 3 | +0.0 | [-5.9, +5.7] | 23 / 19 | 0.563 (1.000) | +0.11 |
| Ours − full-trace content | 3 | +4.1 | [+0.0, +8.1] | 25 / 14 | 0.113 (0.567) | +0.30 |
| Ours, unrestricted refinement | 3 | +1.6 | [-3.0, +6.6] | 22 / 21 | 0.808 (1.000) | +0.09 |

## Mutant-gen (tests generated against the buggy service), ours (final) vs others, precise

| Other | Δ mutant-gen % | 95% CI | ours better / worse (req.) | Wilcoxon p (Holm) | r |
|---|---|---|---|---|---|
| B2 refinement | -8.8 | [-16.8, -0.7] | 32 / 35 | 0.447 (1.000) | -0.16 |
| B3 prompt-fixed | -9.4 | [-17.2, -1.4] | 29 / 33 | 0.328 (1.000) | -0.22 |
| Ours − self-consistency | -0.8 | [-3.8, +2.4] | 26 / 25 | 0.859 (1.000) | +0.04 |
| Ours − full-trace content | +1.2 | [-1.4, +3.9] | 26 / 26 | 0.722 (1.000) | +0.16 |
| Ours, unrestricted refinement | -2.7 | [-5.7, +0.4] | 21 / 31 | 0.156 (0.778) | -0.25 |

## Adaptation gap (caught % − mutant-gen %), mean ± sd over runs

| Tool | Wording | Caught | Mutant-gen | Gap | Wilcoxon p (caught vs mutant-gen, per req.) |
|---|---|---|---|---|---|
| B2 refinement | precise | 31.3 ± 0.7 | 24.8 ± 0.6 | 6.5 ± 1.3 | 0.004 |
| B3 prompt-fixed | precise | 24.9 ± 0.3 | 25.4 ± 3.0 | -0.5 ± 2.9 | 0.722 |
| Ours (final) | precise | 19.2 ± 2.2 | 16.0 ± 3.0 | 3.2 ± 3.0 | 0.299 |
| Ours − self-consistency | precise | 19.2 ± 2.1 | 16.7 ± 0.5 | 2.4 ± 1.8 | 0.894 |
| Ours − full-trace content | precise | 15.1 ± 1.8 | 14.8 ± 1.3 | 0.3 ± 1.2 | 0.709 |
| Ours, unrestricted refinement | precise | 17.5 ± 2.3 | 18.7 ± 3.1 | -1.1 ± 2.1 | 0.197 |
| B2 refinement | vague | 11.3 ± 1.4 | 4.3 ± 2.0 | 7.0 ± 2.8 | 0.004 |
| B3 prompt-fixed | vague | 2.8 ± 1.8 | 3.1 ± 0.9 | -0.3 ± 1.3 | 0.856 |
| Ours (final) | vague | 12.7 ± 2.7 | 14.8 ± 1.2 | -2.1 ± 3.4 | 0.058 |
| Ours, fp16 weights | vague | 13.6 | 14.3 | -0.7 | 0.634 |
| Ours − self-consistency | vague | 13.6 ± 0.8 | 12.9 ± 0.4 | 0.7 ± 0.4 | 0.875 |
| Ours − full-trace content | vague | 16.4 ± 1.7 | 14.5 ± 1.1 | 1.9 ± 2.3 | 0.919 |
| Ours, unrestricted refinement | vague | 15.5 ± 3.3 | 14.9 ± 1.1 | 0.6 ± 3.8 | 0.682 |

## Gap closed (vague): (12.7 − 2.6) / (31.3 − 2.6) = 35% of the way from B1 vague to the best precise score

## Refinement safety: rounds that copy an observed value into an expectation

| Tool | Wording | Refinement rounds | Rounds attempting a copied value | Rate | Copies in the next executed plan |
|---|---|---|---|---|---|
| B2 refinement | vague | 1931 | 261 | 13.5% | all (no guard) |
| B2 refinement | precise | 1825 | 151 | 8.3% | all (no guard) |
| B3 prompt-fixed | vague | 33 | 2 | 6.1% | all (no guard) |
| B3 prompt-fixed | precise | 74 | 8 | 10.8% | all (no guard) |
| Ours (final) | vague | 739 | 21 | 2.8% | 3 of 787 (0.4%) |
| Ours (final) | precise | 960 | 57 | 5.9% | 8 of 972 (0.8%) |
| Ours, unrestricted refinement | vague | 743 | 33 | 4.4% | 33 of 807 (4.1%) |
| Ours, unrestricted refinement | precise | 903 | 135 | 15.0% | 135 of 932 (14.5%) |

## Final configuration, intermediate steps (vague, 3 runs, tests generated on correct and buggy services)

- generations: 1002 (318 on the correct service); planner failed: 29 (2.9%)
- plans with lint issues: 516 (51%), 1041 issues, LLM lint repair used on 515
- spec clean-up events: {'unstated_status_generalized': 252, 'reference_redirected': 143, 'undocumented_status_format_check_dropped': 130, 'invented_credentials_replaced': 253, 'impossible_check_dropped': 443, 'path_param_autofilled': 3, 'redundant_signup_removed': 194, 'check_on_removed_signup_dropped': 73, 'actor_defined_as_new_user': 2}
- rounds: 1760; rounds with a setup error: 1019 (58%); kinds {'reference to a missing field': 662, 'setup call rejected by the service': 363, 'other': 94}
- judge on the correct service: 117 API_BUG of 117 verdicts (every failure there is the test's fault, so these are wrong blames); on buggy services: 295 of 295
- refinement rounds: 739; guard events {'locked_check_restored': 497, 'check_added': 157, 'locked_check_reference_repaired': 62, 'new_check_with_copied_value_blocked': 9, 'field_repair_rejected_not_in_spec': 4, 'format_check_dropped': 1, 'value_copy_blocked': 2, 'format_check_changed_with_spec_evidence': 1}; tampering attempts blocked: 512, accepted: 0
- cost per generated test: 7.7 LLM calls, 13918 tokens, 123.2 s

Outcome of each test generated on the correct service (vague): {'fails on correct service: wrong check, judge blamed the API': 112, 'passes, catches ≥1 planted bug': 63, 'fails on correct service: setup error': 85, 'passes, catches no planted bug (checks too weak)': 31, 'no valid plan (planner failed)': 15, 'fails on correct service: wrong check, rounds exhausted': 12}

## Final configuration, intermediate steps (precise, 3 runs, tests generated on correct and buggy services)

- generations: 1002 (318 on the correct service); planner failed: 31 (3.1%)
- plans with lint issues: 629 (63%), 1483 issues, LLM lint repair used on 623
- spec clean-up events: {'unstated_status_generalized': 230, 'undocumented_status_format_check_dropped': 184, 'impossible_check_dropped': 841, 'reference_redirected': 194, 'redundant_signup_removed': 262, 'check_on_removed_signup_dropped': 188, 'invented_credentials_replaced': 14, 'path_param_autofilled': 9}
- rounds: 1943; rounds with a setup error: 1358 (70%); kinds {'setup call rejected by the service': 891, 'reference to a missing field': 425, 'other': 64}
- judge on the correct service: 95 API_BUG of 96 verdicts (every failure there is the test's fault, so these are wrong blames); on buggy services: 317 of 318
- refinement rounds: 960; guard events {'check_added': 398, 'locked_check_restored': 699, 'locked_check_reference_repaired': 33, 'new_check_with_copied_value_blocked': 64, 'value_copy_blocked': 2, 'format_check_changed_with_spec_evidence': 3, 'field_repair_rejected_not_in_spec': 8, 'format_check_dropped': 2}; tampering attempts blocked: 773, accepted: 0
- cost per generated test: 8.0 LLM calls, 16506 tokens, 183.2 s

Outcome of each test generated on the correct service (precise): {'fails on correct service: setup error': 136, 'fails on correct service: wrong check, judge blamed the API': 93, 'fails on correct service: wrong check, rounds exhausted': 3, 'passes, catches ≥1 planted bug': 61, 'passes, catches no planted bug (checks too weak)': 10, 'no valid plan (planner failed)': 15}

## Precision layer on the human-labelled sample (catalogue prompt v2, vague)

| Labeller | Gold checks | Fully covered | Soft recall | Derived checks | Match | Extra | Contradiction |
|---|---|---|---|---|---|---|---|
| u_QEqDg48GphuOyh0KmLImjA | 51 | 0.41 | 0.58 | 67 | 0.67 | 0.24 | 0.09 |
| u_zD8qKuBsV8dI9zYbc9wbEQ | 51 | 0.29 | 0.54 | 67 | 0.69 | 0.22 | 0.09 |
| llm_judge_same_items | 51 | 0.45 | 0.55 | 67 | 0.61 | 0.03 | 0.30 |

