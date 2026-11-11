<!-- Component sweep on FastAPI (vague), one run per variant -->
| Variant | Wording | Caught % | Δ vs default | Mutant-gen % | Fail-OK % |
|---|---|---|---|---|---|
| default | vague | 13.5 | +1.4 | 10.1 | 76.2 |
| default__rep2 | vague | 11.3 | -0.7 | 12.3 | 69.8 |
| default__rep3 | vague | 11.3 | -0.7 | 11.4 | 76.2 |
| architecture__blackboard | vague | 3.5 | -8.5 | 5.2 | 87.3 |
| architecture__multi_persona | vague | 10.6 | -1.4 | 8.6 | 76.2 |
| architecture__react | vague | 5.0 | -7.1 | 6.5 | 81.0 |
| communication__a2a | vague | 14.9 | +2.8 | 14.4 | 69.8 |
| communication__mcp | vague | 12.1 | -0.0 | 14.3 | 74.6 |
| content__full | vague | 18.4 | +6.4 | 15.7 | 71.4 |
| content__verdict | vague | 13.5 | +1.4 | 12.1 | 71.4 |
| generation__raw | vague | 0.0 | -12.1 | – | 100.0 |
| judge__none | vague | 17.0 | +5.0 | 10.9 | 69.8 |
| judge__restestbench | vague | 14.9 | +2.8 | 8.1 | 77.8 |
| model__sft | vague | 15.6 | +3.5 | 14.2 | 69.8 |
| model__sft_dpo | vague | 14.2 | +2.1 | 16.4 | 69.8 |
| precision__direct_rewrite | vague | 11.3 | -0.7 | 9.6 | 73.0 |
| precision__finetuned | vague | 17.0 | +5.0 | 16.0 | 66.7 |
| precision__none | vague | 12.8 | +0.7 | 13.9 | 79.4 |
| precision__self_consistency | vague | 17.0 | +5.0 | 16.5 | 65.1 |
| refinement__hidden | vague | 11.3 | -0.7 | 12.5 | 77.8 |
| refinement__none | vague | 9.2 | -2.8 | – | 79.4 |
| refinement__prompt_fixed | vague | 15.6 | +3.5 | 12.7 | 71.4 |
| refinement__strengthen_only | vague | 7.1 | -5.0 | 15.5 | 82.5 |
| refinement__unrestricted | vague | 15.6 | +3.5 | 12.6 | 71.4 |
| retrieval__bm25 | vague | 13.5 | +1.4 | 11.5 | 74.6 |
| retrieval__dense | vague | 16.3 | +4.3 | 13.6 | 69.8 |
| retrieval__full | vague | 7.1 | -5.0 | 12.4 | 77.8 |
| retrieval__hybrid | vague | 13.5 | +1.4 | 13.6 | 69.8 |
| retrieval__query_rewrite | vague | 14.9 | +2.8 | 14.1 | 66.7 |
