# Deep Dive: Requirements as the Oracle

*Research notes behind Team PreTrained's project. Literature searched up to 1 Oct 2026; RESTestBench analysis from the authors' released code and raw result files. Last updated 7 Oct 2026, aligned with the proposal (`Proposal/PreTrained.pdf`) and the presentation (`Presentation/PreTrained.pptx`).*

---

## 0. TL;DR

**What exists today**
- **Crash-finding is solved well enough.** AutoRestTest won the first SBFT 2026 REST tool competition, finding about **2.7× more faults** than the next tool. But its "faults" are HTTP 500 crashes. An API that wrongly returns `200 OK` passes.
- **Checking correct behaviour is the open frontier.** LLM response judges (LogiAgent, ARMeta) produce many false alarms. Oracle generators need the source code (MASTOR) or ignore requirements (SATORI).
- **RESTestBench (EASE 2026)** is the public benchmark for turning natural-language requirements into API tests: 3 services, 106 requirements (each vague and precise), 228 planted bugs, 10 LLMs.

**What RESTestBench shows** (verified against their raw results)
- Single-step generation: the best model catches **92%** of planted bugs with precise requirements, but only **2–54%** with vague ones.
- Refinement (write → run → judge → rewrite) helps on vague requirements, but **when the API is buggy the test learns the bug**. On vague requirements every model loses **6–20 points** when refining against the buggy API instead of the correct one.
- Their prompts actively encourage this: the judge's own example teaches copying the observed value, and there is no "API bug" outcome.
- Small models fail even earlier: **44–100% of Llama 3.1 8B's tests fail on the correct API** (wrong login, URLs, payloads).

**What we build:** an agent that retrieves only the relevant API operations, derives and **locks** the checks a vague requirement implies, writes a structured plan that **tools** turn into requests, and **restricts refinement** so execution feedback can fix *how* a test runs but never *what it expects*. An evidence-restricted **judge** decides "API bug or test bug". A fine-tuned **Llama 3.1 8B** (SFT + DPO) keeps it cheap. Every component is evaluated with several variants and its own metric.

---

## 1. Background in plain words

- A **REST API** is how apps talk to a backend: `POST /items`, `GET /items/{id}`, `DELETE /items/{id}`. Its structure is described in an **OpenAPI** file (endpoints, parameters, data types, response codes).
- A **test** has two parts:
  - **Actions and inputs**: e.g. "log in as user A, create an item, log in as user B, try to delete A's item".
  - The **expected result (oracle)**: e.g. "B's request is refused and the item still exists".
- **The oracle problem:** generating inputs is the easy half. Knowing the *correct* expected behaviour is the hard half. The OpenAPI file says what's *possible* (types, status codes), not what's *correct* ("only the owner can delete").
- Business rules live in **requirements** (user stories, acceptance criteria). That's why an LLM is needed: it can read them, plan the call sequence, and turn rules into checks.
- **The catch:** LLM-written checks tend to describe what the code *does*, not what it *should* do. Shown buggy behaviour, they copy it (Konstantinou et al., 2024).

---

## 2. The landscape: what exists and how well it performs

*All figures as reported by each paper on its own setup. They are not directly comparable across rows.*

### 2.1 Generation 1: classic fuzzers (oracle = "did it crash?")

| Tool | Idea | Reported performance |
|---|---|---|
| **EvoMaster** | Search-based test generation, black-box and white-box | ISSTA 2022 study of 10 tools on 20 services: best tool (EvoMaster white-box) reached **< 53% line and 37% branch coverage** in 1 h. 2nd at SBFT 2026. |
| **RESTler** (Microsoft) | Stateful fuzzing with producer–consumer dependencies | 34.7% method coverage in the AutoRestTest study |
| **Schemathesis** | Property-based testing from the OpenAPI schema | Widely used. Oracle = schema conformance + server errors. |
| **RESTest**, **CATS** | Model-based / catalogue of negative tests | Competed at SBFT 2026 |
| **ARAT-RL** (ASE 2023) | Reinforcement learning to prioritise operations | 42.1% method coverage, 33 server errors (AutoRestTest study) |

**Infrastructure:** EMB/WFD (36 runnable REST APIs), RESTgym (11 APIs + 6 tools in Docker; used at SBFT 2026).

### 2.2 Generation 2: LLM-boosted fuzzers (better inputs, same crash oracle)

| Tool | What the LLM does | Reported performance |
|---|---|---|
| **RESTGPT** (ICSE-NIER 2024) | Extracts rules and example values from OpenAPI descriptions | 97% rule-extraction precision, 73% value accuracy |
| **AutoRestTest** (ICSE 2025; SBFT 2026 winner) | 4 RL agents + semantic dependency graph + LLM input values (GPT-3.5, few-shot) | 12 services: **58.3% method / 32.1% branch coverage**, 42 server errors. SBFT 2026: **67.09 vs. 24.65** average faults; 1st in all three categories. |
| **LlamaRestTest** (FSE 2025) | Two fine-tuned Llama3-8B models for inputs and dependencies | Valid inputs **72.4%** vs. 22.9% for base Llama3-8B |
| **APIPilot** (2026) | Verifies LLM-inferred dependencies by execution | 16 services: 92.3% operation coverage, 197 server errors |
| **RESTifAI** (2025) | Happy paths + negative tests (2xx and 4xx) | Authors note it lacks business-logic assertions |

➡️ **Takeaway:** we do not compete on coverage or crash-finding.

### 2.3 Generation 3: oracle generation and response judges

| Work | Inputs | Reported performance | Limitation for us |
|---|---|---|---|
| **AGORA+** | Execution traces | F1 69.3% | Learns from actual behaviour |
| **SATORI** (ASE 2025) | OpenAPI only | **F1 74.3%** on 12 industrial APIs | No requirements |
| **MASTOR** (2026) | Java source + OpenAPI | **75.4% mutation score** on 13 APIs | Needs the source; truth comes from the implementation |
| **RESTOR** (2026) | One request/response pair; RL-fine-tuned small model | Key-field F1 **85.4%** | No requirements |
| **LogiAgent** (2025) | OpenAPI + docs; LLM validator | 234 logical issues, **66.2% accuracy** | Judges *after* seeing the response |
| **ARMeta** (2026) | OpenAPI; metamorphic relations | True-positive rate **57–79%** | Only 2 APIs |

### 2.4 Generation 4: requirements → API tests

| Work | Setup | Reported performance |
|---|---|---|
| **APITestGenie** (2026) | User stories + OpenAPI, GPT-4-Turbo, RAG for large specs | 25 requirements, 10 APIs: 69.3% valid scripts; 16% of tests found real defects |
| **MASTEST** (2025) | OpenAPI → scenarios → pytest | Does not use NL requirements |
| **RESTestBench** (EASE 2026) | 3 services, 106 requirements × {vague, precise}, 228 mutants, 10 LLMs | See Section 3 |

---

## 3. RESTestBench in detail (our analysis)

### 3.1 What's in it
- **Services (Docker):** FastAPI full-stack template (Python, 63 requirements, 141 mutants), NestJS RealWorld (TypeScript, 29 / 65), TodoApp (.NET, 14 / 22).
- **Requirements:** each JSON file has `requirement_vague`, `requirement_precise` and its mutants (small code changes that break that one requirement). One reference test per requirement in `data/tests/`.
- **Scoring:** a test earns credit for each mutant it kills, **only if it also passes on the correct service**. Score = mutants killed ÷ 228.
- **Example (FastAPI #1):**
  - Vague: *"Creating a valid item yields an appropriate response"*
  - Precise: *"Register a new user, then authenticate… create a new item… Verify that the response status code is 200, and the returned item contains the same title and description as the input, the 'id' field is present, and the 'owner_id' matches the registered user's ID."*
  - Mutants: status changed to 400; description replaced with random text; `owner_id` set to the wrong user.

### 3.2 How they generate tests (prompts from `tools/`)
- **Single-step:** one zero-shot prompt with role, rules ("make assertions SPECIFIC and STRICT"), the requirement and the **entire OpenAPI spec pasted in**. The model writes the whole pytest file. Temperature 0.2 (GPT-5 models 1.0).
- **Refinement (3 rounds by default):** generate → run while capturing every HTTP request/response → a **judge** prompt answers `REFINEMENT` or `FINISH` → regenerate with the previous test, run output and full HTTP trace.

**Why refinement learns the bug**, from their code:
1. The judge's few-shot example: *"The test expects response.message == 'Authentication failed', but the actual response contains 'Authentication error'"* → REFINEMENT. It teaches copying the observed value.
2. Only the judge is warned the service may be buggy. The generator is told to *"use the Test Execution Output and the HTTP Trace to enhance the effectiveness of the test"*.
3. The judge's reasoning is discarded: `USE_JUDGE_FEEDBACK_IN_REFINEMENT = False`.
4. There is no "API bug" outcome, and an unclear judge answer defaults to REFINEMENT.

### 3.3 Results, recomputed from the raw files
Mean of 3 runs, all services pooled, % of the 228 mutants killed.

**Single-step generation**

| Model | Precise | Vague |
|---|---|---|
| Claude Sonnet 4.5 | **91.7** | 51.6 |
| GPT-5 Nano | 84.1 | 50.9 |
| DeepSeek V3.2 | 80.6 | **54.1** |
| GPT-5 Mini | 79.8 | 51.0 |
| GPT-OSS 120B | 77.5 | 48.7 |
| Kimi K2 | 72.5 | 41.2 |
| Mistral Medium | 61.3 | 30.4 |
| Llama 4 Scout | 42.5 | 14.8 |
| Llama 3.3 70B | 41.7 | 9.9 |
| Llama 3.1 8B | 13.3 | **2.0** |

**Bug adaptation (vague requirements, refinement)**

| Model | Refined on correct API | Refined on buggy API | Drop |
|---|---|---|---|
| GPT-5 Nano | 79.6 | 59.9 | −19.7 |
| GPT-OSS 120B | 68.6 | 51.1 | −17.5 |
| DeepSeek V3.2 | 74.6 | 58.3 | −16.3 |
| GPT-5 Mini | 75.0 | 61.9 | −13.1 |
| Kimi K2 | 59.3 | 47.0 | −12.3 |
| Llama 4 Scout | 29.8 | 19.2 | −10.6 |
| Mistral Medium | 45.0 | 34.7 | −10.3 |
| Sonnet 4.5 | 68.6 | 60.5 | −8.1 |
| Llama 3.3 70B | 23.2 | 17.1 | −6.1 |

The drop is consistent across all 9 models and far larger than the paper's maximum run-to-run variance (3.4 points). Several vague-refinement settings have only 2 complete runs, so exact values are approximate. Precise-requirement drops are smaller (best refined score: GPT-5 Mini 94.0% on the correct API).

**Llama 3.1 8B** (`Meta-Llama-3.1-8B-Instruct`, temperature 0.2, ~$0.15/M tokens), our model of choice:

| Requirement | Mode | Score (runs 2–3) | Tests failing on correct API |
|---|---|---|---|
| Vague | Single-step | 3.1% | 92–100% |
| Vague | Refined on correct API | 12.3% (run 3 only) | 75–100% |
| Vague | Refined on buggy API | 3.5% (run 3 only) | 89–100% |
| Precise | Single-step | ≈20% (21.5 / 18.4) | 70–100% |
| Precise | Refined on correct API | ≈32% (27.6 / 36.4) | 44–93% |
| Precise | Refined on buggy API | ≈28% (28.6 / 27.2) | 56–95% |

**Data-quality notes:**
- Run 1 for Llama 3.1 8B is broken (0 valid tests), likely predating a workaround in their code for "LangChain bugs with llama8b". The paper's **13%** for this model includes that run; without it, precise single-step is about **20%**.
- Run 2 has duplicate precise-refinement attempts; we use the later, complete one.
- When we compare against their baselines, we recompute from the raw files and state which runs are excluded.

---

## 4. The gaps we target

| Failure (measured) | Evidence |
|---|---|
| **Tests fail even on the correct API** | Llama 3.1 8B: 44–100% of its tests |
| **Vague requirements give weak tests** | Single-step: 2–54% across models |
| **Refinement learns the bug** | Vague: −6 to −20 points on the buggy API |
| **Small models are cheap but weak** | Llama 3.1 8B: 2% on vague single-step |
| **Response judges are noisy** | LogiAgent 66% accuracy; ARMeta 57–79% true positives |

---

## 5. Our approach

### 5.1 Pipeline

```
Vague requirement + OpenAPI spec
   │
   ├─(1) Spec retrieval (RAG) ── only the operations this requirement needs
   │
   ├─(2) Precision layer ─────── expand into atomic checks, tagged MEANING (locked)
   │                              or FORMAT (changeable only with spec evidence)
   │
   ├─(3) Structured plan ─────── actors, setup, action, checks (not raw code)
   │      + tool executor ─────── deterministic login, URLs, payloads, ID passing → pytest
   │
   ├─ run against the service (Docker sandbox)
   │
   ├─ setup / execution error ─→ (4) Restricted refinement: full error shown, fix setup only
   │
   ├─ assertion fails ─────────→ (5) Judge: API_BUG or TEST_BUG, citing only
   │                                  requirement/spec text; API_BUG ends the loop
   │
   └─ all checks pass ─────────→ pytest test (+ bug report)
```

**(1) Spec retrieval.** Each OpenAPI operation (path, method, resolved schemas, description) is one chunk. Replaces RESTestBench's "paste the whole spec".

**(2) Precision layer.** Before any execution, the LLM expands the vague requirement using a catalogue of implied properties:
- **Round-trip:** what you create or update is what you read back
- **Echo:** input fields come back unchanged
- **Ownership / isolation:** the owner is the current user; others can't see or change it
- **Before/after deltas:** e.g. the count goes up by exactly 1
- **Filter soundness and completeness:** every result matches; nothing matching is missing

These are relationships, not exact values, so a test doesn't need the precise version's numbers and can't copy them from a buggy response. **The benchmark's precise requirements are used only for evaluation, never as input.**

**(3) Structured plan + tool executor.** The LLM writes steps and checks; code builds the actual requests from the spec. Targets the "fails on the correct API" problem.

**(4) Restricted refinement.**
- Execution errors (401/422 on setup, exceptions, missing spec-declared fields) are shown in full and may be fixed.
- **Assertion failures are reported without the observed value** ("check M2 failed").
- A **strengthen-only guard** allows adding or tightening checks, never weakening them.
- A **value-copy detector** flags any new expected value equal to a value from the last response.

**(5) Evidence-restricted judge.** "The server returned it" is never valid evidence. A format detail may be relaxed only by quoting the spec. Uncertain cases abstain.

**(6) Fine-tuning.** Llama 3.1 8B Instruct (same model as RESTestBench's row), QLoRA:
- **SFT** on derived-check plans verified to pass on the correct service and kill at least one mutant
- **DPO** preferring "report API bug" over "adapt to the response", with pairs built automatically from runs (we always know whether the service was buggy). Include correct-API episodes where a spec-backed change *was* right, so the model doesn't learn "never change anything".
- **Leave-one-service-out** training to avoid leakage.

### 5.2 Worked example (FastAPI)

**Requirement (vague):** *"Users can only delete their own items."* **Mutant:** ownership check removed.

| Step | RESTestBench refinement | Ours |
|---|---|---|
| Checks | "expect 403" | Locked: M1 B's delete is refused (4xx); M2 A's item still exists. Format: exact code, dropped if the spec doesn't define it. |
| Run on buggy API | 200 OK → test fails | 200 OK → M1, M2 fail |
| Next | Generator sees the trace and rewrites to expect 200. **Bug baked in.** | Refiner sees only "M1 failed". Judge confirms the setup is right (B ≠ owner), finds no spec support for 200 → **API_BUG** |
| Outcome | Mutant survives | Mutant killed |

On the **correct** API (returns 400), M1 and M2 pass; the unspecified exact code is dropped, so no false alarm.

### 5.3 Where the five capabilities show up

| Capability | Role | Metric |
|---|---|---|
| **Tool calling** | HTTP executor, auth helper, ID store, spec lookup, test runner | Valid-request rate, setup-error rate, argument validity, error recovery |
| **Agentic** | Retrieve → derive → plan → run → judge, with state and abstention | Requirements completed, steps per requirement, abstention precision |
| **RAG** | Retrieve relevant operations from the spec | Recall@k, MRR, nDCG; context tokens |
| **Reasoning** | Vague requirement → checks; judge's blame decision | Check recall and contradiction rate vs. precise requirement; judge accuracy |
| **Fine-tuning** | SFT + DPO on the 8B model | Score, adaptation gap and cost vs. base 8B and an API model |

---

## 6. Research questions and hypotheses

> **RQ1 (vague requirements).** Given only a vague requirement and the OpenAPI spec, with no human input and no precise version at test time, can an LLM agent generate tests that catch requirement-violating bugs? How much of the vague-to-precise gap can it close?
>
> **RQ2 (safe refinement).** Does restricted refinement (locked checks, hidden observed values, strengthen-only edits, evidence-restricted judge) remove bug adaptation without increasing false alarms on the correct service?
>
> **RQ3 (component value).** Which components contribute measurably, and which variant of each works best? Can a fine-tuned 8B model approach a cheap API model?

- **H1:** implied-property derivation raises the vague-requirement score well above the single-step baseline.
- **H2:** with restricted refinement, the adaptation gap is ≈ 0, with a false-alarm rate no higher than unrestricted refinement.
- **H3:** tool-based execution sharply reduces tests failing on the correct service.

**Note on clarifying questions:** we considered having the agent ask a (simulated) product owner. We dropped it: answering from the benchmark's precise version turns the study into "can questions recover the precise text", which is circular and avoids the real problem of generating good tests from vague requirements alone.

---

## 7. Evaluation plan

### 7.1 Baselines (same LLM and budget)
1. RESTestBench single-step (their prompt)
2. RESTestBench refinement (their prompts, 3 rounds)
3. **Prompt-fixed refinement:** their pipeline with only the prompts corrected (copying example removed, generator warned, judge feedback passed on, API_BUG outcome). Separates "better prompting" from our structural changes.
4. A LogiAgent-style response judge
5. A crash/schema fuzzer (EvoMaster or Schemathesis), to show what crash oracles miss

### 7.2 End-to-end and safety metrics
- **Bug-catching score** (RESTestBench mutation score), vague and precise separately
- **Gap closed** = (ours − baseline) ÷ (precise upper bound − baseline), on vague requirements
- **Adaptation gap** = score refined on correct API − score refined on buggy API
- **False-alarm rate** on the correct service
- **Tampering rate** (value copying, weakening edits), out-of-sandbox calls, forbidden edits (`pytest.skip`, removed assertions)
- **Tool-call accuracy**, **cost** (tokens, time)

### 7.3 Component-level evaluation (TA feedback)

| Component | Variants | Metric (ground truth) |
|---|---|---|
| Precision layer | none · direct rewrite · property catalogue · self-consistency · fine-tuned 8B | Check recall, contradiction rate, invented-check rate (atomic checks of the precise requirement) |
| Retrieval | full spec · BM25 · dense · hybrid (RRF) · hybrid + cross-encoder reranker · + query rewriting | Recall@k, MRR, nDCG, context tokens (operations used by reference tests) |
| Generation format | raw pytest · structured plan + tool compiler | Tests passing on correct service, setup-error rate |
| Refinement | none · unrestricted · prompt-fixed · hidden values · strengthen-only + copy detector · combined | Adaptation gap, value-copy rate, false alarms |
| Judge | none · RESTestBench judge · evidence-restricted · larger judge model | Verdict accuracy (we know which service was buggy) |
| Architecture | fixed pipeline · single agent with all tools · multi-persona · + shared blackboard | Tool-call accuracy, steps, cost |
| Communication | tools in-process vs. MCP · direct messages vs. A2A · full trace / summary / verdict | Latency, token overhead, end-to-end score |
| Model | Llama 3.1 8B base · +SFT · +SFT+DPO · GPT-5 Nano (reference) | Score, adaptation gap, cost |

MCP concerns how agents reach **tools**; A2A concerns how **agents talk to each other**. Protocol choice mainly affects engineering (latency, overhead); *what* is passed between agents (full trace vs. verdict) affects accuracy.

### 7.4 Protocol
1. **Baseline reproduction:** run the harness and reproduce the published baselines.
2. **Component sweep:** one component at a time on FastAPI (63 requirements), 1 run each.
3. **Finals:** best variants on all 3 services, vague and precise, 3 runs, paired Wilcoxon tests and effect sizes; full ablation.
4. **Human validation:** any LLM-judged metric checked against ≥100 human labels (Cohen's κ).
5. **Failure analysis:** taxonomy of surviving mutants and false alarms, with examples.

---

## 8. Data

- **RESTestBench** is the core dataset and needs no extra data to answer RQ1–RQ3.
- **Optional:** frozen OpenAPI specs exported from the correct services (so mutants can't leak through the spec); gold operations per requirement from the reference tests; atomic checks from the precise requirements (LLM-assisted, human-verified sample); vagueness labels.
- **Optional extension:** 2–3 more APIs from EMB/WFD or RESTgym with ~40 requirements and ~80 mutants in RESTestBench's format.
- **Fine-tuning data** is generated automatically and filtered by whether it kills mutants: realistically ~500–1,500 verified examples per fold.

---

## 9. Compute

| Item | Plan |
|---|---|
| Model | Llama 3.1 8B Instruct. Component sweep: 4-bit on the team's laptops. Final runs: fp16 on Kaggle (comparable with RESTestBench's Llama row). Optional GPT-5 Nano reference. |
| Hardware | Two laptops (RTX 5050, RTX 4060, 8 GB VRAM each) running the 4-bit model and the Docker services together; Kaggle 2× T4, ~30 h/week per account × 4 accounts |
| Inference | ≈ 40–100 GPU-hours total (to be measured). RESTestBench's Llama 8B refinement runs used ~4–13M tokens each, mostly from pasting the full spec; retrieval shortens prompts. |
| API budget | $0 for main experiments; ≈ $10–30 for the optional GPT-5 Nano reference |
| Fine-tuning | QLoRA SFT + DPO, 3 folds: ≈ 20–30 h on Kaggle T4 (e.g. Unsloth); laptops only for debugging |
| Infrastructure | Services and harness in Docker on the laptops (Kaggle has no Docker); for final runs the agent calls the Kaggle-hosted model through a tunnel. Mutation scoring is CPU-bound. |

**Practical notes:** confirm VRAM with `nvidia-smi`; the RTX 5050 (Blackwell) may need recent CUDA/PyTorch for vLLM, with llama.cpp or Ollama (4-bit GGUF) as fallback. Kaggle quota and the ~12 h session limit should be checked on the account.

---

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| The adaptation gap is small for our models | RQ1 stands on its own; RQ2 is reported as measured, including a null effect |
| Locked checks cause false alarms when a derived check is wrong | Format checks relaxable with spec evidence; report the trade-off |
| 4-bit quantisation changes model behaviour | Reported runs use fp16; compare 4-bit vs. fp16 on a subset |
| The 8B model can't produce usable plans | Tool executor reduces what the model must get right; GPT-5 Nano fallback |
| Too little fine-tuning data | Multiple verified variants and paraphrases; few-shot fallback, reported honestly |
| Leakage through the spec or precise requirements | Spec frozen from the correct service; precise requirements only for scoring; leave-one-service-out |
| A service is hard to run | Start with FastAPI (63 requirements, 141 mutants) |
| Kaggle quota / session limits | Sweep on laptops; spread finals across 4 accounts; checkpoint runs |

---

## 11. Reading list (priority order)
1. **RESTestBench** (EASE 2026): https://arxiv.org/abs/2604.25862 · code: https://github.com/casablancahotelsoftware/RESTestBench
2. **Konstantinou et al. 2024**, LLM oracles capture actual, not expected, behaviour: https://arxiv.org/pdf/2410.21136
3. **LogiAgent** (2025): https://arxiv.org/pdf/2503.15079
4. **APITestGenie** (2026): https://arxiv.org/html/2604.02039v1
5. **SATORI** (ASE 2025) · **MASTOR** (2026): https://arxiv.org/abs/2508.16318 · https://arxiv.org/abs/2606.10465
6. **AutoRestTest** (ICSE 2025) + **SBFT 2026 REST League**: https://arxiv.org/html/2411.07098 · https://arxiv.org/html/2607.01063v1
7. **LlamaRestTest** (FSE 2025): https://arxiv.org/abs/2501.08598
8. **RESTifAI** · **APIPilot**: https://arxiv.org/html/2512.08706v1 · https://arxiv.org/html/2608.17546
9. **ARMeta** (2026): https://arxiv.org/html/2605.28321
10. **RESTOR** (2026): https://arxiv.org/html/2607.23963
11. **ISSTA 2022 "No time to REST yet"**: https://arxiv.org/abs/2204.08348
12. Infrastructure: **RESTgym** https://github.com/restgym/restgym · **EvoMaster/EMB** https://github.com/WebFuzzing/EvoMaster

**Other sources:** SBFT 2026 REST League https://doi.org/10.1145/3786155.3795704 · EvoMaster access-policy oracles https://arxiv.org/html/2604.00702v1 · MASTEST https://arxiv.org/html/2511.18038 · RESTGPT https://arxiv.org/pdf/2312.00894 · Requirements-only oracles pilot https://arxiv.org/html/2607.10277v1
