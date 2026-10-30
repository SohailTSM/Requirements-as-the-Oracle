# API Testing Agent That Reads the Requirements (Software Engineering)

**In one line:** Given a web service's API specification (OpenAPI) and its **requirements in plain English**, the agent explores the API like a tester: it creates data, chains calls, and checks the service actually **behaves as the requirements say**. It finds bugs that pure "random fuzzing" tools miss.

> 📄 **Detailed research available:** see [REST_API_Testing_Agent_DEEP_DIVE.md](REST_API_Testing_Agent_DEEP_DIVE.md). It covers existing tools and their published numbers, our analysis of RESTestBench's prompts and raw results, and the final approach (locked checks from vague requirements, tool-based execution, restricted refinement, an evidence-restricted judge, and a fine-tuned 8B model). This file is the original idea write-up and is superseded by it.

---

## The problem, in plain words
Most modern apps are backends with **REST APIs** (`POST /users`, `GET /orders/{id}`, …). Testing them automatically is hard:
- **Fuzzing tools** (EvoMaster, RESTler, Schemathesis) send many random or smart-random requests. They're great at finding **crashes** (500 errors), but they don't know what the *correct* behaviour is. If "a cancelled order can't be refunded twice" is violated and no crash happens, they miss it.
- **Requirements** like that live in documents written in English.
- Getting *deep* into an API needs **valid sequences**: create a user → log in → create a product → place an order → cancel → refund. Random tools often never get past step 2.

LLMs can read the requirements and the spec, understand dependencies between endpoints, generate realistic inputs, and **turn requirements into checks (oracles)**.

## What we would build
A testing agent for backend developers.

**Example walkthrough**
```
Input: OpenAPI spec of a shop API + requirement doc:
  R7: "An order can be refunded at most once."
  R9: "Stock must decrease when an order is placed and increase when it is cancelled."

Agent:
 1. Builds a dependency map: /orders needs a product_id and a user token, etc.
 2. For R7 plans a scenario:
      POST /users → POST /login → POST /products → POST /orders → POST /orders/{id}/refund (200 ✔)
      → POST /orders/{id}/refund again → returns 200 ❌ (should be 4xx)  → BUG FOUND (R7 violated)
 3. For R9: GET /products/{id} (stock=10) → order 2 → stock=8 ✔ → cancel → stock=8 ❌ → BUG (R9)
 4. Writes each bug as a reproducible pytest test + short report citing the requirement.
 5. Also tracks coverage (which endpoints/branches were exercised) to decide where to explore next.
```

## How it works under the hood
```
OpenAPI spec + requirement doc ──► RAG index (endpoints, schemas, requirements)
Planner   ──► picks a requirement or uncovered endpoint → plans a call sequence
Executor  ──► HTTP tool (calls the running API in a sandbox), stores created IDs (state)
Oracle    ──► turns requirement into assertions; compares responses
Coverage  ──► tool reports code coverage of the service (JaCoCo / coverage.py)
Reporter  ──► reproducible test file + bug report
```

## Where each capability shows up
| Capability | What it actually does in this project |
|---|---|
| **Tool calling** (main) | Real HTTP calls, state store for created objects, coverage reports, test-file writer |
| **Agentic** (main) | Long exploration: plans call sequences, tracks state and coverage, decides what to test next |
| **Reasoning** (main) | Converts English requirements into checkable assertions (oracles) and decides whether a response violates them |
| **RAG** | Retrieves the relevant endpoints and requirements for each scenario from large specs/docs |
| **Fine-tuning** | Small model for generating valid, realistic input values from schemas (or classifying requirement types) |

## Research question
> **RQ1:** Does a **requirements-aware agent** find more *functional* bugs (wrong behaviour, not just crashes) than state-of-the-art fuzzers on the same APIs and time budget?
>
> **RQ2:** Do the requirement-derived checks **raise false alarms** (claiming a bug that isn't one)? How often is the LLM's interpretation of a requirement wrong?

**What we expect (hypothesis):** Fuzzers win on crash-finding and raw coverage. The agent finds requirement-violation bugs that fuzzers can't. False alarms are the main risk, so a validation step (re-run + check requirement wording) keeps them manageable.

## Method (how we'd answer it)
1. **Pick ~6–8 open-source REST APIs** from the **EMB benchmark** (EvoMaster Benchmark) or **RESTestBench** (2026, with natural-language requirements).
2. **Plant functional bugs** (mutations that violate specific requirements without crashing), about 5–10 per API, so we know exactly what should be found. Also keep the APIs' real known bugs.
3. **Build tools:** HTTP executor, state store, coverage collector, test writer. Everything runs in Docker locally.
4. **Fine-tune a small model** (LoRA) for schema-valid input generation (compare with a big model).
5. **Compare (same time budget):** EvoMaster / Schemathesis (fuzzers) → LLM agent without requirements (spec only) → our requirements-aware agent.
6. **Ablations:** no requirement doc, no state tracking, no coverage feedback, no validation step.
7. **Measure:** planted-bug detection rate, real bugs found, false-alarm rate, code coverage, valid-request rate, tool-call errors, cost.

## Data we can use
- **EMB** (EvoMaster Benchmark): a public corpus of runnable REST APIs.
- **RESTestBench** (2026): REST APIs with natural-language requirements.
- Fuzzers as baselines: **EvoMaster**, **Schemathesis**, **RESTler** (all open source).

## Difficulty and effort
- **Difficulty:** Medium. Running the APIs in Docker is standard. Planting bugs needs understanding of each API.
- Moderate API token usage (long exploration). Keep a fixed step budget.

## Why pick it / why not
- ➕ Very "agentic": long-horizon, stateful, tool-heavy, with objective results (bug found or not).
- ➕ Practical and industry-relevant, and a natural fit for CS students.
- ➖ An active area (several LLM REST-testing tools in 2025–26), so our angle is **requirements as oracles + false-alarm analysis**.
- ➖ Setting up each API takes time, so start with 3–4.

## Links
- RESTestBench (2026): https://arxiv.org/html/2604.25862
- Multi-agent REST API testing: https://arxiv.org/pdf/2411.07098
- AutoRestTest at SBFT 2026: https://arxiv.org/pdf/2607.01063
- EvoMaster + EMB: https://github.com/WebFuzzing/EvoMaster
- Schemathesis: https://github.com/schemathesis/schemathesis
