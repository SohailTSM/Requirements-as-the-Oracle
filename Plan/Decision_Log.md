# Decision Log — PreTrained

Every decision about the project, with the reason behind it. Newest entries at the bottom of each section; entries are numbered and never deleted (a replaced decision is marked **superseded → D-xx**).

**Who:** *Team* = decided by the team / Sohail; *Claude* = proposed and made by Claude while building, open to change.
**Status:** *fixed* (settled) · *default* (current choice, tested by the sweep or ablations) · *open* (still to be decided, with the date it will be decided).

The proposal (`Proposal/PreTrained.pdf`) is frozen. Any change from it is recorded here instead.

---

## 1. Problem and research question

**D-01 · Problem: API test generation on RESTestBench** — Team · fixed
- Decision: build an LLM agent that writes REST API tests from natural-language requirements, evaluated on RESTestBench (3 services, 106 requirements in vague and precise form, 228 planted bugs (mutants)).
- Why: it fits the course focus (agents, tool calling, RAG); the benchmark scores tests objectively (did the test catch the planted bug while passing on the correct service?); and the paper reports a weakness we can work on.
- Rejected: the other ideas in the idea bank (kept as the published idea-bank artifact).

**D-02 · Research question: good tests from vague requirements alone, with no clarification questions** — Team · fixed
- Decision: the agent gets only the vague requirement and the API spec. It never asks questions.
- Why: when the expectation is vague, there is nobody to answer the questions. Answering them with the precise requirement would just leak the precise version into the vague setting, so the experiment would show nothing.
- Rejected: an agent that asks clarification questions.

**D-03 · The weakness we target: "bug adaptation" in refinement** — Team · fixed
- Decision: the main contribution is refinement that cannot learn the planted bug.
- Why: in RESTestBench, refinement runs the test against the service (which may be buggy), shows the LLM what the service returned, and lets it rewrite the test. On vague requirements this costs 6–20 points: the LLM copies the buggy output into the expected values. Their prompts make it worse (the judge example encourages copying, judge feedback is not used in refinement, and the judge has no "this is an API bug" answer).
- How: hide the observed values, add a strengthen-only guard, detect copied values, and use an evidence-based judge (D-30 to D-33).

**D-04 · Measure every component separately and try variants** — Team (TA feedback) · fixed
- Decision: every component gets its own metric and alternative versions: retrieval (none / BM25 / dense / hybrid / hybrid + reranker / query rewriting), architecture (single agent / multi-persona / blackboard), communication (MCP / A2A).
- Why: the TA asked for per-component metrics and variants, so the report can show which part helps.

---

## 2. Scope, people and schedule

**D-05 · The proposal is frozen** — Team · fixed
- Decision: `Proposal/PreTrained.tex/.pdf` is never edited after submission. Any change from it goes in the plan and this log.

**D-06 · No scope reduction, no go/no-go gates** — Team · fixed
- Decision: everything in the proposal scope gets built and run.
- Why: the team commits to the full scope. If compute runs short, we add machines (teammates' Kaggle accounts, the RTX 4060 laptop) rather than drop experiments.

**D-07 · Sohail and Claude build the whole project** — Team · fixed
- Why: one person driving the code keeps it consistent. Teammates' machines and accounts are a reserve only.

**D-08 · Deadlines** — Team · fixed
- Mid submission **Fri 30 Oct 2026**; final submission **Sun 15 Nov 2026**. Internal milestones: baselines start Thu 22 Oct, default configuration working by Tue 27 Oct, sweep done Wed 4 Nov, freeze Wed 11 Nov 23:59 (see `Implementation_Plan.md`).

---

## 3. Compute, infrastructure and models

**D-09 · Inference on this PC; training on Kaggle; no paid APIs, no institute clusters** — Team · fixed
- Decision: all LLM inference runs on the local RTX 5050 (8 GB) through Ollama. Training (SFT, DPO) runs on a Kaggle T4. Budget: $0.
- Why: free and under our control. Kaggle cannot host a model as a lasting endpoint (notebooks are batch jobs), so it is used only for training. Institute clusters are not used.

**D-10 · W&B for logging, Hugging Face Hub for model storage** — Team · fixed
- W&B project `pretrained-restestbench` (training runs: `pretrained-finetune`); private HF repos for adapters. Keys are in `secrets/` and never printed or committed.

**D-11 · Llama 3.1 8B Instruct only** — Team · fixed
- Decision: every LLM role (baselines, precision layer, planner, refinement, judge) uses Llama 3.1 8B Instruct, either base or with our fine-tuned adapters. No larger models, no separate judge model.
- Why: the proposal compares against RESTestBench's published Llama 3.1 8B results, and the question is what a small open model can do with the right pipeline. Other models are considered only if the team decides so later.

**D-12 · 4-bit Q4_K_M weights; baselines rerun locally** — Claude · fixed
- Decision: serve the Ollama Q4_K_M build. All baselines (B1–B3) are rerun on this same 4-bit model; the published fp16 numbers are reported only as a reference.
- Why: fp16 weights (~16 GB) do not fit in 8 GB of VRAM. Rerunning every baseline on the same model makes the comparison fair. The 4-bit vs fp16 effect is measured once by the extra run (Plan §5 D).
- Observed: local B1 precise run 1 = 35.0% vs published fp16 21.5% / 18.4%. 4-bit does not make the baseline weaker, so our gains cannot come from a weakened baseline.

**D-13 · 24,576-token context, flash attention, q8_0 KV cache** — Claude · fixed
- Why: at 32k context, ~10% of the model spilled onto the CPU (slow). At 24k it runs fully on the GPU (6.3 GB VRAM). That covers RESTestBench prompts (full spec plus test plus trace). The Ollama model is named `Meta-Llama-3.1-8B-Instruct` so RESTestBench's own `llama8b` tools run unchanged through the OpenAI-compatible endpoint.

**D-14 · Ollama prompt cache limited to 2 GB of RAM** — Claude · fixed
- Why: by default the cache grew to 8.6 GB. If RAM runs out, VS Code crashes and the working session dies. Every long job also has a RAM guard (waits if less than 4 GB is free).

**D-15 · Temperature 0.2, 3 refinement rounds** — Claude · fixed
- Why: these are RESTestBench's own defaults (`temperature 0.2`, `DEFAULT_REFINEMENT_ROUNDS = 3`), so the budgets match the baselines. The only exception is self-consistency, which samples at 0.7 because it needs different samples.

**D-16 · One LLM job at a time, through a resumable queue** — Claude · **superseded → D-51**
- Decision: `scripts/run_queue.py` runs jobs one after another. It skips finished jobs, gives up on a job after 2 failures, pauses while `results/queue.pause` exists, and checks free RAM.
- Why: one GPU. Two jobs at once would compete for VRAM and RAM and make the timings meaningless. The queue lets the experiments run unattended overnight.

---

## 4. Harness and baselines

**D-17 · The RESTestBench harness is never edited** — Claude · fixed
- Decision: the harness is a git submodule pinned at commit `4decc1e3`. Every adaptation lives outside it: the launcher loads their runner via importlib, our tools are subclasses, and fixes are applied by the launcher.
- Why: scores must be comparable with the paper, and anyone must be able to reproduce them from the pinned commit.

**D-18 · Fix TodoApp root-owned files by chown before each reset** — Claude · fixed
- Why: TodoApp containers write files owned by root into the working copy, which breaks the harness's reset between mutants. Before each reset the launcher runs a throwaway container that chowns those files back (`common/harness.py`), so the harness code stays unchanged.

**D-19 · Baselines B1–B4 and a fuzzer** — Team (proposal) / Claude (implementation) · fixed
- B1 single-step and B2 refinement: RESTestBench's tools unchanged. B2 is wrapped by `LoggedRefinementTool`, which only records each round. Those records are needed to measure adaptation and to build the DPO data.
- B3 prompt-fixed refinement: their loop with the prompt problems fixed (the judge can answer API_BUG, the judge's feedback is passed to refinement, no example that encourages copying). It answers the question "is fixing the prompts alone enough?"
- B4 response judge (LogiAgent-style), plus a Schemathesis/EvoMaster fuzzer as a non-LLM reference: pending.

**D-20 · Queue paused while our agent is validated** — Claude · ended 20 Oct ~03:50 (→ D-51)
- Why: the baselines do not depend on our code, so they can fill idle GPU time later. Our agent's default configuration lies on the critical path (default → sweep → finals), so it gets the GPU first. Remove `results/queue.pause` to resume.

---

## 5. Our agent

**D-21 · Structured plan, compiled to pytest by a deterministic compiler** — Claude · default (variant "raw pytest" in sweep)
- Decision: the LLM writes a JSON test plan (actors, steps, checks). A compiler turns it into a pytest file.
- Why: (1) an 8B model writes code that often breaks (70% of B1's tests fail even on the correct API); (2) a plan can be checked against the spec before any call to the service; (3) restricted refinement needs check-level structure, because it must know which expected values are locked and whether one was copied. Raw generation stays as a sweep variant to measure this choice.

**D-22 · Generated tests are self-contained** — Claude · fixed
- Decision: each test file carries its own small runtime (placeholders, actors, expectation types, JSON report) and imports nothing from our package.
- Why: the harness runs test files in its own environment and scores them like any other tool's output.

**D-23 · Authentication inferred from the spec** — Claude · fixed
- Decision: the compiler reads `securitySchemes` and the login/register/me operations to pick the token type, prefix, login format and response envelope. It does not hard-code each service.
- Why: the agent must work on any OpenAPI service. The inference was checked by hand on all three: FastAPI (Bearer, form login with username = email), NestJS ("Token" prefix, `user` envelope), TodoApp (Bearer, `accessToken`).

**D-24 · Retrieval: hybrid + reranker, top 5, plus dependency closure** — Claude · **refined → D-78** (reranker only for short requirements)
- Why: best on vague requirements offline (recall@5 0.888; all needed operations in the top 5 for 0.809 of requirements). Dependency closure adds the operations that create what the selected operations need (for example the POST that creates an item before a DELETE). The first smoke run failed partly because of missing create steps.
- Note: on precise requirements the reranker does worse (0.573 vs 0.798 for hybrid without it). **Open (decide Wed 4 Nov, with the sweep winners):** use the same retrieval for both variants, or hybrid for precise.

**D-25 · Retrieval evaluated without authentication operations** — Claude · fixed
- Why: the compiler adds login/register itself (D-23), so leaving them in would inflate recall without saying anything about the requirement.

**D-26 · Precision layer: property catalogue with meaning and format checks** — Claude · default (prompt revised → D-43)
- Decision: the vague requirement becomes atomic checks of two kinds. *Meaning* checks come from what the requirement says. *Format* checks are details the requirement does not state (for example the exact status code). Status codes may only come from the responses documented in the spec.
- Why: the split lets refinement keep meaning checks locked while format guesses can be corrected. Offline (vague): catalogue and direct rewrite reach the same recall (0.40). The catalogue invents fewer checks (0.05 vs 0.09) but contradicts the gold more often (0.28 vs 0.21). Self-consistency is being evaluated. The sweep makes the final choice on end-to-end bug-catching score, not on these offline numbers.

**D-27 · Gold checks drafted by Claude from the precise requirements** — Claude · open (human check before Fri 30 Oct)
- Decision: 268 atomic checks (FastAPI 148, NestJS 94, TodoApp 26) in `gold/checks/*.yaml`, used for the offline precision metrics and as SFT targets.
- Why: RESTestBench has no check-level labels. The precise requirements are human-written expectations, and splitting them into atomic checks is mostly mechanical.
- Risk: the SFT model learns Claude's way of splitting. Mitigation: human spot-check (~60 checks) and human validation (≥100 labels, Cohen's κ) before the numbers are reported.

**D-28 · Offline precision metrics judged by Llama 3.1 8B, then checked by humans** — Claude · default
- Why: matching generated checks to gold checks is semantic, and D-11 rules out other models. The human labels (D-27) measure how far that judge can be trusted.

**D-29 · Lint and one static repair round before the first run** — Claude · fixed
- Decision: the plan is checked against the spec (unknown operations, missing path parameters, setup steps that act on random ids, undefined actors, references to later steps, checks that compare a field with itself, and others) and sent back to the LLM once.
- Why: these errors can be found without calling the service, so fixing them can never teach the model a bug.

**D-30 · Failed format checks are relaxed once, without the LLM** — Claude · **superseded → D-41**
- Decision: if only format checks failed after the first run, they are dropped once and the test reruns. This never counts as an API bug.
- Why: in smoke run 1, guesses such as "status 403" failed on the correct API, the judge called them API bugs, and the test lost credit because it failed on the correct service.
- Cost: on a mutant, a format check that caught the bug is dropped too. Meaning checks are never relaxed. **To do:** add a "no format relaxation" ablation to measure this cost.

**D-31 · Strengthen-only guard locks step, expectation, key and literal value** — Claude · default
- Decision: for a meaning check, refinement may not change the step it is about, the expectation type, the list key, or a literal expected value. Changed or dropped meaning checks are put back. Field paths and `{{references}}` may be repaired. A new expected value equal to something the service returned is blocked, unless it is a status code documented in the spec on a format check.
- Why: a value copied from a buggy response is always a literal, so references can be repaired safely. The first version also locked references, which kept broken checks such as `{{A.id}}` for an undefined actor forever.

**D-32 · Refinement does not see observed values** (`hidden`, `combined`) — Claude · default (refined → D-42)
- Why: refinement cannot copy a value it never sees. Refinement still gets the status codes and setup errors, which is enough to fix calls, bodies and setup.

**D-33 · Evidence judge: TEST_BUG needs a quote, API_BUG ends the loop** — Claude · default
- Decision: the judge may answer TEST_BUG only by quoting the requirement or the spec. If the quote is missing, the answer counts as API_BUG and the test is kept. A check that could never pass on a correct service (it compares a field with itself, or reads a field the spec does not have) is a TEST_BUG.
- Why: what the service returns is never evidence that an expectation is wrong. That rule is what separates our judge from RESTestBench's.

**D-34 · Default configuration** — Claude · default (the sweep changes one setting at a time)
- `retrieval hybrid_rerank · top_k 5 · precision catalogue · generation plan · refinement combined · judge evidence · architecture pipeline · model base` (`pretrained/configs/agent/default.yaml`).

---

## 6. Experiments

**D-35 · Sweep: one component at a time, FastAPI vague, 27 runs** — Team (plan) · fixed
- Why: a full factorial would need thousands of runs. Changing one setting at a time from the default shows each component's effect at our budget. FastAPI has the most requirements and gold checks, and vague is where adaptation hurts most.

**D-36 · Smoke tests use FastAPI vague requirements 1, 4, 9, 12, 25** — Claude · fixed
- Why: they have hand-written reference plans (the executor passes all 10 reference plans and catches 26/28 bugs), and together they cover create, read, update, delete and authorization. A failure therefore points at the LLM side, not the executor.

---

## 7. Fine-tuning

**D-37 · QLoRA SFT of the precision layer, leave-one-service-out folds** — Claude · default
- Decision: Unsloth `Meta-Llama-3.1-8B-Instruct-bnb-4bit`, LoRA r = 16 on all attention and MLP projections, 3 epochs, learning rate 2e-4, effective batch 8, loss on the answer only. Each fold leaves one service out (training examples: 86 with FastAPI held out, 154 with NestJS out, 184 with TodoApp out). Each service is tested only with the adapter that never saw it. Input: vague or precise requirement plus operations. Target: the gold meaning checks.
- Why: the fine-tuned layer never sees the service it is scored on. Both vague and precise forms are used so training has enough examples.

**D-38 · Adapters converted to GGUF on top of the same Ollama base** — Claude · **superseded → D-57**
- Why: inference stays on the one local Q4 model. Only the small LoRA is added, so the fine-tuned and base runs differ only by the adapter. The adapters are also stored on HF Hub.

**D-39 · DPO from B2 refinement logs** — Team (proposal) · pending (source of the pairs: see D-54)
- Pairs: "reports the bug / keeps the check" (chosen) vs "copies the buggy response" (rejected), mined from B2's recorded rounds (D-19).

---

## 8. Changes after smoke run 2 (Tue 20 Oct, ~03:30)

Smoke run 2 (FastAPI vague, requirements 1, 4, 9, 12, 25, all mutants). On the correct service the judge kept answering API_BUG for checks that could never pass, so those tests failed there and earned no credit. Causes found by reading `agent_history.json`:
(a) the precision layer copied few-shot Example 1 almost word for word (owner = caller, "appears in the list", "status 201"), even for a delete;
(b) checks the spec rules out: reading `id` from the list wrapper `{data, count}`, reading `owner_id` from a 404, `id ≠ the id used in the request path`;
(c) a format check (status 201, not documented for that operation) failed together with a meaning check, so the "only format checks failed" relaxation never fired;
(d) a missing `{id}` path parameter was never repaired, neither by the static repair round nor by any of the 3 refinements;
(e) under hidden values the judge saw only status codes, so it could not tell an impossible check from an API bug.

**D-40 · Spec-schema lint, impossible-check drop, path-parameter autofill** — Claude · default (switch: `spec_cleanup`)
- Decision: check fields and `{{sN.field}}` references are validated against the response schema the spec documents for that operation. A step expected to fail may only be read for fields that some error response documents (`detail` in FastAPI, `errors` in NestJS/TodoApp). Problems go to the static repair round. After the repair, and after every refinement, a deterministic clean-up (1) fills missing path parameters from the latest earlier step whose documented response has a matching field (creating POST steps first), and (2) drops checks the spec shows can never pass, plus format status checks whose code the spec does not document for that operation.
- Why: all of this uses the spec only, never the service's output, so it cannot learn a bug, and it removes (b) and (d), the most common reasons tests failed on the correct service. The spec was checked against the real services first (NestJS `POST /api/articles` really returns the article unwrapped, as documented), and all 10 FastAPI reference plans pass the new lint unchanged.
- Rejected: letting the LLM repair round delete such checks. The guard restores deleted meaning checks, and an LLM deletion would not be traceable to a spec rule.
- Measured by: the sweep switch `spec_cleanup: false`.

**D-41 · Format relaxation: drop a failed format check when the spec backs the observed status** (supersedes D-30) — Claude · default (switch: `format_relaxation`)
- Decision: in the first round, failed format checks are dropped even if meaning checks failed too, with one exception: a status format check that failed with a status the spec does *not* document for that operation is kept. That status is evidence against the spec, for example a mutant returning 500.
- Why: (c). The old "only format checks failed" condition almost never held. Keeping undocumented-status failures preserves the kills where a format guess was right and the mutant breaks the documented contract.
- To measure: ablation "no format relaxation" (added to the ablation list).

**D-42 · Under hidden values, the judge and the refiner see response field names but not values; the guard accepts field repairs only onto documented fields** — Claude · default
- Decision: hidden-value traces show each response's shape (`{data: [{id, title, ...}], count}`, with no values and no list lengths). The error body of a failed *setup* step is shown in full, because it explains how the request was malformed (for example 422 "field required"). A meaning check's field or reference may be repaired only onto a field the spec documents; otherwise the old check is restored (`field_repair_rejected_not_in_spec`).
- Why: (e). Field names let the judge see that a check reads a field that does not exist. Values are what get copied, and values stay hidden. List lengths are hidden too, because a count could be copied into a `count_equals` check. The spec restriction closes a loophole: without it, a mutant that renames a field could be "repaired" into a passing test.
- Fixed at the same time: the old code showed the full response values to hidden-value refinement whenever a setup error had occurred. That was a leak, and it is removed.

**D-43 · Precision prompt v2: four varied few-shot examples and a "do not copy the examples" rule** — Claude · default
- Decision: the examples are create (borrow), delete (remove, then 404 and gone from the list), isolation, and filter, plus rules: "the examples show the format only… after a delete, the resource must be gone", and "an echo check compares the same field". `PROMPT_VERSION = "v2"` in `agent/precision.py`.
- Why: (a). Spot check on FastAPI requirements 1, 9, 12, 25: the delete case is now right (succeeds, 404 afterwards, gone from the list). Some odd checks remain (for example "access_token equals the input credentials"). This is accepted for now: the precision variants and fine-tuning are where this layer should improve, not more prompt tweaking on smoke requirements (to avoid tuning the prompt to them).
- Consequence: the offline precision numbers in STATUS.md (catalogue / direct rewrite) are **v1** results, archived in `results/offline/precision/v1/`. The self-consistency v1 evaluation was stopped at 15/106 because it would be redone anyway. Offline precision results are now stored per prompt version (`results/offline/precision/<version>/`, W&B run `offline_precision_<version>`), and all variants are rerun on v2.

**D-44 · The current Kaggle SFT run (v2) counts as a pipeline test; final adapters are retrained on the frozen prompt** — Claude · done (retrained → D-69)
- Why: the SFT examples embed the catalogue prompt, including the few-shot examples, so the adapters trained now saw prompt v1. Serving them with v2 would differ from training, and comparing them with the base model would mix two changes. The run still checks the whole Kaggle → GGUF → Ollama path and gives the first loss curves. The SFT data is rebuilt and retrained once the prompt is frozen (after smoke run 3). Kaggle quota allows it (30 GPU-hours a week).

**D-45 · Bug fix: our tool's test command now uses this interpreter and quoted paths** — Claude · fixed
- Found: the harness runs `get_test_command()` through a shell from the test's folder. Ours returned a bare, unquoted `pytest <path>`, but pytest is not on that shell's PATH and the workspace path contains spaces. So every evaluation run of our agent failed before the test ran. Both smoke scores of 0% were partly caused by this, not only by the pipeline. (Our own in-loop runs used a different code path, which is why req 4 "passed" during generation.)
- Decision: `"<venv python>" -m pytest "<path>" -v --tb=short`, the same form RESTestBench's tools use. The baselines (B1–B3) inherit their tools' command and were not affected.

**D-46 · Re-scoring without regenerating (`run_benchmark.py --phases execute evaluate`)** — Claude · fixed
- Why: generation is the expensive LLM part. If an evaluation-side problem is found, tests already generated can be scored again at no LLM cost. `run_meta.json` keeps the original generation record and adds a `rescored` entry.
- Note on scoring (checked in `evaluation_results.json`): every generated test, including those generated against a mutant, is run against the correct service and **all** mutants of its requirement. It scores only if it passes on the correct service.

**D-47 · Robustness fixes in plan parsing and placeholders** — Claude · fixed
- The plan parser counted braces inside JSON strings, so one malformed placeholder (`"{{s1.id}"`) made the whole plan unreadable and the test was lost (req 9, mutant 0). The parser now skips strings and repairs `{{x}` to `{{x}}`. All 37 earlier planner outputs still parse the same way.
- The placeholder pattern in both the runtime and the lint did not allow `[n]`. So `{{s1.data[0].id}}` was sent to the service as a literal string and never checked. The runtime's path reader already supported `[n]`, so only the patterns changed.
- New lint messages for `{{sN.req[0]…}}` and for reading the request body of a step that sent none (they point to `{{sN.data[0].id}}`).
- An actor used in a step but never defined is created as a fresh registered user (`new_user`), the only sensible reading of a plan that names a second user. This is part of the spec clean-up (D-40).

**D-48 · Two more spec-only clean-up rules: invented credentials and misdirected references** (part of D-40) — Claude · default
- An `existing` actor whose email the requirement does not quote becomes a `new_user`. Why: the plan format allows existing accounts only when the requirement gives credentials. Made-up ones fail at login (req 25: "Incorrect email or password") and the test is lost.
- `{{sN.req.f}}` where sN sent no field `f` but its documented response has one is rewritten to `{{sN.f}}`, and the reverse. Why: req 12 and req 25 lost tests to exactly this. The rewrite uses only the spec and the plan's own request bodies. If the rewrite turns a check into a self-comparison, the clean-up then drops it (for example "access_token equals the input credentials").
- The 10 FastAPI reference plans are unchanged by both rules.

**D-49 · B4 response-judge baseline = our scenario + an LLM verdict at test time** — Claude · default
- Decision: B4 uses the same retrieval and planner as our agent (precision layer off, no refinement), but drops the planned checks. Instead, the generated test makes the calls and then asks Llama 3.1 8B (temperature 0, seed 0) whether the full requests and responses satisfy the requirement, and asserts on PASS. The verdict is computed every time the test runs, so the correct service and each mutant get their own verdict. This is the LogiAgent-style oracle that "judges after seeing the response".
- Why: the scenario is held fixed and only the oracle changes, so B4 against ours isolates the main claim (checks fixed in advance vs judging after seeing the response). It also uses the same single model (D-11).
- Cost: one extra LLM call per test execution (about 1 + 2.2 executions per generated test), which is fine locally.

**D-50 · Fuzzer baseline (E) = Schemathesis, requirement-blind oracle, on the requirement's operations** — Claude · default
- Decision: one harness test per requirement. It registers a fresh user (same auth inference as our agent) and runs `st run` with a fixed seed (= repetition number), 25 examples per operation, phases `examples,fuzzing`, and checks `not_a_server_error` + `response_schema_conformance`, on the paths our retrieval selects for the requirement (top 5).
- Why: crash and schema oracles are what classic fuzzers use, so the baseline shows how many planted (logic) bugs they miss. Restricting to the requirement's operations makes the per-requirement score comparable. `status_code_conformance` is left out: FastAPI documents only 200/422, so normal 401/403/404 answers would fail every test on the correct service.
- Chosen over EvoMaster: Schemathesis is a Python package already installed, runs on the CPU, and needs no JVM or extra service setup.

**D-51 · Up to 2 benchmark jobs at once, on different services; baseline runs split per service** (supersedes D-16) — Claude · default
- Decision: the queue runs up to `max_parallel` (2) jobs at the same time, but only on services that do not overlap. Each baseline run is split into one job per service (`exp_id …__<service>`), and `analyze_results.py` merges the parts back into one run. Runs started by hand (for example development smoke runs) count against the limit and block their service. The baseline queue was resumed at ~03:50.
- Why: in smoke run 2 the GPU was busy for only about 34 s of every ~1.9 min per generated test. The rest was harness work (service resets, test executions against every mutant). Ollama serves one request at a time, so two overlapping runs fill that idle GPU time without competing for VRAM. Same-service runs cannot overlap, because each service has one working copy and one port. The harness is designed for several services on different ports. Splitting by service also means a development run on FastAPI no longer blocks the NestJS and TodoApp baselines.
- Safeguards that still apply: RAM guard (≥ 4 GB free before any start), pause file, skip-if-done, 2-failure limit.
- Effect on measurements: when jobs overlap, wall-clock time is inflated, and so are per-call LLM times, because they include waiting in Ollama's queue. So compute is reported from token counts (logged for every tool), converted to GPU time with a throughput measured once on an idle GPU.

## 9. Changes after looking at smoke run 3 while it ran (Tue 20 Oct, ~04:00)

**D-52 · Two more deterministic repairs: unclosed plans and wrapper words in references** — Claude · fixed
- The planner sometimes stops before the final `}` (req 4, mutant 2: the whole plan was there, only the closing brace was missing, so the test was lost). The parser now tracks open brackets outside strings and closes any left open.
- `{{s1.resp.id}}`, `{{s1.response.id}}`, `{{s1.body.id}}` and similar are rewritten to `{{s1.id}}` when the rest is a documented response field. In req 9 this one reference survived the lint repair and all 3 refinements, and the test never got past setup.
- All planner outputs collected so far (v1 to v3) parse.

**D-53 · Requirement-support filter for derived checks** — Claude · **decided → D-58 (not the default)**
- Decision: an optional step (`support_filter: true`) after the precision layer. One LLM call asks, for each derived check, whether the requirement states or clearly implies it, and drops the "no"s. If every meaning check would be dropped, or the answer cannot be read, nothing is dropped.
- Why: in v3 the precision layer added "the item no longer appears when listing items" (from the new delete example) to a read-access requirement (req 4). On the correct service the check failed, and the judge defended it as an API bug. Checks unsupported by the requirement make tests fail on the correct service, which costs all their credit.
- Why a filter before execution, not a per-check judge after a failure: the filter sees every check, whether it failed or not, and never sees service output, so it cannot learn a bug. A judge that drops failed checks would drop correct checks that failed because of a real bug.
- Measured offline (unsupported rate down? recall kept?) and, if adopted, by a sweep switch.

**D-54 · Where the DPO pairs come from** — Claude · **open: decide when the first B2 vague logs and our `strengthen_only` / `unrestricted` sweep runs exist (≈ Thu 22 Oct)**
- The plan says DPO pairs come from B2 logs ("report the bug, don't copy the response"). But B2 refines raw pytest code in RESTestBench's prompt format, while the DPO adapter would serve our refinement role, which writes JSON plans. Pairs in the wrong format would train the model on prompts it never sees.
- Option A (as planned): mine copy events from B2 (a refinement adds an asserted literal that appeared in the trace) and rewrite them into our plan format. Faithful to the proposal, but needs a conversion step that may add noise.
- Option B: use our own guard's records from the visible-value variants. Each `value_copy_blocked` / `locked_check_restored` event already stores the LLM's attempted plan (rejected) and the guarded plan (chosen), in our format, with the same prompt the model saw.
- Leaning: B as the main source, with A's B2 copy events used to check that B covers the same kinds of copying. Deciding with real counts, because DPO needs at least a few hundred pairs.
- Update (20 Oct ~09:20): the builder for B exists (`finetune/build_dpo_data.py`). Refinement rounds now also store the guarded plan (`refine_guarded`), so earlier runs give no pairs. **Fold problem:** guarded runs before the finals happen only on FastAPI (the sweep), so the fold that holds FastAPI out would get no B pairs. Fix candidates: (1) pairs from B2's NestJS/TodoApp logs (source A, converted); or (2) data-generation runs of our agent with `strengthen_only` on NestJS and TodoApp (~2.2 h), used **only** in the folds that hold out a different service. (2) keeps D-55, because no fold ever trains on the service it is scored on. Decide on Thu 22 Oct with counts.

**D-55 · FastAPI is the development service; NestJS and TodoApp are held out** — Claude · fixed
- Decision: every pipeline fix is driven only by FastAPI runs (smoke runs, the sweep). We do not read NestJS or TodoApp test histories to change the agent before the freeze (Wed 11 Nov). Baseline runs on those services are collected but used only for scores. The report gives per-service results and marks FastAPI as the development service.
- Why: the fixes in sections 8–9 came from reading individual FastAPI failures, so FastAPI scores may be optimistic. NestJS (29 requirements) and TodoApp (14) then show whether the fixes generalise. This matches the leave-one-service-out folds of the fine-tuning (D-37).
- Also: smoke runs reuse requirements 1, 4, 9, 12, 25, so other FastAPI requirements keep some held-out value within the sweep.

## 10. Results of smoke run 3 and the morning checks (Tue 20 Oct, ~08:45–09:00)

Smoke run 3 (FastAPI vague, 5 requirements): tests generated on the correct service catch 14.3% (same as v2); tests generated on mutants catch **14.3% (v2: 4.8%)**. Tests that pass on the correct service went from 1 of 19 to 7 of 19. Offline precision, prompt v2 (vague, 106 requirements):

| variant | recall | soft recall | contradicts gold | unsupported | checks per requirement |
|---|---|---|---|---|---|
| catalogue | 0.392 | 0.474 | 0.273 | 0.055 | 3.6 |
| direct rewrite | 0.396 | 0.468 | 0.206 | 0.091 | 5.7 |
| self-consistency | 0.373 | 0.468 | 0.258 | 0.070 | 3.1 |
| catalogue + support filter | 0.325 | 0.420 | 0.269 | 0.041 | 2.5 |

Prompt v2 keeps v1's recall (0.392 vs 0.396) with fewer checks (3.6 vs 4.8).

**D-56 · NestJS was never evaluated in any run so far; fixed with a longer health check; false "ok" runs re-opened** — Claude · fixed
- Found: in every run, including the first B1 precise run, the harness skipped NestJS with "Service unreachable" and still exited normally. Our launcher recorded `status: ok`, so the queue treated 9 NestJS jobs as done. **The B1 "35.0%" reported last night covers FastAPI and TodoApp only.**
- Cause: right after starting a service, the harness copies `original/` over `working/` and then checks health once. NestJS runs in watch mode, so the copy makes it recompile, and the single check fails. (Starting it by hand works.)
- Fixes: (1) our launcher wraps the harness's health check so it polls for up to 3 minutes, a monkeypatch from outside the harness like D-18; (2) a run with no evaluation file for a requested service now ends with an error status and exit code 1, so the queue retries it; (3) the 9 false "ok" runs were re-opened. For B1 precise rep 1 the FastAPI and TodoApp parts stay valid, and NestJS runs as a separate part. Verified: a 2-requirement NestJS smoke run (B1) now generates and executes its tests.

**D-57 · Fine-tuned models are served as merged Q4_K_M models, not as adapters** — Claude · default
- Found: the local Ollama (0.40.3) refuses LoRA adapters ("LoRA adapters are no longer supported"), so D-38 (base plus adapter) cannot be used. The SFT run itself worked: 20, 32 and 41 min per fold on a T4, final losses 0.35 / 0.21 / 0.22.
- Decision: a second Kaggle job (`finetune/kaggle_export/`, `sohailmemon/pretrained-sft-export-gguf`) loads each fold's adapter on the 16-bit base, merges it, and exports a **Q4_K_M** GGUF, the same quantization as our base model. Each fold becomes its own Ollama model.
- Why on Kaggle: merging needs the 16-bit weights (~16 GB of RAM), which would risk crashing VS Code on this PC. Why Q4_K_M: the fine-tuned and base models then differ only by the fine-tuning, not by quantization. Disk: 3 × ~4.9 GB, with 127 GB free.
- Rejected: building llama.cpp with CUDA to serve adapters separately (a second inference server, harder to compare), and downgrading Ollama (it would change the base runs already made).
- D-38 is superseded by this entry.
- Export run 1 failed: loading the merged model in 16-bit on the GPU needs ~15 GB of weights against the T4's ~6.9 GB. Run 2 loads the adapter in 4-bit and lets Unsloth's GGUF export do the 16-bit merge on the CPU (Kaggle has ~30 GB of RAM). It runs on the v1-prompt adapters to validate the route before the final SFT models (D-69) are exported.

**D-58 · Support filter is not part of the default** (closes D-53) — Claude · fixed
- Why: offline it removes about one check per requirement (3.6 → 2.5), but recall drops 0.39 → 0.32 and the contradiction rate barely moves (0.273 → 0.269). The checks it drops are mostly correct ones, and the contradicting checks it was meant to remove stay. It remains available as a switch (`support_filter: true`).

**D-59 · Two more spec/requirement-only clean-up rules** (part of D-40) — Claude · default
- *Exact status codes in meaning checks become status classes unless the requirement states the code.* This enforces the catalogue's own rule ("the exact code is format unless the requirement states it"). In req 25 a meaning check demanded 404, while the correct service answers 403 to a forbidden read. On precise requirements, which state their codes, nothing changes: all 10 reference plans pass through unchanged when checked against their precise texts.
- *A signup step is removed when the requirement is not about signing up or logging in*, and references to that step's user are pointed at the step's actor (`{{s1.id}}` → `{{A.id}}`). Actors are registered automatically (plan format), so such a step only creates a third, unrelated user. In req 25 the plan then read "A's own user" using that third user's id and got 403 on the correct service.

**D-60 · The queue leaves services listed in `results/queue.reserve` free; jobs stopped by hand are not failures** — Claude · fixed
- Why: the parallel queue (D-51) took FastAPI, the development service, for a 5-hour B2 run. FastAPI is now reserved while development smoke runs need it, and released for overnight runs. A job ended by a signal no longer counts toward the 2-failure limit, because stopping a job is an operator decision, not a job failure.

**D-61 · Compute: this PC alone, queue running day and night** — Claude · default (review Tue 27 Oct)
- Measured: our agent ≈ 1 min per generated test (smoke run 3: 19 tests in 18 min, while sharing the GPU); B2 ≈ 2 min per test (TodoApp: 36 tests in 73 min); B1 ≈ 30–43 min per FastAPI run. Interactive runs generate 204 / 94 / 36 tests (FastAPI / NestJS / TodoApp).
- Remaining ≈ 380 run-hours. With two services overlapping (D-51) it fits before the freeze. Schedule: baselines finish ~27 Oct, the sweep (FastAPI-only, one run at a time, ≈ 95 h) by 4 Nov, finals and ablations (≈ 145 h, overlapping by service) by ~10 Nov.
- Why no second machine yet: every run on the same GPU, model build and Ollama version keeps timings and quantization identical, and the estimate fits. The RTX 4060 laptop is the buffer if the sweep slips past Tue 3 Nov.

**D-62 · Architecture variant "react": one tool-calling agent; the safety rules live inside its tools** — Claude · default (sweep: `sweep__architecture__react.yaml`)
- Decision: Llama 3.1 8B with five tools (`search_operations`, `derive_checks`, `check_plan`, `run_plan`, `submit_plan`) through Ollama's OpenAI-compatible tool calling, at most 14 turns and 3 runs. `run_plan` returns the same hidden-value report as the pipeline. After the first run, every new plan goes through the same guard (reverted expectation changes are reported back to the agent), plus the spec clean-up and the one-time format relaxation. If the agent never submits, its last run plan is used; if it never wrote a plan, the pipeline's planner writes one (marked `react_fallback`).
- Why: the sweep question is whether a single agent that chooses its own order of steps beats a fixed pipeline. Enforcing the rules inside the tools keeps that comparison about control flow, so this variant cannot win by bypassing the guard.
- Difference to note in the report: there is no separate judge. The agent itself decides when a failure is a bug and submits; that is the point of a single-agent design.
- Checked: Llama 3.1 8B issues proper tool calls through Ollama (first call `search_operations("create item")`).

**D-63 · Architecture variants "multi_persona" and "blackboard"** — Claude · default (sweep configs `sweep__architecture__{multi_persona,blackboard}.yaml`)
- Personas (same Llama 3.1 8B, each with its own system prompt): Analyst (precision layer), Designer (planner and revisions), Critic (reviews the plan against the requirement and spec before any run), Reviewer (the judge), Debugger (refinement). The Executor is code.
- multi_persona: fixed order Analyst → Designer → Critic → Designer, then ≤ 3 × (Executor → Reviewer → Debugger). blackboard: the same personas on a shared board; a Controller persona picks who acts next, from the options the rules allow at that moment (`_options()`: no run before a plan exists, the debugger only after a TEST_BUG review, at most 3 runs, …).
- Why: these are the TA's suggested variants (single agent / multiple personas / blackboard). The Critic is the one new capability: a pre-run review that sees no service output, so it cannot teach the model a bug. The guard, hidden values, spec clean-up and run limit are enforced in code in both variants, so the comparison is about organisation, not about safety rules.

**D-64 · Communication-content variants: what the judge passes to refinement** — Claude · default
- Values: `summary` (default: the judge's one-line reason, i.e. the behaviour so far), `full` (the judge's whole answer plus the hidden-value trace it saw), `verdict` (nothing but the decision). This applies in the pipeline and the persona variants. Under hidden values, `full` still contains no response values.
- The plan lists "content: full trace / summary" as the two sweep runs. Since `summary` is already the default, the two runs are `full` and `verdict`, so the sweep covers all three settings.

**D-65 · Fuzzer baseline: keep the requirement-blind oracle, and also report false alarms and raw failures** — Claude · default
- Smoke test (NestJS, 2 requirements): Schemathesis finds real 500 errors on the **correct** service (malformed `user` bodies, `DELETE /profiles/{username}/follow`). So every fuzzer test fails on the correct service and gets no RESTestBench credit.
- Decision: the fuzzer stays as designed (D-50); a crash/schema oracle is exactly what we want to compare against. The report gives its RESTestBench score plus two numbers the score hides: the **false-alarm rate** (tests failing on the correct service) and the **raw failure rate on mutants**. A crash oracle "detects" everything here, and it cannot tell a planted logic bug from a robustness bug the service already had.
- Rejected: using the correct service as a reference to subtract known failures. No other tool gets to know which service is the correct one.

**D-66 · Communication variant "mcp": the pipeline's non-LLM tools served by an MCP server** — Claude · default (sweep `sweep__communication__mcp.yaml`)
- Decision: retrieval (`search_operations`), plan lint (`check_plan`) and test execution (`run_test`) run in a separate MCP server (`tools_ext/mcp_tools_server.py`, MCP Python SDK 2.3, stdio). The pipeline is the MCP client and keeps one server per run. Every tool call's wall time is logged (`mcp_timings` in each history).
- Why: the TA asked for MCP/A2A variants. MCP is the standard way to give an agent tools, and the variant measures what it costs (latency, failure modes) with everything else identical. Scores should match the in-process default; a difference would point to a transport problem. Checked: search and lint round-trips return the same results as in-process calls (the first call takes ~24 s while the server loads the retrieval models).
- Note: SDK 2.x renamed `FastMCP` to `MCPServer`, so the code follows the 2.x API.

**D-67 · Communication variant "a2a": the Reviewer and the Debugger as separate A2A agents** — Claude · default (sweep `sweep__communication__a2a.yaml`)
- Decision: two local A2A agents (a2a-sdk 1.2.3, JSON-RPC over HTTP, ports 9101/9102, `tools_ext/a2a_agents.py`), each with an agent card and one skill. The pipeline sends one JSON payload per request as a text message and gets one JSON answer back. The agents call the same `judge()` / `refine()` functions (same prompts, guard and spec restriction) and return their LLM call records, so token counts stay complete. The agents are started per run and stopped on exit.
- Why: the roles that "talk" in a multi-agent design are the judge and the refiner. Putting exactly those behind A2A measures the cost of agent-to-agent messaging without changing behaviour. Checked: one Reviewer round trip with a real smoke-run failure works (76 s while the GPU was shared). The client's default HTTP timeout was too short for an LLM call, so it is raised to 900 s.

**D-68 · What the sweep row "Model: SFT / SFT + DPO" means** — Claude · default
- The plan lists `precision: fine-tuned` (SFT model for the precision layer only) in the precision row, and "Model: SFT · SFT + DPO" as a separate row. To avoid running the same configuration twice: **`model: sft`** = the SFT model serves *every* LLM role (does tuning one role help or hurt the others?); **`model: sft_dpo`** = SFT model for the precision layer, DPO model for the judge and refinement (the roles where copying happens), base model for the planner.
- Fine-tuned models are per held-out service (`<name>-<service>`), as in D-37.
- With these two, the sweep has its full 27 configurations (`configs/agent/default.yaml` + 26 `sweep__*.yaml`). The architecture, communication and model variants have not yet run end to end; they will be smoke-tested on FastAPI before the sweep.

**D-69 · Precision prompt frozen at v2; SFT retrained on it now (Kaggle training v3)** — Claude · fixed
- Decision: no more changes to the catalogue prompt or its few-shot examples. The SFT data was rebuilt on v2 (examples tagged `prompt_version: v2`; fold sizes unchanged: 86 / 154 / 184), the private Kaggle dataset was versioned, and training kernel v3 was pushed at ~09:10. Training finished at 10:58: final losses 0.352 / 0.193 / 0.216 (FastAPI / NestJS / TodoApp held out), 23 / 37 / 47 min on a T4. Loss curves are in W&B `pretrained-finetune` (group `sft_v3`) and `pretrained/finetune/logs/sft_v3/`. These are the final SFT models, to be registered as `pt-precision-sft-<service>` after the merge/export step (D-57). The v1-prompt models become `pt-precision-sft-v1prompt-<service>` and are not used in experiments.
- Why now and not after smoke run 4: smoke run 4 tests pipeline fixes (clean-up rules), not the prompt. D-43 already ruled out tuning the prompt on smoke requirements. Training now has the fine-tuned models ready days before the sweep needs them (the `precision: finetuned` and `model` rows). Kaggle quota is fine (~2 h of 30 h used this week).

**D-70 · Getting the fine-tuned models onto this PC needs a ~15 GB download** — **decided 20 Oct ~10:05: option (a)**, Sohail switched to the fast connection ("do whatever is needed")
- Ollama 0.40.3 rejects LoRA adapters in every form (GGUF and safetensors folder: "LoRA adapters are no longer supported"), and there is no CUDA compiler here to build a separate llama.cpp server. So each fine-tuned fold has to arrive as a merged Q4_K_M model (~4.9 GB; 3 folds ≈ 15 GB, and later the same again for DPO).
- Options: (a) download the merged models from Kaggle when the fast connection is available (preferred: same engine and quantization as the base); (b) install the CUDA toolkit (~3 GB) and serve base + adapters with llama.cpp (smaller per model, but a second inference engine that would also have to serve the base runs it is compared with).
- Until decided: nothing large is downloaded. The v1-prompt test models are not downloaded at all; only the export job's summary is fetched (`--file-pattern`) to confirm the export works.

**D-71 · Human validation through a labelling page with a private-per-person label store** — Claude · default
- Decision: a private artifact, "PreTrained Check Labelling" (https://claude.ai/artifact/51FenN3D8LqZBbgCgHvgEq), with a stratified random sample (seed 20261020) of **45 requirements = 181 labels**, stored in `gold/human_validation/items.json`.
  - *Gold checks* (25 precise requirements: 12 FastAPI / 8 NestJS / 5 TodoApp; 63 labels): correct / wrong / unsure, plus "missing checks".
  - *Coverage* (20 vague requirements: 10 / 6 / 4; 118 labels): the same full / partial / none and match / extra / contradiction definitions the Llama judge uses, on the catalogue-v2 output.
- Access rules: everyone with access reads the items. Each labeller writes and reads only their own `labels/<id>` document, and only the owner (Sohail) reads them all. Labels stay independent, so human–human κ is meaningful. The Llama judge's labels are **not** on the page (kept in `llm_labels.json`), so they cannot bias anyone.
- Analysis: `scripts/human_agreement.py` gives Cohen's κ human vs Llama (coverage), human vs human, and the share of wrong gold checks. Checked with a synthetic labeller that copies the LLM (κ = 1.0).
- Why a page: it removes the "open a JSON file and edit it" step for teammates, and the labels come back to Claude without copying. The plan needs ≥ 100 labels; 181 gives room for "unsure".
- Teammates can only save labels if Sohail shares the page with them as Contributor (or Editor when invited by email); the page shows a read-only notice otherwise.

**D-72 · Temporarily hold new baseline jobs so B2 FastAPI finishes sooner** — Claude · temporary (~09:15 → until B2 FastAPI vague rep 1 ends)
- All three services are listed in `results/queue.reserve` (jobs already running continue). Why: the GPU is the bottleneck (98% busy), and B2 FastAPI shares it with whatever else runs. Our agent's next smoke run, and the first end-to-end tests of the architecture and communication variants, need FastAPI and lie on the critical path (default configuration by Tue 27 Oct). Total work is unchanged; only the order changes. Afterwards FastAPI stays reserved for development, and NestJS and TodoApp are released to the queue.
- Also added: `scripts/make_tables.py` builds the report tables (main / sweep / retrieval / precision, Markdown + LaTeX) straight from the result files, so no number in the reports is copied by hand.

**D-73 · DPO training job: base model, refinement pairs, train and export in one Kaggle run** — Claude · default (push when the pairs exist)
- `finetune/kaggle_dpo/train_dpo.py` (`sohailmemon/pretrained-refine-dpo`, private dataset `sohailmemon/pretrained-dpo-data`). Per fold: QLoRA r = 16 on the same base, Unsloth's DPO patch with TRL, β = 0.1, learning rate 5e-6 (Unsloth's DPO default), 2 epochs, effective batch 8, prompts up to 5k tokens. Then merge and export Q4_K_M (D-57) in the same run. A fold with fewer than 40 pairs is not trained and is reported as such.
- Starts from the base model, not the SFT model: the DPO model serves the judge and refinement roles (D-68), which the SFT data (vague requirement → checks) does not teach. Chosen/rejected pairs come from `finetune/build_dpo_data.py` (D-54).

**D-74 · B4 runs use a 600 s test timeout; fuzzer runs queued now, B4 runs from Wed 28 Oct** — Claude · default
- B4 smoke (TodoApp, 2 requirements): it runs end to end, 1 of 4 bugs caught, verdicts computed at test time. One execution hit exactly the harness's default 60 s timeout. A timed-out test counts as failing, so on a mutant it would score as a caught bug. B4 jobs therefore pass `timeout: 600` (the queue now forwards a per-job `timeout`). The other tools keep RESTestBench's default (60 s); their tests make no LLM calls.
- The 9 fuzzer jobs (3 services × 3 seeds) are queued now; the fuzzer does not depend on our agent. B4 shares our planner and spec clean-up, so its finals start once those are frozen (as in the plan).

**D-75 · Finding: the evidence judge always answers API_BUG; new judge metrics; default unchanged until the sweep** — Claude · open (decide with the sweep's judge rows, Wed 4 Nov)
- New component metrics in `analyze_results.py`: the judge's **false API_BUG rate on the correct service** (every failure there is the test's fault, so API_BUG is always wrong; `req_N` folders are the correct service, `req_N_mut_k` the mutants) and its API_BUG rate on mutants.
- Smoke runs 2 and 3: API_BUG in **17 of 17** judged rounds (100% false alarms on the correct service). Refinement ran only after setup errors. Likely cause: TEST_BUG needs a quote, and a missing quote counts as API_BUG (D-33), so the 8B model takes the safe answer.
- Why the default stays for now: with hidden values and locked meaning checks, a wrong TEST_BUG costs little (the guard reverts expectation changes), but a wrong API_BUG costs a repair chance. So the question is whether the judge is needed at all, which the sweep's `judge: none` and `judge: restestbench` runs answer directly on all of FastAPI. 17 rounds on 5 requirements are too few to change the default.
- Implication for the report: where the guard locks expectations, the remaining failures on the correct service come from the precision layer's unsupported or contradicting checks, which refinement may not remove without risking adaptation. That makes the precision layer (and its fine-tuning) the component to watch.

**D-76 · Mid report drafted as a short LaTeX progress report with generated tables** — Claude · draft (Sohail, 20 Oct: no need to worry about the mid requirements; the draft is kept and updated with results)
- `Reports/Mid/PreTrained_Mid.tex` (4 pages, same style as the proposal, which stays untouched): status, system figure, safety rules, setup and metrics, results (tables `\input` from `pretrained/results/tables/`, so the numbers refresh on each build), findings, plan to the final. Smoke v4 and the B2/B3 text are marked TODO.
- Why now: the system description and methodology are stable, and writing them early leaves the last days for results. Assumption to confirm: the course wants a written progress report; the exact requirement is asked in STATUS.md.

**D-77 · Finding: B2 copies response values; B3's judge (like ours) always answers API_BUG** — Claude · finding (feeds D-75 and the report)
- Value copying in B2 (RESTestBench's own refinement): a refinement round adds an asserted literal that appeared in the previous run's HTTP trace in **20% of rounds (TodoApp vague, 14/69), 22% (TodoApp precise, 16/72) and 12% (FastAPI vague, partial run, 18/148)**. This is the behaviour our guard blocks.
- B3 (their loop with an API_BUG option): the judge answered API_BUG in **36 of 36** TodoApp rounds, so B3 hardly ever refined. Together with D-75 (our evidence judge: 17/17), Llama 3.1 8B appears to take an "it's the API's fault" option whenever it is offered. The judge sweep rows and B3 vs B2 measure the consequence.
- Caveat: the copy detector is a heuristic (a copied literal can be legitimate, e.g. a documented constant). A sample of flagged rounds will be labelled by hand before the rate is reported (to be added to the labelling page once more runs exist).

**D-78 · Default retrieval = "adaptive": rerank short requirements only (≤ 30 words)** (closes the open question in D-24) — Claude · default
- Decision: `adaptive` uses hybrid + cross-encoder reranking for requirements of at most 30 words and plain hybrid for longer ones. It is the default and the base of every sweep config (22 configs switched from `hybrid_rerank`; the five retrieval rows stay as they were).
- Why: the cross-encoder (MiniLM, trained on short search queries) truncates long inputs. On FastAPI, the development service, reranking was equal to hybrid on vague requirements (≤ 25 words: C@5 0.81 vs 0.81) but far worse on precise ones (25–50 words: 0.47 vs 0.93; > 50 words: 0.54 vs 0.81). The threshold was chosen on FastAPI only (D-55).
- Checked on all 106 requirements afterwards (`scripts/retrieval_non_auth.py`, which reproduces the earlier non-auth numbers exactly): adaptive matches the best variant on both wordings, vague C@5 0.809 (= hybrid_rerank; hybrid 0.730) and precise C@5 0.798 (= hybrid; hybrid_rerank 0.573).
- Effect on the sweep: none for vague (the requirements are short, so adaptive ≡ hybrid_rerank). Our precise finals no longer pay the reranker penalty, and one configuration serves both wordings. The SFT data already used exactly this pairing (hybrid_rerank for vague, hybrid for precise).

**D-79 · GGUF export: convert in /tmp; final SFT models fetched as soon as they exist** — Claude · fixed
- Export run 2 got through the first fold, then failed on the second: Unsloth writes its ~17 GB f16 intermediate to the *working directory*, and `/kaggle/working` (20 GB) already held the first fold's 4.9 GB model. Fix: `os.chdir` to `/tmp` for the conversion (export job and DPO job). Only the final Q4_K_M files stay in the output (3 × 4.9 GB fits).
- Not repeated on the v1-prompt adapters: fold 1 already showed that loading in 4 bit plus a CPU merge works, so the next export run uses the final SFT v3 adapters directly. A background chain waits for SFT v3, pushes the export, downloads the three merged models (~15 GB), registers them in Ollama as `pt-precision-sft-<service>` (base model's chat template and stop tokens, 24k context), and backs the adapters up to the private HF repo `sohailtsm/pretrained-precision-sft`. The first chain died on a single Kaggle status call that timed out (~10:10, around the network switch). It is now a script (`finetune/fetch_models.py`) that retries polling errors, and the DPO models will use it too.

- Done 20 Oct 12:31: the three merged models (4.58 GB each, streamed at ~6 MB/s) are registered in Ollama as `pt-precision-sft-{fastapi,nestjs-realworld,todoapp}`, and the adapters are backed up to the private HF repo `sohailtsm/pretrained-precision-sft`. Not yet test-run (GPU paused, D-81): first a quick offline check that each model answers in the checks format, then the offline precision evaluation of the `finetuned` variant.

**D-80 · Large Kaggle outputs are streamed to disk, never through the Kaggle CLI** — Claude · fixed
- Found (11:50): the export worked (all three folds merged and converted, ~14 min each). But `kaggle kernels output` keeps each whole file in memory before writing it (`out.write(response.content)`): the process grew ~8 MB/s while the file on disk stayed at 0 bytes. A 4.9 GB model would have pushed RAM toward the limit with Ollama, VS Code and a benchmark running (the VS Code crash risk). Stopped after ~1 min (~0.7 GB).
- Fix: `finetune/kaggle_download.py` lists the files through the same Kaggle API and streams each one in 8 MB chunks to a `.part` file, resuming with a Range request after a network error (5 attempts). `install_adapters.py` uses it for the merged models. Checked: the downloader stays at ~65 MB RAM while writing.

**D-81 · GPU paused after the current job, at Sohail's request (cool-down)** — Team · in force until Sohail says otherwise
- From ~12:00, 20 Oct: `results/queue.pause` is set (no new jobs), the chain that would have started smoke run 4 is cancelled, and the model is unloaded from the GPU once B2 FastAPI vague rep 1 finishes (~13:00). Work that needs no GPU continues (the model download, code, documents).
- Schedule effect: every paused hour moves the end of the compute plan (D-61) by about an hour; reviewed when the GPU is released.

## 11. Remote LLM servers on Kaggle (Fri 23 Oct)

**D-82 · Inference also on Kaggle T4s: a shareable server notebook with links in the log** — Team (Sohail) / Claude (design) · default
- Why: Sohail asked to keep the laptop GPU cool (D-81). Kaggle gives 2 × T4 per session and 2 sessions at once, i.e. up to 4 model servers per account, and teammates' accounts can add more. The services under test stay in Docker on the laptop (Kaggle has no Docker); only LLM calls cross the internet.
- Notebook `pretrained/kaggle_llm_server/pretrained_llm_server.ipynb` (source `serve_llm.py`), run with *Save Version → Save & Run All*: one Ollama server per T4, a Cloudflare quick tunnel per server, and a clearly marked log block **"SHARE THESE LINKS WITH SOHAIL"**. A NEW LINK block is printed if a tunnel restarts. The session stops itself at 11.8 h. Teammates run it on their own accounts and send the two lines.
- Same model and settings as the local server: Ollama 0.40.3, `llama3.1:8b` with the digest checked against the local copy (`sha256:667b0c1932bc…`, logged as SAME/DIFFERENT), alias `Meta-Llama-3.1-8B-Instruct`, 24576 context, flash attention, q8_0 KV cache, one request at a time per GPU, model kept loaded.
- Access: each link is `https://<random>.trycloudflare.com/<secret>/v1`. A small proxy refuses requests without the secret (Ollama has no authentication). The secret is hidden in everything recorded (`public_llm_url()`: run_meta, W&B).
- Cloudflare quick tunnels end a request after 100 s without a response. Answers that take longer than 20 s are sent as an immediate 200 response, then a space every 15 s, then the JSON body (leading spaces are valid JSON). Tested locally with the OpenAI client: a 25 s answer arrives and parses; a wrong secret is rejected (403).
- Laptop side: `PRETRAINED_LLM_URL` selects the server (default: local Ollama). Queue jobs take an `llm_url` key, which sets it and RESTestBench's endpoint variable for that job only. Tools: `read_links.py` (reads the links from my own running sessions' logs), `check_endpoint.py` (answer and speed with a real planner prompt).
- Fairness rule: a run uses one server from start to end; the server (with the secret hidden) is recorded in its run_meta. Runs that are compared with each other (repeats of a baseline, sweep variants) should share a GPU type; any mix is reported.
- Known limits: a session ends at 12 h (runs must fit in the time left); quota is 30 GPU-hours per account per week; B4 tests call the LLM at test time, so they can only be re-scored while their server is alive; running a server in a Kaggle notebook is a grey area in Kaggle's terms (risk: a session gets stopped).
- Started 23 Oct ~17:49 UTC: `sohailmemon/pretrained-llm-server-a` and `-b` on Sohail's account (two sessions, 4 × T4).

**D-83 · First Kaggle server session: checks, and how the four servers are used** — Claude · in force until ~05:38 UTC 24 Oct
- Both sessions were ready about 3.5 min after the push: Ollama installed, model digest SAME as the project copy, model loaded on both T4s, and a public check through each tunnel answered "Ready.". Live logs of a running session need `kaggle kernels logs -f` (the output API is empty until the run ends), so `read_links.py` uses that and stores the full links only in `secrets/llm_endpoints.json` (mode 600).
- Speed through the tunnel, real planner prompt (1,378 tokens in, 328 out): 12.7–14.8 s per call (~22–26 tokens/s end to end) on all four servers, all giving the same answer.
- Use: server A-1 → my FastAPI development runs (smoke v4 with the default, then ReAct, multi-persona, blackboard, MCP, A2A on the same 5 requirements); A-2 → NestJS baseline jobs; B-1 → TodoApp baseline jobs; B-2 spare. Mapping in `secrets/queue_endpoints.json`. The queue starts a job only if its service has a server with ≥ 3 h left, so nothing falls back to the laptop GPU (still off, D-81). `max_parallel` is 3.
- Recorded per run: `model_server` (secret hidden). These NestJS/TodoApp baseline runs use T4s while the earlier FastAPI baseline runs used the RTX 5050; the report states the backend per run (D-82 fairness rule).

**D-84 · A second FastAPI instance, so all four Kaggle servers run benchmark jobs** — Team (Sohail: "utilize the running quota maximally") / Claude (design) · default
- Why: a run needs its own service instance (one working copy and fixed host ports per service), so with three services at most three runs could overlap, and the fourth server sat idle. The sweep is FastAPI-only (27 runs, ~95 h), so a second FastAPI instance also halves the sweep's critical path.
- `scripts/make_service_copy.py fastapi b --offset 100` copies `harness/` to `service_copies/harness-fastapi-b/` (74 MB; git-ignored) and changes only three things: the Docker project name (`fastapi` → `fastapi-b`), every host port in the service's compose files (+100: 8000→8100, 5432→5532, 80→180, …; container ports unchanged, in both `original/` and `working/`), and the copy's own port in `src/config.py`. `diff` against the original shows exactly those five files. The original harness is untouched (D-17 holds); a run on the copy uses the same code, data, requirements and mutants.
- Checked: both FastAPI stacks run side by side (8000 and 8100 both answer `/docs`), and the copy stops cleanly.
- Use: `PRETRAINED_HARNESS_ROOT` selects the instance for one run, and `run_meta.json` records `harness_instance`. The queue treats instances as slots (`fastapi`, `fastapi@b`); an external process's instance is read from its environment. A job starts only on a free slot that has a remote server and only if its estimated length (×1.2 + 15 min; table from measured runs) fits the time the session has left, so no run is cut off at the 12-hour limit. `max_parallel` is 4.
- Since ~18:20 UTC: A-1 our FastAPI development runs (original instance), A-2 NestJS baselines, B-1 TodoApp baselines, B-2 FastAPI baselines on `fastapi@b` (first: B3 vague run 1). B-2 also finishes the offline precise-wording precision evaluation.

**D-85 · FastAPI requirement 15 always runs on the original instance** — Claude · fixed
- Found while checking the FastAPI copy (D-84): the generated tests call their own instance (port 8100; the harness passes each tool the instance's address), but requirement 15 says "verify via mailcatcher at **localhost:1080**". On the copy, that requirement's emails go to the copy's inbox (port 1180), so a test written to the requirement would read the original's inbox. It is the only port-bound requirement or reference test in the benchmark (checked with grep).
- Decision: every full FastAPI job is split by the queue into a main part (the other 62 requirements, any instance) and an `__r15` part (requirement 15, original instance only); analysis merges the parts (sums per service). For any run made on a copy, requirement 15 is dropped by `summarize()` (via `harness_instance`), and its `__r15` part supplies it. This affects the B3 FastAPI vague run 1 that started on the copy before the split existed. Runs made earlier on the original are complete and stay as they are.
- Why not share one inbox: the original's mail catcher is stopped whenever no run uses that instance, and two instances writing to one inbox could mix up emails.

**D-86 · Environment check with RESTestBench's reference tests ("golden" runs)** — Claude · default (first in the queue)
- Question from Sohail: "Did we verify the baseline first?" Partly. The LLM baseline matches the paper: B1 vague 2.2% vs 2.0% published, precise 25.2% vs 13.3% published (≈ 20% without their broken run 1). But the environment had not been checked with a test oracle we trust.
- Decision: run the harness's own `golden_tests` tool (the 106 hand-written reference tests, no LLM) on every service and on the FastAPI copy. They should pass on the correct services and catch nearly all planted bugs; any gap points to the environment (services, mutants, resets, health patch D-56, chown patch D-18, the copy). Our `golden` wrapper changes only the reference tests' hard-coded `BASE_URL` to the instance's address (needed for the copy on 8100; a no-op on originals). Golden FastAPI runs are pinned: all 63 requirements on the original, the 62 without requirement 15 on the copy (D-85). Results go to group `verify`.
- Results: **FastAPI original (23 Oct 21:25 UTC): 63/63 reference tests pass on the correct service, 141/141 planted bugs caught (100%).** The FastAPI environment (services, mutants, resets, patches) is sound. **NestJS: 29/29 pass, 65/65 caught; TodoApp: 14/14 pass, 22/22 caught (both 100%).** So every original service is verified, including NestJS with the health-check patch (D-56). **FastAPI copy `fastapi@b`: 62/62 pass, 138/138 caught (100%; requirement 15 excluded by design, D-85).** All four instances are verified. **NestJS copy `nestjs-realworld@b` (24 Oct): 29/29 pass, 65/65 caught (100%).**

**D-87 · Run-to-run noise is large: the sweep repeats the default 3 times** — Claude · default
- Evidence: smoke run 4 (5 FastAPI requirements, 14 bugs) of three configurations with *identical logic*, which differ only in transport: default 21.4% / MCP 42.9% / A2A 64.3% caught. All three made about the same number of LLM calls with the same decision pattern, so the spread is sampling noise (temperature 0.2) on a tiny set. Five-requirement smokes can find bugs in the pipeline but cannot rank variants.
- Decision: the sweep (63 requirements) runs the default configuration 3 times (`ours__default`, `__rep2`, `__rep3`; +2 runs, ≈ 7 GPU-hours) to measure the run-to-run spread. A variant counts as better or worse only if its difference from the default exceeds that spread, and the winners are confirmed in the finals (3 runs each). The report gives the spread next to every sweep difference. Sweep runs: 29.
- Also measured: MCP and A2A work as intended (66 MCP tool calls, 16 A2A agent calls, no errors) at about the same generation time as in-process (13–14 vs 13 min for 19 tests), so their overhead is small.

**D-88 · The ReAct variant was broken; fixed before the sweep** — Claude · fixed
- Smoke run 4: all 19 ReAct tests came from the fallback planner (`react_fallback` = 19/19). Llama 3.1 8B called `submit_plan` with the literal argument `{"plan": "plan JSON"}`, copied from my tool description, and then wrote the real plan as plain text, which was not accepted. So the 14.3% measured the fallback, not the agent.
- Fix: the plan argument is now a JSON **object** with a concrete example in its description; string or object are both accepted. A plan written as plain text is run, if no run has happened yet, or submitted, because that is what the agent means. A submission without steps gets a specific error with the expected shape. Rerun on the same 5 requirements once FastAPI is free; the rate of `react_fallback` is now reported with every ReAct run.
- Rerun (smoke v5): the agent now acts itself (52 `run_plan` calls + 11 plans written as text), and fallbacks fell from 19/19 to 8/19, all after `search_operations > derive_checks`. In those 8, Llama 3.1 wrote a tool call *as text* in its own JSON format with an invented name (`{"name": "write_plan", "parameters": {...plan...}}`). Such text tool calls are now unwrapped: a known tool is executed; a plan inside an unknown one is run or submitted. Applied to the recorded turns, this recovers usable plans from 5 of 12; the others were cut off or had no steps. The sweep run (63 requirements) reports the remaining fallback rate. Score on 5 requirements: 7.1% (noise, D-87).

**D-89 · Finding: on precise wording, direct rewrite beats the catalogue** — Claude · open (precision layer for the precise finals; decide with the sweep, Wed 4 Nov)
- Offline precision layer, prompt v2, precise wording (106 requirements, Llama judge), run on a Kaggle T4: catalogue recall 0.474 / soft 0.519 / contradictions 0.136 / unsupported 0.103 / 6.8 checks; direct rewrite **0.541 / 0.601 / 0.075 / 0.067** / 6.8 checks. On vague wording both had about the same recall (0.39–0.40), and the catalogue had fewer unsupported but more contradicting checks (D-43, D-58).
- Reading: the catalogue's guesses (format details, implied properties) help when the requirement says little, but compete with explicit statements when it says a lot. Even with precise text, only about half the gold checks are recovered. That bounds what any precision layer gives the planner, and the human labels (D-71) will show how much of the gap is the judge.
- Open: whether the final configuration keeps one precision variant for both wordings (simpler, one configuration) or picks by requirement length, the way retrieval does (D-78). Decided with the sweep (vague) plus one precise check run per candidate.

**D-90 · Lessons from the first Kaggle night: run-length estimates, the session end, and connection drops** — Claude · fixed
- (All times IST.) The two sessions ran from 23:20 (23 Oct) to 11:08 (24 Oct) as planned. 33 jobs ran: baselines B1/B2/B3/fuzzer, the golden checks on all four instances, and the development smokes.
- **A run overran the session end.** B2 precise run 1 on FastAPI started at 03:21 with an estimate of 5.9 h (from laptop timings) and was still running at 11:08, when the tunnel disappeared (Cloudflare error 1016). Its last 17 of 197 tests failed to generate. It was stopped at 11:40 and marked failed, to be rerun from scratch. Fix: the queue's run-length table now uses the times measured on T4s tonight. B2 is much slower than estimated: FastAPI ≥ 8.3 h (now 9.0), NestJS 7.8 h (now 8.0), TodoApp 2.0 h (now 2.1); the 1.2× + 15 min margin stays.
- **Short connection drops through the tunnel** (5 `APIConnectionError`s each) hit three baseline runs. Our agent retries every call 5 times, so it is unaffected, but RESTestBench's tools do not retry. In B1 vague run 2 on NestJS one test (requirement 15) failed to generate, so that run is marked failed and will be rerun. The two B2 runs (NestJS vague 1, TodoApp vague 3) lost no test; they are flagged in `run_meta` (`transient_llm_errors: 5`) and kept, and the report lists them. If more drops appear, a retrying wrapper around the baseline tools' client is the fix; it would change only the transport, not their prompts.
- After the session end the queue started nothing (no server had time left); the laptop GPU and local Ollama stayed off all night.

**D-91 · Server notebook ready for teammates: one shareable file, settings checks** — Claude · fixed
- Shareable file: `PreTrained_LLM_Server.ipynb` (workspace root; same as `pretrained/kaggle_llm_server/pretrained_llm_server.ipynb`). It is self-contained: no dataset or secret from Sohail's account is needed (it downloads Ollama 0.40.3, cloudflared and `llama3.1:8b` itself). `pretrained-llm-server-a` and `-b` on Sohail's account are two identical copies, needed only because one notebook runs one session at a time.
- Added: a preflight check that stops at once with a `>>> SETTINGS PROBLEM` message naming the fix, if fewer than 2 GPUs are attached or Internet is off (both need a phone-verified Kaggle account). Instructions cell: phone verification; a second copy for a second session; `RUN_HOURS` to spend less than 12 h of quota per session. Cell ids added (silences an nbformat warning in the log). Tested locally: the GPU check stops with the right message.

**D-92 · Second server batch: a teammate's two sessions** — Team · in force until 02:35 IST 25 Oct
- (Times IST.) A friend ran the shared notebook as two sessions (links pasted by Sohail at ~14:50 on 24 Oct; valid until about 02:35–02:37 on 25 Oct). All four servers answered the real planner prompt in 11.0–13.6 s with the same answer as the earlier sessions. The links are stored only in `secrets/` and mapped one per instance (FastAPI, FastAPI copy, NestJS, TodoApp). The queue runner was restarted so it uses the T4-measured run lengths (D-90).
- First jobs: B2 precise run 1 rerun on FastAPI (original; its requirement-15 part follows), B2 vague run 2 on the FastAPI copy, B1 vague run 2 rerun on NestJS, B3 precise run 3 on TodoApp.

**D-93 · TodoApp baselines complete; second NestJS instance takes the freed server** — Claude · default
- (IST.) By 17:00 on 24 Oct every TodoApp baseline run (B1/B2/B3 × 3 runs × 2 wordings, fuzzer × 3) was done, and the TodoApp server sat idle. Remaining work: NestJS 13 jobs ≈ 51 h on one instance, FastAPI 20 jobs ≈ 45 h on two.
- `make_service_copy.py` now also suffixes fixed `container_name`s and named volumes (NestJS has `nestjs_realworld_app/_db` and the volume `nestjs_realworld_mysql_data`, which a copy would otherwise share) and rewrites `localhost:<port>` in the service's own scripts (NestJS's start and reset scripts wait on `localhost:3000`). `nestjs-realworld@b` runs on 3100 / 3407. Checked: both NestJS instances answer side by side. Its golden check is first in line, before it takes baseline jobs.
- The friend's TodoApp server is now mapped to `nestjs-realworld@b`.

**D-94 · Long runs are split into requirement chunks of at most ~4 h** — Claude · default
- Why: B2 on FastAPI needs 9–11 h and on NestJS ~8 h, close to a whole 12-hour Kaggle session. B2 vague run 2 on the FastAPI copy (started 14:59 IST on 24 Oct) was on pace to finish generating around 02:25 IST, about 10 minutes before the friend's sessions end (~02:35). If it missed, ~11 GPU-hours would have been lost. Only generation needs the server; executing and scoring the generated baseline tests needs no LLM. The job is checked again at 01:30 IST.
- Decision: the queue splits any not-yet-done single-service run into requirement chunks of at most ~4 h (`MAX_PART_HOURS`; exp_id `…__partN`), on top of FastAPI's requirement-15 part (D-85). Analysis merges `__partN` and `__r15` parts into one run (per-service sums). A run already going stays whole (no duplicates; checked). A failed earlier run (e.g. B2 precise run 1 on NestJS) is redone as chunks.
- Effect: no part can be cut off by a session ending if the time check (D-90) lets it start, and the last hours of a session are used, because shorter parts still fit.
- Chunking does not change any result: every requirement is generated and scored exactly as in a whole run; the harness treats requirements independently.

**D-95 · Default configuration frozen; the sweep starts on the FastAPI instances** — Claude · fixed (T3 of the schedule)
- (IST.) Third server batch: Sohail's links for two sessions at 12:40 on 25 Oct, valid until ~00:35 on 26 Oct. All four answered the planner prompt in 12.6–15.1 s.
- Freeze: nothing in the default pipeline changed after smoke run 4 (21.4% / 29.8% on 5 requirements). The remaining open questions are decided by the sweep (judge: D-75; architecture rows incl. the twice-fixed ReAct: D-88) or by the finals (precision layer for precise wording: D-89). So the default is frozen (`configs/agent/default.yaml`), the critical path (Plan/Remaining_Work_Schedule.md: T3 → T5) moves, and B4 can start once a server is free.
- Sweep: `configs/queue_sweep.yaml` runs on `fastapi` and `fastapi@b` (default ×3 first, then the variants; each FastAPI run is split into its 62-requirement main part and the requirement-15 part on the original instance, D-85). The baseline queue leaves the FastAPI instances alone (`skip_instances`) and runs the NestJS baselines on both NestJS instances. The FastAPI baselines resume after the sweep. Both runners count each other's jobs (max_parallel 4 each; instance slots are the real limit).
- The 3 rows that need the fine-tuned models (`precision: finetuned`, `model: sft`, `model: sft_dpo`) are held back. These Kaggle servers only carry the base model, so they need a server notebook that also loads the merged fine-tuned GGUFs (a Kaggle dataset). That is the next infrastructure step; the DPO model has to be trained first anyway (D-73).
- First jobs (12:45): `ours__default` (FastAPI), `ours__default__rep2` (FastAPI copy), B2 precise run 1 part 1 (NestJS), B2 vague run 2 part 1 (NestJS copy).

**D-96 · Sweep variants get their own run-length factors; slow variants run in parts** — Claude · fixed
- (IST.) At 18:10 on 25 Oct the blackboard run (`ours__sweep__architecture__blackboard`, FastAPI copy) was on pace to write its last test at about 02:14, while the servers end at about 00:35. The queue had estimated every sweep run like the default (~3.5 h). Stopped at 18:12 after 0.8 h. The still-running old queue process restarted it whole at 18:12 before the queue was restarted, so it was stopped again at 18:24; since then `…blackboard__part1` runs (≈3.2 h).
- Measured per test on the same 5 FastAPI requirements (smoke v4/v5, dedicated server): default 41 s, MCP 44 s, A2A 41 s, ReAct 51 s, multi-persona 75 s, **blackboard 146 s (3.6×, ≈ 12 h for a full FastAPI run)**. The queue's estimates now multiply by these factors (`CONFIG_FACTOR`), with cautious guesses for untimed variants (full spec 1.3×, query rewriting 1.15×, self-consistency 1.4×, raw generation 0.6×, …). Chunking (D-94) then splits long variants: blackboard 4 parts, multi-persona 2, ReAct 2; every waiting sweep part ≤ 4.0 h.
- Lesson: after changing the queue's code, restart the runner at once, because the running process keeps its old rules.
- Effect on results: none (chunks are merged; every requirement is generated and scored as in a whole run). The sweep's compute grows: blackboard alone ≈ 12 h instead of 3.5 h.
- First sweep results (FastAPI vague, all 63 requirements): default run 1 caught 13.5% (tests made with a bug planted 10.4%), run 2 caught 11.6% (12.5%).

**D-97 · A server is busy only while its job is still generating; extra service copies fill the scoring gaps** — Team (Sohail: "make sure to utilize the full kaggle run") / Claude (design) · default
- Why: after a run has written all its tests it scores them against every planted bug for 30–45 min (FastAPI). That needs the service instance but not the model, so the run's Kaggle server sat idle, roughly 15–20% of FastAPI server time.
- Decision: two more copies each of FastAPI (`@c` +200, `@d` +300) and NestJS (`@c`, `@d`), made with the same script (D-84/D-93), each mapped to the server of an existing instance (fastapi@c ↔ fastapi's server, @d ↔ @b's; the same for NestJS). The queue now counts a *server* as busy only while some job on it is still generating, judged per process from its `PRETRAINED_LLM_URL` and whether its run has written `generation_manifest.json`. So when a job reaches scoring, its server starts the next job on a free copy. A server never serves two generating jobs, so the run-length estimates stay valid. Golden checks and the fuzzer need no server and ignore the rule. `max_parallel` is now 8 (4 generating + 4 scoring); the RAM guard (≥ 4 GB free before any start) still applies.
- Safety: golden checks on all four new copies go first (no server needed); the FastAPI copies sit in the sweep queue, the NestJS copies in the baseline queue.
- Started 20:25 IST on 25 Oct.

**D-98 · Near a session's end, NestJS servers also serve FastAPI copies** — Claude · default
- (IST.) At 21:10 on 25 Oct, with 3.4 h left, no remaining NestJS job fit any more (all ≈ 3–4 h), and the server of `nestjs-realworld@b` sat idle since 20:35. The sweep still had FastAPI jobs that fit (ReAct parts ≈ 2.2 h, requirement-15 parts ≈ 0.2 h).
- Decision: the server map lets one server serve instances of both services. `fastapi@d` now points at the NestJS-b server, and a new copy `fastapi@e` (+400, golden-checked first) at the NestJS server. The one-generating-job-per-server rule (D-97) keeps every server to one generating job, so a FastAPI job only starts there when that server's NestJS job is past generation. The queues re-read the map every loop; no restart needed.
- Result: at 21:13 ReAct part 1 started on `fastapi@d`, and all four servers are generating again. Golden checks: FastAPI copies c, d and NestJS copies c, d all 100% (20:47–20:49).
- Default configuration, 3 sweep runs (FastAPI vague, 63 requirements): caught 13.5 / 11.6 / 11.6% (mean 12.2%, sd ≈ 1.1); tests made with a bug planted 10.4 / 12.5 / 11.7%. This is the run-to-run spread the other sweep variants are judged against (D-87).

**D-99 · ReAct is ~2.3× the default on the full set; parts are now ≤ 1.5 h, and existing parts are never reshuffled** — Claude · fixed
- (IST.) At 22:10 on 25 Oct `ours__sweep__architecture__react__part1` (31 requirements, 116 tests) ran at ~27 tests/h and would have written its last test at ~01:20, after the session end (~00:35). On the full requirement set ReAct is ≈ 2.2–2.3× the default, not 1.25× as on the 5 smoke requirements (more turns per test). Stopped at 22:12 after 1.0 h, marked failed; it reruns unchanged in a fresh session (≈ 4 h). Factor set to 2.3. Blackboard measured on its part 1: ≈ 3.3× (factor 3.6 kept).
- Parts are now at most ~1.5 h (`MAX_PART_HOURS`), so that pieces still fit when a session has 2–3 h left; the per-part overhead (service start, scoring) is small. To make changing the size safe, parts that already exist (done, running or failed) keep their requirements exactly; only requirements no existing part covers are chunked anew, numbered after the highest existing part. Checked over every job in both queues: no requirement appears in two parts. Blackboard: parts 1 and 4 done; the rest became parts 5–9. ReAct: part 1 kept, parts 2–4 new.
- Both queue runners restarted at once (lesson from D-96). By 22:15 all four servers were generating again (blackboard parts 4 and 5, NestJS B2 precise run 2 parts 1 and 2).

**D-100 · Two requests per GPU (server notebook) and two generating jobs per server (queue)** — Team (Sohail asked for it, after the utilisation check) / Claude (design) · in testing
- Why: measured utilisation of the four servers on 25 Oct (12:40–23:46 IST): 81 / 85 / 89 / 97% of the session had a job generating on them (≈ 88% overall; ≈ 78% if the work later thrown away is excluded). Even while a job is generating, its GPU waits whenever the job runs a test against the service between two LLM calls. Two jobs per GPU with two parallel request slots fill those pauses (llama.cpp batches the two sequences).
- Notebook: `PARALLEL` (default 2) sets `OLLAMA_NUM_PARALLEL`. Each request keeps the full 24k context; the KV cache doubles (~3 GB at q8_0), which fits a 15 GB T4 next to the 4.9 GB model. The share block says "(2 parallel requests per GPU)". Rebuilt `PreTrained_LLM_Server.ipynb`.
- Queue: the endpoints file gets `slots_per_server`. A server takes up to that many generating jobs (per process, as in D-97). With 2 slots, estimates are multiplied by `SHARED_SLOWDOWN` (1.5 until measured), so the end-of-session check stays safe. With 1 slot nothing changes (tonight's sessions).
- Tests: (a) proxy with two simultaneous 25 s requests through the real OpenAI client: both correct, both 25.0 s (passed). (b) `local_cpu_test.py`: the notebook's pieces on the laptop CPU (no GPU, D-81) with a real tunnel, two simultaneous requests, 2 slots, wrong secret refused; runs after tonight's jobs finish. (c) `throughput_test.py`: real prompts at concurrency 1 vs 2 on a Kaggle T4. That gives the actual gain and the value of SHARED_SLOWDOWN; it needs a short Kaggle session (asked Sohail first).
- Test (b) results (26 Oct 00:13–00:45 IST): the first two attempts could not open a Cloudflare tunnel from this laptop. Its network blocks Cloudflare's tunnel port 7844 for QUIC/UDP *and* TCP (`--protocol http2` failed too); on Kaggle the tunnels worked in all three batches (12 links). Added `TUNNEL_PROTOCOL` (optional; the notebook's default is unchanged) and a `NO_TUNNEL` mode for the local test; fixed the test leaving a cloudflared process behind on failure. Third attempt, through the proxy without tunnel: **PASS**. Ollama ran CPU only (`inference compute id=cpu`; the laptop GPU stayed at 0%, 12 MiB) with `n_seq_max = 2`; two simultaneous requests both answered correctly at the same time; a wrong secret got 403. The timing ratio is not meaningful (the solo request included model loading); the T4 throughput comes from test (c).

**D-101 · One local request queue (dispatcher) in front of all model servers** — Team (Sohail's design) / Claude (build) · default from the next batch
- Idea (Sohail): instead of pairing each run with one server, all runs send their LLM requests to one queue and any free model slot takes the next one; run enough runs that the queue always holds ~1–2 requests, so no model compute is wasted.
- `scripts/llm_dispatcher.py` (aiohttp, 127.0.0.1:11500/v1, OpenAI-compatible): FIFO queue; any free slot on any live server takes the oldest request. A conversation (same first two messages) returns to the server it last used when that server is free, to reuse its prompt cache. A request that fails on a server (connection lost, Cloudflare 52x/530/1016, empty body) goes back to the front of the queue and another server takes it (up to 4 tries); the failed server is pinged every 30 s and revived. No new work goes to a server in its last 2 minutes. A request waits at most 540 s (below the clients' 600 s timeout), then gets a clear 503. Servers come from `secrets/llm_backends.json`, re-read every 30 s; `kaggle_llm_server/add_links.py NAME` turns a pasted log block into entries (end time, slots). `/status` gives queue length and per-server use; `results/dispatcher_requests.jsonl` logs every request with the server *name* only.
- Queue runner, dispatcher mode (active when the backends file exists): every model-using job gets `PRETRAINED_LLM_URL=http://127.0.0.1:11500/v1`; any free service instance of the job's service can take it (originals and all copies); a new model-using job starts only while the dispatcher's queue is ≤ 1 and fewer than slots + 2 jobs are generating, then waits 150 s for it to ramp up. A job must still fit before the *last* server ends (estimate × 1.5 shared-server allowance × 1.2 + 15 min). Jobs that need no model (golden, fuzzer) ignore those limits.
- What it replaces: the fixed server-per-instance map (D-83/D-84), the scoring-gap workaround (D-97/D-98) and the per-server slot counting (D-100's queue part). A session ending mid-request no longer kills the request if another server is up.
- Fairness: a run's requests can now be answered by different T4s (same Ollama version, model digest and settings). The request log records which server answered each request; `run_meta` records the dispatcher as the server.
- Tests: (1) `scripts/test_dispatcher.py`, 14/14 with fake servers: round trip; slot limits never exceeded under 25 concurrent requests (peaks 1/2/2, wall 5.0 s = ideal); failover after 530; revival after recovery; dropped connection mid-answer → retried elsewhere; keep-alive spaces pass through; live reload; conversation affinity (6/6 on one server); no work for a server about to end; no server → 503 after the max wait; 300 requests × 40 clients without an error; our agent's LLM class through it; no secrets in the log. (2) Queue decisions, 12/12 with simulated states. (3) Real chain on the laptop CPU (`scripts/test_dispatcher_real.py`; CUDA and Vulkan off, GPU stayed at 0% / 12 MiB): real Llama 3.1 8B (2 slots) → the Kaggle notebook's proxy code → dispatcher. Real-prompt replay: 4/4 answered at concurrency 1 and at 2, no errors, the planner answer parses. One real RESTestBench job (B1, TodoApp requirement 5) through the harness: exit 0, status ok, evaluated; the dispatcher served all 9 of its requests (9 ok, 0 retries, 0 failures), and the request log has no secret. **PASS.** The CPU speed numbers (gain ×2.59 at concurrency 2) are not meaningful for a T4 and are inflated, because the second pass reuses the prompt cache from the first. The T4 gain still comes from the Kaggle throughput test (D-100 (c)), which now runs through the dispatcher in the first minutes of the next batch.
- Use (next batch): `scripts/start_dispatcher.sh`; for each pasted log block `kaggle_llm_server/add_links.py <name>`; creating `secrets/llm_backends.json` switches the queue runners into dispatcher mode (the old `queue_endpoints.json` is then ignored).

**D-102 · Batch 4 on Sohail's account, first batch with the dispatcher and 2 requests per GPU; T4 gain measured: 1.40×** — Team (Sohail: use my account, 30 h available) / Claude · in force until ~13:37 IST 26 Oct
- (IST.) Pushed `pretrained-llm-server-a/-b` (version 2: PARALLEL = 2) at 01:49 on 26 Oct. Both sessions were ready at ~01:53, with model digest SAME, public checks passed and "(2 parallel requests per GPU)". The links went straight from the live logs into the dispatcher (`add_links.py`: sohail-A-1/-2, sohail-B-1/-2, 2 slots each, until 13:37). About 24 of the 30 h of quota.
- **T4 throughput (D-100 (c)), measured before any run started:** the same 12 real prompts on separate GPUs at the same time, so no GPU reused another's prompt cache. One request at a time: 25.3 and 23.2 tok/s (mean latency 9.0 / 10.3 s). Two at a time: 31.8 and 36.2 tok/s (13.9 / 12.6 s). **Gain ×1.40 per GPU, each request ×1.37 slower**, with no errors and every plan parsed. `SHARED_SLOWDOWN` stays at 1.5 as a safety margin. In real runs the second slot also fills the pauses while a job runs its tests, so the gain should be at least this.
- Queues restarted in dispatcher mode at 01:56 (`max_parallel` 12; the RAM guard ≥ 4 GB still decides how many runs the laptop carries). Sweep on the 5 FastAPI instances, baselines (NestJS, TodoApp) including the newly queued **B4** (18 job parts, timeout 600 s), and the FastAPI baselines still after the sweep. By 02:00: 4 runs, dispatcher 4/8 slots busy, 12 requests answered, 0 retries, 0 failures.
- `status_report.py` now reads the dispatcher's server list for the session end (it had read the old endpoints file and shown false "after session end" flags).

## D-103 — Two more FastAPI copies (f, g) once NestJS/TodoApp work ran out (26 Oct, ~10:40 IST)

- **Situation at 10:31 IST:** the NestJS and TodoApp baselines (B1–B4) were all finished; only FastAPI work was left (67 sweep job parts, 43 FastAPI baseline parts held back until the sweep ends, D-95). All 5 FastAPI instances were busy with sweep runs, so the dispatcher had only 3–4 of its 8 slots busy and an empty queue: the limit was service instances, not model servers.
- **Decision:** make `fastapi@f` (+500 ports) and `fastapi@g` (+600) with `make_service_copy.py`. Each FastAPI copy needs ~0.3 GB RAM (docker stats); 10 GB was free. Both get a golden check (RESTestBench reference tests, D-86) before any run uses them, queued first in `queue_sweep.yaml`. The baselines queue skips f and g too (D-95: sweep first).
- **Slip and fix:** in the ~30 s between creating the copies and queueing their golden checks, the sweep queue started `ours__sweep__content__verdict__part2` on f and the baselines queue started `b3_prompt_fixed__vague__rep2__fastapi__part1` on g. Both were stopped after one test each, and their partial result folders were deleted so they rerun from scratch on a verified instance. **Lesson:** queue the golden check (and update `skip_instances`) *before* creating a copy, because the queues pick up new folders in `service_copies/` straight away.
- **Expected effect:** at most a few more parts fit into this session (the time-fit rule needs ≥ 2.1 h for the smallest 1.03 h part). The copies mainly pay off in the next batch: 7 FastAPI instances instead of 5 keep 8 slots busy.

## D-104 — Teammates send the whole link block, not just the two URL lines (26 Oct, ~13:10 IST)

- In dispatcher mode (D-101), `add_links.py` reads the slots per server from the block's "(N parallel requests per GPU)" line and the session end from "valid until about HH:MM UTC". With only the two URL lines, each server would count as 1 slot (half the throughput, D-100) and have no end time, so the dispatcher's end margin and the queues' time-fit rule could not work.
- So the instructions cell of `PreTrained_LLM_Server.ipynb` (and its copy `pretrained/kaggle_llm_server/pretrained_llm_server.ipynb`) now asks for the whole block, from the first `====` line to the last, including any NEW LINK block. The code is unchanged: it is the PARALLEL = 2 version used for batch 4.

## D-105 — Batch 5: a teammate's two sessions through the dispatcher (26 Oct, ~17:55 IST)

- A teammate ran the updated notebook (PARALLEL = 2, D-104) as two sessions and sent the whole link blocks. Added as `mate-A-1/-2` (until 05:32 IST 27 Oct) and `mate-B-1/-2` (until 05:40 IST), 2 slots each, 8 slots in all. Batch 4's expired servers were removed from `secrets/llm_backends.json`.
- The laptop had been restarted (13:49 IST), so the dispatcher and both queue runners were started again; no run was cut off (nothing was running after 13:03). Check: 8 requests sent at once through the dispatcher, all answered in 1.0–1.4 s, 2 by each server.
- Work: the sweep continues on the 7 FastAPI instances (one new run every 150 s); the FastAPI baselines still wait for the sweep (D-95).

## D-106 — Two more FastAPI copies (h, i) and a utilization check every 20 min (26 Oct, ~18:10 IST)

- The user asked that the queue be checked at a reasonable interval so the models are never idle, starting more runs when needed. A session job checks every 20 min (:07/:27/:47) until the servers end (~05:40 IST 27 Oct): status, both queue runners alive, dispatcher health, RAM, slot-time busy over the last 20 min, stuck or failing runs.
- **Why more instances:** batch 4 averaged 6.1 of 8 slots busy with about 9–11 runs at a time (one run keeps ~0.6 slot busy; the rest of its time it runs tests or waits on the service). Only FastAPI work is left, so 7 FastAPI instances would leave ~3 slots idle. Each extra instance costs ~1 GB RAM (containers ~0.3 GB, run ~0.7 GB); 11 GB was free.
- **Decision:** `fastapi@h` (+700 ports) and `fastapi@i` (+800). Following D-103, their golden checks and the baselines' `skip_instances` were added *before* the copies were created; no run used them first. More copies (j…) only if the 20-min checks show idle slots and RAM stays ≥ 6 GB available (4 GB is the queue's guard; VS Code needs headroom, see memory).

## D-107 — A tenth FastAPI instance (j) to cover scoring gaps (26 Oct, 20:12 IST)

- With 9 runs, slot use was 7.5–7.8 of 8 (19:11–19:51 IST) but fell to 6.65 in the 10 min before 20:11, when two runs were scoring at once. A part's scoring takes ~15 min with no model use, and its instance stays busy, so 9 instances leave gaps whenever parts finish together.
- `fastapi@j` (+900 ports) added under D-106's rule (idle slots; RAM 7.5 GB available, ~6.5 GB expected after). Golden check and baselines `skip_instances` were queued before the copy was made (D-103). No further copies beyond j: the RAM margin would drop under 6 GB.

## D-108 — Parts sized to the time left; FastAPI baselines start before the sweep ends (27 Oct, ~00:55 IST)

- **Parts sized to the time left.** At most 1.5 h parts (D-99) stop fitting about 3 h before a session ends (fit rule: est × 1.5 × 1.2 + 0.25 h), so up to 10 instances would sit idle for the last ~3 h. `run_queue.py` now chunks parts *not started yet* with `part_hours()`. In dispatcher mode that is min(1.5, (hours left − 0.4) / (1.5 × 1.4)), at least 0.25 h, so the sizing has more margin than the fit rule (measured parts ran ~10–20% over their estimate under 10-run load). Parts that already exist keep their requirements exactly (D-99), so nothing running or finished is reshuffled. Dry run: 2.9 h left → 1.19 h parts (21 requirements), 2.0 h → 0.76 h (14), 1.2 h → 0.38 h (7). With a fresh 11.6 h session it is 1.5 h as before.
- **Baselines on copies f–j now.** Only 9 sweep parts were left (4 full-size, plus 5 requirement-15 parts that run on the original only); the 3 fine-tuned rows still wait for a server with those models (D-95). Keeping all FastAPI instances for the sweep would leave them idle. The baselines queue now skips only `fastapi` and `fastapi@b…e`: the sweep keeps 5 instances to itself (enough for its remaining parts) and can still use f–j, while the 43 FastAPI baseline parts (B1–B4, fuzzer) use f–j.
- Both queue runners were restarted to load the change. The 11 running jobs continued unaffected.

## D-109 — Sweep fully started: every FastAPI instance goes to the baselines (27 Oct, 02:53 IST)

- By 02:51 IST every sweep part had started (the queue had nothing pending; the 3 fine-tuned rows remain commented out, D-95). The original instance and copies c and d were idle because the baselines still skipped them (D-108), and slot use fell to 3.5 of 8 over 10 minutes. The baselines queue's `skip_instances` is now empty, so the FastAPI baselines (B1–B4, fuzzer) use every free instance. Parts are sized to the time left (D-108). The first start was B2 vague rep2's requirement-15 part on the original instance.
- Lesson: when a queue runs out of work, release its reserved instances at once. The 20-minute check now looks for "free instance + empty queue" as well as low slot use.

## D-110 — Two late B2 parts stopped; B2 FastAPI estimate 9.0 → 10.5 h (27 Oct, 03:35 IST)

- After D-109, eight B2 precise rep2 FastAPI parts started between 02:59 and 03:30 IST. B2 is the most model-heavy tool (long prompts, refinement rounds), and the servers became oversubscribed: queue 3, mean wait 14.6 s, 8.3 of 8 slots over 10 minutes. Parts 5 and 6, both started after 03:15, were forecast to finish generating at ~05:44, after the servers end (05:38 with the dispatcher margin). Both were stopped after 2–3 tests, and their folders were deleted so the requirements are re-chunked. Nothing that runs to completion was lost; the model time they would have used goes to the other runs.
- Measured this batch: B2 precise parts of 7 requirements generate 14–20 tests/h under 10-run load, i.e. ~1.3–1.7 h each, against an estimate of 1.02 h × 1.5. `EST_HOURS[b2_refinement, fastapi]` is raised from 9.0 to 10.5 h, so late B2 parts are cut smaller (5 requirements with 2.1 h left) and the fit rule is stricter. The dispatcher-mode rule (start only while the queue ≤ 1) keeps new parts from starting while the servers are saturated.
- The baselines runner was restarted to load the estimate.
- (03:52 IST) Restarting the runner let two new 5-requirement B2 parts start (9 and 10) during a short dip in the queue. Part 9 was again forecast past the end (05:47) and was stopped after 2 tests and deleted. With the servers saturated (mean wait 13–16 s), new parts add no throughput and only slow the rest, so `results/queue.pause` now holds new starts. The 20-minute checks lift it if slots go idle and a small part still fits.
- (04:53 IST) Pause lifted with 0.8 h left. The saturating parts were finishing and slot use fell to 6.6 of 8. The parts that still fit are tiny (1–4 requirements, est ≤ 0.25 h; the fit rule stops new starts by itself below ~0.7 h left, at ~04:58).
- (05:13 IST) Last ~25 minutes: the fit rule's smallest estimate (0.2 h, + 0.25 h margin) blocked every start, but 1-requirement B3 parts had just run in 5–6 min end to end. Six 2-requirement B3 precise rep2 parts (part17–22) were started by hand on the free copies d, e, f, g, i, j through `run_queue.start` (same naming and run_meta as queue parts; the chunker treats them as existing parts and re-chunks the rest). If any is cut off at 05:38, the 05:52 wrap-up moves it to `results/_cutoff/`.

## D-111 — Batch 5 summary and cut-off handling (27 Oct, ~06:00 IST)

- **Use:** 4 servers × 2 slots for 11.7 h; dispatcher slot-time busy 86% (dips came from scoring phases, start ramps and the last hour). 13,158 requests answered, 6 retried on another server, 0 failed.
- **Done:** 85 parts finished, all `ok` and each with a generated test for every one of its requirements. The sweep (all variants except the 3 fine-tuned rows) is complete. Among the FastAPI baselines, B3 vague rep2 and most of B2 precise rep2 are done.
- **Cut off:** two 1-requirement parts started at 05:32 (B3 precise rep2 part25 = requirement 8, part27 = requirement 28) were still generating when the servers ended; their requests timed out after 05:40. They were stopped and moved (not deleted) to `results/_cutoff/baselines/`, so the queue sees those requirements as uncovered and reruns them. No other part was affected.
- **Left:** baselines queue 36 parts ≈ 44 est-h (≈ 66 T4-slot-h: B2 16 parts / 23 h, B3 8 / 10 h, B4 12 / 10 h), about one 2-session batch. Sweep queue: nothing until the fine-tuned rows can run (server with the merged fine-tuned GGUFs, D-95). The pause file had already been removed at 04:53, and the queue runners are idle, waiting for new links.

_New decisions are added here as they are made._

## D-112 — Mid report written; report tables merge sweep parts and leave out incomplete runs (30 Oct, IST)
- `Reports/Mid/PreTrained_Mid.tex` / `.pdf` (7 pages, proposal style; the proposal itself untouched) covers what the course asks for in the mid submission: progress, datasets, baselines, ablations (the one-at-a-time sweep), results, challenges with mitigations, timeline, and the GitHub repo link (https://github.com/SohailTSM/Requirements-as-the-Oracle). The 20 Oct draft is replaced.
- `scripts/make_tables.py`: the part suffix pattern now also matches sweep parts (`__partN`, `__r15` without a service name), so each sweep variant is one row; a service's score counts only when every requirement of that service is scored (63 / 29 / 14), so FastAPI baseline runs still in progress are left out instead of pulling the means down. Δ in the sweep table is against the mean of the 3 default runs.
- Why: the old table averaged half-finished FastAPI runs (e.g. B2 precise 32.8 ± 28.4 instead of 49.6 from its one complete run) and listed sweep parts separately.
- Framing chosen for the report: the main result is the adaptation gap on FastAPI (B2 vague 17.0 → 8.1 when generated against mutants; ours 12.0 → 11.3, best variants ~17 → ~16), stated together with the weakness that our default is below B2 on the correct service and ~70% of our tests fail there.
