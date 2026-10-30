"""Baseline B3: RESTestBench's refinement loop with only the prompts corrected.

Same model, same loop, same rounds, same HTTP-trace feedback as B2. Changes:
  1. The judge's few-shot example that teaches copying the observed value is replaced
     by examples of genuine test bugs (wrong field name, missing auth header).
  2. The generator is told the service may be buggy and that expectations must come
     from the requirement/spec, never from observed responses.
  3. The judge's feedback is passed to the generator (USE_JUDGE_FEEDBACK_IN_REFINEMENT=True).
  4. The judge may answer API_BUG: the loop stops and the test is kept unchanged.

This isolates "better prompting" from our structural changes. The prompts are swapped in
the RESTestBench module at construction time; each benchmark job runs in its own process.
"""
import re

import tools.refinement_tool.refinement_tool as rt

from .logged_refinement import LoggedRefinementTool

GENERATOR_SYSTEM_PROMPT = rt.GENERATOR_SYSTEM_PROMPT.replace(
    "Generate ONLY the Python test code, no explanations.",
    "The service under test MAY CONTAIN BUGS. Expected values in your assertions must come from the\n"
    "requirement and the OpenAPI specification, never from what the service happened to return.\n\n"
    "Generate ONLY the Python test code, no explanations.")

REFINEMENT_USER_PROMPT = rt.REFINEMENT_USER_PROMPT.replace(
    "Check how you could use the Test Execution Output and the HTTP Trace to enhance the effectivness of the test.",
    "Use the Test Execution Output and the HTTP Trace ONLY to fix how the test runs (authentication,\n"
    "URLs, request payloads, ids passed between calls). The service may be buggy: do NOT change an\n"
    "expected value just because the service returned something different.")

JUDGE_PROMPT = """You are a test quality reviewer. Analyze the test execution results and decide what to do.
The service you are testing MAY CONTAIN BUGS. A test that fails because the service violates the
requirement is a GOOD test and must be kept.

## Requirement Being Tested:
{scenario_description}

## OpenAPI Specification:
{spec_content}

## Test Code:
```python
{test_code}
```

## Test Execution Output:
```
{test_output}
```

## HTTP Trace (actual requests/responses made during test):
```json
{http_trace}
```

## Your Task:
1. Compare the test case against the requirement.
2. Decide whether a failure comes from the test itself (wrong URL, missing authentication, wrong
   payload, wrong field name according to the specification) or from the service violating the requirement.
3. Expected values must follow the requirement and the specification, never the observed response.

## Response Format:
Respond with EXACTLY one of these three formats:

If the test itself is broken and needs fixes:
REFINEMENT: <short description of what needs to be changed>

If the test fails because the service violates the requirement:
API_BUG: <short description of the violated requirement>

If the test passes and asserts everything the requirement asks for:
FINISH: <short description why the test is correct>

Examples:
- REFINEMENT: The test validates a field 'email' but the specification names it 'Email'.
- REFINEMENT: The request is missing the Authorization header, so the call returns 401 before reaching the logic under test.
- API_BUG: The requirement says a second user cannot delete the item, but the service returned 200 and deleted it.
- FINISH: The test correctly asserts all required fields and fulfills the requirement because of...
"""


class PromptFixedRefinementTool(LoggedRefinementTool):
    APPROACH_NAME = "refinement-prompt-fixed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rt.GENERATOR_SYSTEM_PROMPT = GENERATOR_SYSTEM_PROMPT
        rt.REFINEMENT_USER_PROMPT = REFINEMENT_USER_PROMPT
        rt.JUDGE_PROMPT = JUDGE_PROMPT
        rt.USE_JUDGE_FEEDBACK_IN_REFINEMENT = True

    def _parse_judge_decision(self, decision_text):
        if re.search(r"\bAPI_BUG\b", decision_text, re.IGNORECASE):
            self._history.append({"type": "api_bug", "decision": decision_text})
            return False, ""          # stop refining; keep the current test
        return super()._parse_judge_decision(decision_text)
