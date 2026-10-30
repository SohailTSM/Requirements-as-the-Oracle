"""Precision layer: expand a (vague) requirement into atomic checks before any execution.

Variants (component sweep):
    none              no expansion; downstream gets the requirement text only
    direct_rewrite    "rewrite this requirement precisely" -> checks
    catalogue         guided by the property catalogue below (our default)
    self_consistency  catalogue prompt sampled 5 times; keep checks that most drafts agree on
    finetuned         catalogue prompt answered by our SFT adapter (added after training)

Each check: {id, description, property, kind}. kind = "meaning" (business rule, locked
against execution feedback) or "format" (field names, envelopes, exact status codes the
requirement doesn't fix; changeable only with spec evidence).

Few-shot examples use a made-up library API, so no benchmark content leaks into prompts.
"""
from __future__ import annotations

import json
import re

from .spec import Operation

VARIANTS = ["none", "direct_rewrite", "catalogue", "self_consistency", "finetuned"]

PROPERTY_CATALOGUE = """\
- explicit: anything the requirement states directly (a status class, an error message it quotes, a value it names)
- success_status: the main call succeeds (2xx) or, for a negative scenario, is rejected (4xx); the exact code is format unless the requirement states it
- echo: fields sent in the request come back unchanged in the response
- round_trip: what was created or updated can be read back with the same values
- ownership: a created resource belongs to the user who created it (owner field matches the caller)
- isolation: another user cannot read, change or delete someone else's resource, and the resource is unchanged afterwards
- before_after: an action changes state by exactly the expected amount (count goes up by 1, a flag flips, an item disappears)
- persistence: after a change, a separate read shows the change was saved
- not_found: acting on something that does not exist is rejected (usually 404)
- validation: invalid input (missing, too long, wrong format) is rejected (usually 422 or 400) and nothing is created
- filter: every returned item matches the filter, and every matching item is returned
- uniqueness: creating a duplicate is rejected
- auth_required: calling without valid credentials is rejected (401/403)"""

# Prompt version v2 (20 Oct): v1 had two examples and the model copied Example 1's six checks
# (ownership, "appears in the list", "status 201") into unrelated requirements, including deletes.
PROMPT_VERSION = "v2"

FEW_SHOT = """\
Example 1
Requirement: "Borrowing a book works."
Checks:
[{"description": "borrow request succeeds (2xx)", "property": "success_status", "kind": "meaning"},
 {"description": "response book_id equals the requested book's id", "property": "echo", "kind": "meaning"},
 {"description": "the book's available copies go down by exactly 1", "property": "before_after", "kind": "meaning"},
 {"description": "the loan's borrower_id equals the calling user's id", "property": "ownership", "kind": "meaning"}]

Example 2
Requirement: "Removing a book from the catalogue should work for librarians."
Checks:
[{"description": "a librarian's remove request succeeds (2xx)", "property": "success_status", "kind": "meaning"},
 {"description": "reading the removed book afterwards is rejected (404)", "property": "persistence", "kind": "meaning"},
 {"description": "the removed book no longer appears when listing books", "property": "before_after", "kind": "meaning"}]

Example 3
Requirement: "A member must not be able to cancel someone else's reservation."
Checks:
[{"description": "member B cancelling member A's reservation is rejected (4xx)", "property": "isolation", "kind": "meaning"},
 {"description": "member A's reservation still exists afterwards", "property": "isolation", "kind": "meaning"},
 {"description": "rejection status is 403, the code documented for this operation", "property": "success_status", "kind": "format"}]

Example 4
Requirement: "Searching books by author gives the right results."
Checks:
[{"description": "search request succeeds (2xx)", "property": "success_status", "kind": "meaning"},
 {"description": "every returned book's author equals the searched author", "property": "filter", "kind": "meaning"},
 {"description": "a book by that author created beforehand appears in the results", "property": "filter", "kind": "meaning"}]
"""

CATALOGUE_PROMPT = """You are an expert API tester. A requirement for a REST API is often vague. Before writing
a test, list the concrete checks a good test must make, using the API operations below.

Property catalogue (properties a requirement usually implies):
{catalogue}

Rules:
- Only list checks the requirement states or clearly implies. Do not invent unrelated features.
- Prefer relationships ("equals the input title", "count goes up by 1") over guessed exact values.
- kind "meaning": the business rule the requirement is about. kind "format": details such as exact
  status codes or field names that come from the API documentation rather than the requirement.
- If the requirement quotes an exact value (status code, message, credentials), keep it exactly.
- The examples show the format only; do not copy their checks. Pick the properties that fit THIS
  requirement (usually 2 to 5 checks). After a delete, the resource must be gone, never "still listed".
- An echo check compares a response field with the SAME field sent in the request.
- A "format" check about a status code may only use a code listed under that operation's Responses
  below. If the right code is not listed there, do not add a format check for it.

{few_shot}
API operations:
{operations}

Requirement: "{requirement}"
Checks (JSON list only):"""

DIRECT_PROMPT = """You are an expert API tester. Rewrite the vague requirement below as a precise, testable
requirement, then list its individual checks as a JSON list of objects with fields
"description", "property" (use "explicit") and "kind" ("meaning" or "format").

API operations:
{operations}

Requirement: "{requirement}"
Output only the JSON list of checks:"""


def _ops_text(ops: list[Operation]) -> str:
    return "\n\n".join(op.to_text() for op in ops)


def parse_checks(text: str) -> list[dict]:
    """Extract the JSON list of checks from a model response; tolerate surrounding prose."""
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
    candidates = [m.group(0)] if m else []
    candidates.append(text)
    for c in candidates:
        try:
            data = json.loads(c)
            break
        except Exception:
            data = None
    if data is None:  # fall back to one object per line
        data = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{") and line.endswith("}"):
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    checks = []
    for i, d in enumerate(data if isinstance(data, list) else []):
        if isinstance(d, dict) and d.get("description"):
            kind = d.get("kind", "meaning")
            checks.append({"id": f"C{i + 1}", "description": str(d["description"]).strip(),
                           "property": d.get("property", "explicit"),
                           "kind": kind if kind in ("meaning", "format") else "meaning"})
    return checks


def derive_checks(requirement: str, ops: list[Operation], variant: str, llm,
                  samples: int = 5, sample_temperature: float = 0.7) -> dict:
    """Return {"checks": [...], "raw": [responses]} for one requirement."""
    if variant == "none":
        return {"checks": [], "raw": []}
    if variant == "direct_rewrite":
        raw = llm.complete(DIRECT_PROMPT.format(operations=_ops_text(ops), requirement=requirement),
                           max_tokens=1200, tag="precision.direct")
        return {"checks": parse_checks(raw), "raw": [raw]}
    prompt = CATALOGUE_PROMPT.format(catalogue=PROPERTY_CATALOGUE, few_shot=FEW_SHOT,
                                     operations=_ops_text(ops), requirement=requirement)
    if variant in ("catalogue", "finetuned"):
        raw = llm.complete(prompt, max_tokens=1200, tag=f"precision.{variant}")
        return {"checks": parse_checks(raw), "raw": [raw]}
    if variant == "self_consistency":
        # Sampling needs a higher temperature than the 0.2 used elsewhere (documented exception).
        drafts, raws = [], []
        for k in range(samples):
            raw = llm.complete(prompt, max_tokens=1200, tag="precision.self_consistency",
                               temperature_override=sample_temperature)
            raws.append(raw)
            drafts.append(parse_checks(raw))
        return {"checks": majority_checks(drafts), "raw": raws}
    raise ValueError(f"unknown precision variant: {variant}")


def majority_checks(drafts: list[list[dict]], threshold: float = 0.80) -> list[dict]:
    """Keep checks that appear (by embedding similarity) in a majority of drafts."""
    from .retrieval import _embedder
    flat = [(d_i, c) for d_i, d in enumerate(drafts) for c in d]
    if not flat:
        return []
    emb = _embedder().encode([c["description"] for _, c in flat], normalize_embeddings=True)
    need = len(drafts) // 2 + 1
    kept, used = [], set()
    for i, (d_i, c) in enumerate(flat):
        if i in used:
            continue
        members = [j for j in range(len(flat)) if float(emb[i] @ emb[j]) >= threshold]
        if len({flat[j][0] for j in members}) >= need:
            kept.append(dict(c, id=f"C{len(kept) + 1}"))
            used.update(members)
    return kept


# ---------------------------------------------------------------- support filter --
SUPPORT_PROMPT = """A tester listed checks for the API requirement below. Some checks may not follow from the
requirement at all (for example they were copied from an unrelated scenario).

Requirement: "{requirement}"

Checks:
{checks}

For each check, answer "yes" if the requirement states it or clearly implies it for the scenario the
requirement describes, and "no" if it is about something the requirement does not ask for, or
contradicts it. Answer with a JSON object only, e.g. {{"C1": "yes", "C2": "no"}}."""


def support_filter(requirement: str, checks: list[dict], llm) -> tuple[list[dict], dict]:
    """Drop checks the requirement does not support. Runs on every check before any service call,
    so it cannot react to a bug. If the answer cannot be read, all checks are kept."""
    if not checks:
        return checks, {"raw": "", "dropped": []}
    listing = "\n".join(f"{c['id']}: {c['description']}" for c in checks)
    raw = llm.complete(SUPPORT_PROMPT.format(requirement=requirement, checks=listing),
                       max_tokens=300, tag="precision.support_filter")
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        verdict = {str(k).strip(): str(v).strip().lower() for k, v in json.loads(m.group(0)).items()} if m else {}
    except Exception:
        verdict = {}
    if not verdict:
        return checks, {"raw": raw, "dropped": [], "unreadable": True}
    kept = [c for c in checks if verdict.get(c["id"], "yes").startswith("y")]
    if not any(c["kind"] == "meaning" for c in kept):     # never leave a test without a meaning check
        return checks, {"raw": raw, "dropped": [], "all_rejected": True}
    dropped = [c["id"] for c in checks if c not in kept]
    return [dict(c, id=f"C{i + 1}") for i, c in enumerate(kept)], {"raw": raw, "dropped": dropped}
