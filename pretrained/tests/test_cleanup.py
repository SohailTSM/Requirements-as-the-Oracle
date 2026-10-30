"""Regression tests for the spec-only clean-up and parsing (Decision Log D-40, D-47, D-48, D-52, D-59).

    .venv/bin/python -m pytest tests -q
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.compiler import infer_auth  # noqa: E402
from agent.lint import (autofill_path_params, define_missing_actors, drop_impossible, fix_misdirected_refs,  # noqa: E402
                        generalize_unstated_status, lint_plan, remove_redundant_signup)
from agent.planner import parse_plan  # noqa: E402
from agent.spec import load_operations  # noqa: E402
from common.paths import HARNESS_ROOT  # noqa: E402
from tools_ext.pretrained_tool import _spec_cleanup  # noqa: E402

OPS = load_operations("fastapi")
AUTH = infer_auth("fastapi", OPS)
REFS = json.loads((ROOT / "gold/reference_plans/fastapi.json").read_text())


def precise(rid):
    return json.loads((HARNESS_ROOT / f"data/requirements/fastapi/requirement_{rid}.json").read_text())["requirement_precise"]


@pytest.mark.parametrize("rid", sorted(REFS))
def test_reference_plans_are_untouched(rid):
    plan = REFS[rid].get("plan", REFS[rid])
    assert lint_plan(copy.deepcopy(plan), OPS) == []
    cleaned, events = _spec_cleanup(copy.deepcopy(plan), OPS, {}, precise(rid), AUTH)
    assert events == [] and cleaned == plan


def item_plan(checks, steps=None, actors=None):
    return {"actors": actors or {"A": {"type": "new_user"}},
            "steps": steps or [{"id": "s1", "actor": "A", "call": "POST /api/v1/items/", "body": {"title": "t"}, "role": "setup"},
                               {"id": "s2", "actor": "A", "call": "GET /api/v1/items/{id}", "path": {"id": "{{s1.id}}"}, "role": "verify"}],
            "checks": checks}


def test_field_not_in_list_wrapper_is_dropped():
    p = item_plan([{"id": "C1", "step": "s1", "expect": "field_exists", "field": "id", "kind": "meaning"}],
                  steps=[{"id": "s1", "actor": "A", "call": "GET /api/v1/items/", "role": "verify"}])
    _, ev = drop_impossible(p, OPS)
    assert ev and ev[0]["type"] == "impossible_check_dropped"


def test_id_equal_by_construction_is_dropped():
    p = item_plan([{"id": "C1", "step": "s2", "expect": "field_not_equals", "field": "id", "value": "{{s1.id}}", "kind": "meaning"}])
    _, ev = drop_impossible(p, OPS)
    assert [e["type"] for e in ev] == ["impossible_check_dropped"]


def test_undocumented_format_status_is_dropped():
    p = item_plan([{"id": "C1", "step": "s1", "expect": "status", "value": 201, "kind": "format"}])
    _, ev = drop_impossible(p, OPS)
    assert ev[0]["type"] == "undocumented_status_format_check_dropped"


def test_error_detail_field_is_allowed():
    p = item_plan([{"id": "C1", "step": "s2", "expect": "status", "value": 404, "kind": "meaning"},
                   {"id": "C2", "step": "s2", "expect": "field_equals", "field": "detail", "value": "Item not found", "kind": "meaning"}])
    _, ev = drop_impossible(p, OPS)
    assert ev == []


def test_missing_path_param_is_filled_from_creating_step():
    p = item_plan([], steps=[{"id": "s1", "actor": "A", "call": "POST /api/v1/items/", "body": {"title": "t"}, "role": "setup"},
                             {"id": "s2", "actor": "A", "call": "GET /api/v1/items/", "role": "verify"},
                             {"id": "s3", "actor": "A", "call": "GET /api/v1/items/{id}", "role": "verify"}])
    p, ev = autofill_path_params(p, OPS)
    assert p["steps"][2]["path"] == {"id": "{{s1.id}}"}


def test_resp_prefix_and_req_mixups_are_redirected():
    p = item_plan([{"id": "C1", "step": "s2", "expect": "field_equals", "field": "id", "value": "{{s1.req.id}}", "kind": "meaning"}],
                  steps=[{"id": "s1", "actor": "A", "call": "POST /api/v1/items/", "body": {"title": "t"}, "role": "setup"},
                         {"id": "s2", "actor": "A", "call": "GET /api/v1/items/{id}", "path": {"id": "{{s1.resp.id}}"}, "role": "verify"}])
    p, ev = fix_misdirected_refs(p, OPS)
    assert p["steps"][1]["path"]["id"] == "{{s1.id}}" and p["checks"][0]["value"] == "{{s1.id}}"


def test_invented_credentials_become_new_user():
    p = item_plan([], actors={"B": {"type": "existing", "email": "bob@example.com", "password": "x"}})
    p, ev = define_missing_actors(p, "A second user cannot read another user's item.")
    assert p["actors"]["B"] == {"type": "new_user"} and p["actors"]["A"] == {"type": "new_user"}


def test_quoted_credentials_are_kept():
    p = item_plan([], actors={"A": {"type": "existing", "email": "admin@example.com", "password": "password123"}})
    p, _ = define_missing_actors(p, "Confirm deleting an item as admin (admin@example.com / password123) works.")
    assert p["actors"]["A"]["type"] == "existing"


def test_unstated_exact_status_becomes_class_but_stated_is_kept():
    p = item_plan([{"id": "C1", "step": "s2", "expect": "status", "value": 404, "kind": "meaning"}])
    p, ev = generalize_unstated_status(copy.deepcopy(p), "A user cannot read another user's item.")
    assert p["checks"][0] == {"id": "C1", "step": "s2", "expect": "status_class", "value": "4xx", "kind": "meaning"}
    q, ev = generalize_unstated_status(item_plan([{"id": "C1", "step": "s2", "expect": "status", "value": 404, "kind": "meaning"}]),
                                       "Reading a missing item returns 404.")
    assert ev == []


def test_redundant_signup_is_removed_and_refs_point_to_actor():
    p = item_plan([], steps=[{"id": "s1", "actor": "A", "call": "POST /api/v1/users/signup", "body": {"email": "x@e.com"}, "role": "setup"},
                             {"id": "s2", "actor": "A", "call": "GET /api/v1/users/{user_id}", "path": {"user_id": "{{s1.id}}"}, "role": "action"}])
    p, ev = remove_redundant_signup(p, "Confirm a registered user cannot retrieve another user by id.", AUTH)
    assert [s["id"] for s in p["steps"]] == ["s2"] and p["steps"][0]["path"]["user_id"] == "{{A.id}}"
    q, ev = remove_redundant_signup(item_plan([], steps=[{"id": "s1", "actor": "A", "call": "POST /api/v1/users/signup"}]),
                                    "Signing up with a new email works.", AUTH)
    assert ev == []


def test_parser_survives_broken_placeholder_and_missing_brace():
    raw = '{"actors": {}, "steps": [{"id": "s1", "call": "GET /api/v1/items/{id}", "path": {"id": "{{s0.id}"}}], "checks": []'
    plan = parse_plan(raw)
    assert plan is not None and plan["steps"][0]["path"]["id"] == "{{s0.id}}"
