#!/usr/bin/env python3
"""Statistics for the final report (Decision Log D-128). No hand-copied numbers.

    .venv/bin/python scripts/final_stats.py      # writes results/tables/stats.md and results/final_stats.json

1. Per-requirement paired tests: ours (final) against every baseline and every runner-up, per wording.
   Unit = requirement (n = 106). Per requirement and tool, the score is the share of that requirement's
   planted bugs caught by the test generated on the correct service, averaged over the tool's runs.
   Wilcoxon signed-rank (zero differences split), Holm-corrected within a wording, rank-biserial r,
   and a bootstrap 95% CI (requirements resampled) of the pooled caught-% difference.
2. Adaptation gap per tool: caught % minus mutant-gen % (tests generated against the buggy service),
   mean ± sd over runs, plus a per-requirement Wilcoxon test of caught vs mutant-gen within the tool.
3. Gap closed on vague wording: (ours - B1 vague) / (best precise score - B1 vague).
4. Step-level metrics of the final configuration from its agent histories: planner failures, lint and
   spec clean-up, setup errors, judge verdicts on the correct service vs on mutants, refinement guard
   events (tampering attempts blocked), value-copy rates of refinement baselines, cost per test.
5. Failure taxonomy of the final configuration on the correct service (vague and precise).
6. Precision layer judged by humans on the labelled sample, next to the Llama judge on the same items.
"""
import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from common.paths import RESULTS_ROOT  # noqa: E402
from common.wandb_log import summarize  # noqa: E402
from make_tables import N_REQ, SERVICES, TOOL_NAMES, runs  # noqa: E402
from analyze_results import refinement_signals  # noqa: E402

HV = ROOT / "gold" / "human_validation"
rng = random.Random(0)


def name(key):
    return TOOL_NAMES.get(key, key)


# ------------------------------------------------------------------ per-requirement data
def collect(groups=("baselines", "finals")):
    """{(tool, wording): {"runs": [...], "valid": {req: [frac per run]}, "mut": {...}, "mutants": {req: n}}}"""
    data = defaultdict(lambda: {"runs": [], "valid": defaultdict(list), "mut": defaultdict(list),
                                "pass": defaultdict(list), "mutants": {}, "dirs": []})
    for g in groups:
        for rid, (meta, dirs) in runs(g).items():
            s = summarize(dirs)
            if any(s["per_service"].get(svc, {}).get("_n") != N_REQ[svc] for svc in SERVICES):
                continue                                   # only complete runs (all three services)
            tool = meta["tool"] if meta["tool"] != "ours" else f"ours:{(meta.get('agent_config') or {}).get('name', g)}"
            d = data[(tool, meta["variant"])]
            d["runs"].append({"valid": s["pooled"]["valid_pct"], "mut": s["pooled"]["mutated_pct"],
                              "failok": s["pooled"]["valid_tests_failing_pct"]})
            d["dirs"] += dirs
            for svc, req, nm, passes, score, mavg in s["rows"]:
                k = (svc, req)
                d["mutants"][k] = nm
                d["valid"][k].append((score or 0) / nm if nm else 0.0)
                d["pass"][k].append(bool(passes))
                if mavg is not None and s["pooled"]["mutated_pct"] is not None:   # tools that never ran on mutants have none
                    d["mut"][k].append(mavg / nm if nm else 0.0)
    return data


def per_req(d, field="valid"):
    return {k: st.mean(v) for k, v in d[field].items() if v}


def pooled_pct(scores, mutants, keys):
    return 100 * sum(scores[k] * mutants[k] for k in keys) / sum(mutants[k] for k in keys)


def compare(a, b, B=5000, field="valid"):
    """Paired comparison of tool a against tool b over shared requirements."""
    sa, sb = per_req(a, field), per_req(b, field)
    keys = sorted(set(sa) & set(sb))
    m = a["mutants"]
    diff = np.array([sa[k] - sb[k] for k in keys])
    nz = diff[diff != 0]
    if len(nz) == 0:
        p, r = 1.0, 0.0
    else:
        p = float(wilcoxon([sa[k] for k in keys], [sb[k] for k in keys], zero_method="zsplit").pvalue)
        ranks = st_rank(np.abs(nz))
        wpos, wneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        r = float((wpos - wneg) / (wpos + wneg))
    point = pooled_pct(sa, m, keys) - pooled_pct(sb, m, keys)
    boots = []
    for _ in range(B):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        boots.append(pooled_pct(sa, m, ks) - pooled_pct(sb, m, ks))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(keys), "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()),
            "diff_pct": round(point, 1), "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "p": p, "r_rb": round(r, 2)}


def st_rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ step-level metrics
def histories(dirs):
    for d in dirs:
        yield from d.glob("*/generated_tests/*/agent_history.json")


def step_metrics(dirs):
    c = Counter()
    static, events, setup_kinds = Counter(), Counter(), Counter()
    tokens, seconds, calls = [], [], []
    for f in histories(dirs):
        h = json.loads(f.read_text())
        on = "mutant" if "_mut_" in f.parent.name else "valid"
        c["generations"] += 1
        c[f"generations_{on}"] += 1
        if h.get("error"):
            c["planner_failed"] += 1
        c["plans_with_lint_issues"] += bool(h.get("lint_issues"))
        c["lint_issues"] += len(h.get("lint_issues") or [])
        c["lint_repair_calls"] += bool(h.get("lint_repair"))
        for e in h.get("static_events") or []:
            static[e.get("type")] += 1
        tokens.append(sum((x.get("prompt_tokens") or 0) + (x.get("completion_tokens") or 0) for x in h.get("llm_calls", [])))
        calls.append(len(h.get("llm_calls", [])))
        seconds.append(h.get("seconds") or 0)
        for r in h.get("rounds", []):
            c["rounds"] += 1
            rep = r.get("report") or {}
            if rep.get("setup_errors"):
                c["rounds_with_setup_error"] += 1
                for s in rep["setup_errors"]:
                    s = str(s)
                    setup_kinds["setup call rejected by the service" if "returned" in s or "failed" in s
                                else "reference to a missing field" if "has no field" in s else "other"] += 1
            for e in r.get("static_events") or []:
                static[e.get("type")] += 1
            j = r.get("judge")
            if j:
                c[f"judged_{on}"] += 1
                c[f"judged_{on}_api_bug"] += j.get("verdict") == "API_BUG"
            c[f"decision_{r.get('decision')}"] += 1
            for e in r.get("refine_events") or []:
                events[e.get("type")] += 1
                c["copied_value_attempts"] += bool(e.get("copied_value")) or e.get("type") in (
                    "value_copy_blocked", "new_check_with_copied_value_blocked")
    blocked = ("locked_check_restored", "new_check_with_copied_value_blocked", "value_copy_blocked",
               "field_repair_rejected_not_in_spec")
    c["tampering_attempts_blocked"] = sum(events[t] for t in blocked)
    c["refine_rounds"] = c["decision_refine"]
    return {"counts": dict(c), "static_events": dict(static), "refine_events": dict(events),
            "setup_error_kinds": dict(setup_kinds),
            "tokens_per_generation": round(st.mean(tokens)), "llm_calls_per_generation": round(st.mean(calls), 1),
            "seconds_per_generation": round(st.mean(seconds), 1), "total_tokens": sum(tokens)}


def _leaves(o, out):
    if isinstance(o, dict):
        for v in o.values():
            _leaves(v, out)
    elif isinstance(o, list):
        for v in o:
            _leaves(v, out)
    elif isinstance(o, bool) or o is None:
        return
    elif isinstance(o, (int, float)) or (isinstance(o, str) and 0 < len(o) <= 80):
        out.add(str(o))


def accepted_copies(dirs):
    """Our agent's rounds whose next executed plan holds a new or changed expected value that the
    service returned in that round (status codes and response values): copies that reached the test."""
    rounds = copies = 0
    for f in histories(dirs):
        rs = json.loads(f.read_text()).get("rounds", [])
        for cur, nxt in zip(rs, rs[1:]):
            rounds += 1
            seen = set()
            for stp in (cur.get("report") or {}).get("steps", []):
                if stp.get("status") is not None:
                    seen.add(str(stp["status"]))
                _leaves(stp.get("response"), seen)
            before = {(c.get("step"), c.get("expect"), c.get("field"), str(c.get("value")))
                      for c in (cur.get("plan") or {}).get("checks", [])}
            new = [c for c in (nxt.get("plan") or {}).get("checks", [])
                   if (c.get("step"), c.get("expect"), c.get("field"), str(c.get("value"))) not in before]
            copies += any(c.get("value") is not None and str(c.get("value")) in seen for c in new)
    return {"rounds": rounds, "copies": copies}


def taxonomy(dirs):
    """Outcome of each test generated on the correct service (one per requirement and run)."""
    out = Counter()
    for d in dirs:
        for svc_dir in d.iterdir():
            ev_f = svc_dir / "evaluation_results.json"
            if not ev_f.exists():
                continue
            ev = {r["requirement_id"]: r for r in json.loads(ev_f.read_text())["requirements"]}
            for req, r in ev.items():
                hf = svc_dir / "generated_tests" / f"req_{req}" / "agent_history.json"
                h = json.loads(hf.read_text()) if hf.exists() else {}
                vt = r.get("valid_test", {})
                last = (h.get("rounds") or [{}])[-1]
                if h.get("error") or not h:
                    out["no valid plan (planner failed)"] += 1
                elif vt.get("passes_on_valid"):
                    out["passes, catches ≥1 planted bug" if vt.get("score") else "passes, catches no planted bug (checks too weak)"] += 1
                elif (last.get("report") or {}).get("setup_errors"):
                    out["fails on correct service: setup error"] += 1
                elif last.get("decision") == "api_bug":
                    out["fails on correct service: wrong check, judge blamed the API"] += 1
                else:
                    out["fails on correct service: wrong check, rounds exhausted"] += 1
    return dict(out)


# ------------------------------------------------------------------ human-labelled precision layer
def human_precision():
    items = {i["id"]: i for i in json.loads((HV / "items.json").read_text())}
    llm = json.loads((HV / "llm_labels.json").read_text())
    res = {}
    llm_g, llm_d = Counter(), Counter()
    seen = set()
    for f in sorted((HV / "export" / "labels").glob("*.json")):
        labs = json.loads(f.read_text()).get("items", {})
        g, d = Counter(), Counter()
        for iid, lab in labs.items():
            it = items.get(iid)
            if not it or it["task"] == "gold":
                continue
            v = llm.get(iid, {})
            for ck in it["checks"]:
                if lab.get(ck["key"]):
                    g[lab[ck["key"]]] += 1
                    if (iid, ck["key"]) not in seen:
                        llm_g[v.get("gold", {}).get(ck["key"], "none")] += 1
                        seen.add((iid, ck["key"]))
            for dk in it.get("derived", []):
                if lab.get(dk["key"]):
                    d[lab[dk["key"]]] += 1
                    if (iid, dk["key"]) not in seen:
                        llm_d[v.get("derived", {}).get(dk["key"], "extra")] += 1
                        seen.add((iid, dk["key"]))
        res[f.stem] = rates(g, d)
    res["llm_judge_same_items"] = rates(llm_g, llm_d)
    return res


def rates(g, d):
    ng, nd = sum(g.values()) or 1, sum(d.values()) or 1
    return {"n_gold": sum(g.values()), "recall_full": round(g["full"] / ng, 3),
            "recall_soft": round((g["full"] + 0.5 * g["partial"]) / ng, 3),
            "n_derived": sum(d.values()), "match": round(d["match"] / nd, 3),
            "extra": round(d["extra"] / nd, 3), "contradiction": round(d["contradiction"] / nd, 3)}


# ------------------------------------------------------------------ main
def main() -> int:
    data = collect()
    out = {"comparisons": {}, "adaptation": {}, "steps": {}, "taxonomy": {}}
    md = ["# Final statistics (scripts/final_stats.py)", ""]
    ours = "ours:final"
    for wording in ("vague", "precise"):
        if (ours, wording) not in data:
            continue
        a = data[(ours, wording)]
        others = sorted(k for k in data if k[1] == wording and k[0] != ours)
        res = {k[0]: compare(a, data[k]) for k in others}
        adj = holm([r["p"] for r in res.values()])
        for (t, r), pa in zip(res.items(), adj):
            r["p_holm"] = pa
        out["comparisons"][wording] = res
        md += [f"## Ours (final) vs others, {wording} wording (n = requirements; Δ = ours − other, pooled caught %)", "",
               "| Other | Runs | Δ caught % | 95% CI | ours better / worse (req.) | Wilcoxon p (Holm) | r |", "|---|---|---|---|---|---|---|"]
        for t, r in res.items():
            md.append(f"| {name(t)} | {len(data[(t, wording)]['runs'])} | {r['diff_pct']:+.1f} | [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] | "
                      f"{r['better']} / {r['worse']} | {fmt_p(r['p'])} ({fmt_p(r['p_holm'])}) | {r['r_rb']:+.2f} |")
        md.append("")
        mres = {k[0]: compare(a, data[k], field="mut") for k in others if data[k]["mut"]}
        madj = holm([r["p"] for r in mres.values()])
        for r, pa in zip(mres.values(), madj):
            r["p_holm"] = pa
        out["comparisons"][wording + "_mutant_gen"] = mres
        md += [f"## Mutant-gen (tests generated against the buggy service), ours (final) vs others, {wording}", "",
               "| Other | Δ mutant-gen % | 95% CI | ours better / worse (req.) | Wilcoxon p (Holm) | r |", "|---|---|---|---|---|---|"]
        for t, r in mres.items():
            md.append(f"| {name(t)} | {r['diff_pct']:+.1f} | [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] | "
                      f"{r['better']} / {r['worse']} | {fmt_p(r['p'])} ({fmt_p(r['p_holm'])}) | {r['r_rb']:+.2f} |")
        md.append("")
    # adaptation gap
    md += ["## Adaptation gap (caught % − mutant-gen %), mean ± sd over runs", "",
           "| Tool | Wording | Caught | Mutant-gen | Gap | Wilcoxon p (caught vs mutant-gen, per req.) |", "|---|---|---|---|---|---|"]
    for (t, w), d in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rs = [r for r in d["runs"] if r["mut"] is not None]
        if not rs or not d["mut"]:
            continue
        gaps = [r["valid"] - r["mut"] for r in rs]
        sv, sm = per_req(d), per_req(d, "mut")
        keys = sorted(set(sv) & set(sm))
        diffs = [sv[k] - sm[k] for k in keys]
        p = float(wilcoxon([sv[k] for k in keys], [sm[k] for k in keys], zero_method="zsplit").pvalue) if any(diffs) else 1.0
        f2 = lambda xs: f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"
        out["adaptation"][f"{t}|{w}"] = {"gap_mean": round(st.mean(gaps), 1), "gaps": [round(g, 1) for g in gaps], "p": p}
        md.append(f"| {name(t)} | {w} | {f2([r['valid'] for r in rs])} | {f2([r['mut'] for r in rs])} | {f2(gaps)} | {fmt_p(p)} |")
    md.append("")
    # gap closed
    b1v = st.mean(r["valid"] for r in data[("b1_single_step", "vague")]["runs"])
    best_precise = max(st.mean(r["valid"] for r in d["runs"]) for (t, w), d in data.items() if w == "precise")
    oursv = st.mean(r["valid"] for r in data[(ours, "vague")]["runs"])
    out["gap_closed"] = {"b1_vague": round(b1v, 1), "best_precise": round(best_precise, 1), "ours_vague": round(oursv, 1),
                         "share": round((oursv - b1v) / (best_precise - b1v), 3)}
    md += [f"## Gap closed (vague): ({oursv:.1f} − {b1v:.1f}) / ({best_precise:.1f} − {b1v:.1f}) = "
           f"{100 * out['gap_closed']['share']:.0f}% of the way from B1 vague to the best precise score", ""]
    # value copying in refinement loops
    md += ["## Refinement safety: rounds that copy an observed value into an expectation", "",
           "| Tool | Wording | Refinement rounds | Rounds attempting a copied value | Rate | Copies in the next executed plan |", "|---|---|---|---|---|---|"]
    out["value_copy"] = {}
    for t in ("b2_refinement", "b3_prompt_fixed", "ours:final", "ours:unrestricted"):
        for w in ("vague", "precise"):
            if (t, w) not in data:
                continue
            sig = [refinement_signals(dd) for dd in data[(t, w)]["dirs"]]
            rounds = sum(s["refine_rounds"] for s in sig)
            copies = sum(s["value_copy_rounds"] for s in sig)
            if t.startswith("ours"):
                acc = accepted_copies(data[(t, w)]["dirs"])     # copies that reached the next executed plan
                copies = copies if t == "ours:final" else acc["copies"]
                accepted = f"{acc['copies']} of {acc['rounds']} ({100 * acc['copies'] / acc['rounds']:.1f}%)"
            else:
                accepted = "all (no guard)"
            out["value_copy"][f"{t}|{w}"] = {"rounds": rounds, "copies": copies, "accepted": accepted}
            md.append(f"| {name(t)} | {w} | {rounds} | {copies} | {100 * copies / rounds if rounds else 0:.1f}% | {accepted} |")
    md.append("")
    # step-level metrics and taxonomy for the final configuration
    for w in ("vague", "precise"):
        dirs = data[(ours, w)]["dirs"]
        sm = step_metrics(dirs)
        out["steps"][w] = sm
        out["taxonomy"][w] = taxonomy(dirs)
    for w in ("vague", "precise"):
        c = out["steps"][w]["counts"]
        md += [f"## Final configuration, intermediate steps ({w}, 3 runs, tests generated on correct and buggy services)", "",
               f"- generations: {c['generations']} ({c.get('generations_valid', 0)} on the correct service); planner failed: {c.get('planner_failed', 0)} "
               f"({100 * c.get('planner_failed', 0) / c['generations']:.1f}%)",
               f"- plans with lint issues: {c['plans_with_lint_issues']} ({100 * c['plans_with_lint_issues'] / c['generations']:.0f}%), "
               f"{c['lint_issues']} issues, LLM lint repair used on {c['lint_repair_calls']}",
               f"- spec clean-up events: {out['steps'][w]['static_events']}",
               f"- rounds: {c['rounds']}; rounds with a setup error: {c.get('rounds_with_setup_error', 0)} "
               f"({100 * c.get('rounds_with_setup_error', 0) / c['rounds']:.0f}%); kinds {out['steps'][w]['setup_error_kinds']}",
               f"- judge on the correct service: {c.get('judged_valid_api_bug', 0)} API_BUG of {c.get('judged_valid', 0)} verdicts "
               f"(every failure there is the test's fault, so these are wrong blames); on buggy services: "
               f"{c.get('judged_mutant_api_bug', 0)} of {c.get('judged_mutant', 0)}",
               f"- refinement rounds: {c['refine_rounds']}; guard events {out['steps'][w]['refine_events']}; "
               f"tampering attempts blocked: {c['tampering_attempts_blocked']}, accepted: 0",
               f"- cost per generated test: {out['steps'][w]['llm_calls_per_generation']} LLM calls, "
               f"{out['steps'][w]['tokens_per_generation']} tokens, {out['steps'][w]['seconds_per_generation']} s",
               "", f"Outcome of each test generated on the correct service ({w}): {out['taxonomy'][w]}", ""]
    # human-labelled precision layer
    try:
        hp = human_precision()
        out["precision_human"] = hp
        md += ["## Precision layer on the human-labelled sample (catalogue prompt v2, vague)", "",
               "| Labeller | Gold checks | Fully covered | Soft recall | Derived checks | Match | Extra | Contradiction |", "|---|---|---|---|---|---|---|---|"]
        for who, r in hp.items():
            md.append(f"| {who} | {r['n_gold']} | {r['recall_full']:.2f} | {r['recall_soft']:.2f} | {r['n_derived']} | "
                      f"{r['match']:.2f} | {r['extra']:.2f} | {r['contradiction']:.2f} |")
        md.append("")
    except FileNotFoundError:
        pass
    (RESULTS_ROOT / "final_stats.json").write_text(json.dumps(out, indent=1, default=float))
    (RESULTS_ROOT / "tables" / "stats.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
