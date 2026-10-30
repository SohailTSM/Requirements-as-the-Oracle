#!/usr/bin/env python3
"""Report tables straight from the result files (no hand-copied numbers).

    .venv/bin/python scripts/make_tables.py        # writes results/tables/*.md and *.tex

Tables
  main.*       finals: tool x wording, per service and pooled, mean ± sd over runs
               (score = % of planted bugs caught by tests generated on the correct service;
               "mutant-gen" = same for tests generated against mutants; "fail-OK" = share of
               generated tests that fail on the correct service)
  sweep.*      component sweep (FastAPI, vague): each variant against the default configuration
  retrieval.*  offline retrieval (non-auth operations)
  precision.*  offline precision layer (prompt version folder chosen by --prompt-version)
Runs split into per-service parts (exp_id ...__<service>) are merged, as in analyze_results.py.
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.paths import RESULTS_ROOT  # noqa: E402
from common.wandb_log import summarize  # noqa: E402

SUFFIX = re.compile(r"(__(fastapi|nestjs-realworld|todoapp))?(__part\d+)?(__r15)?$")   # parts of one run (D-51, D-85, D-112)
N_REQ = {"fastapi": 63, "nestjs-realworld": 29, "todoapp": 14}   # a service counts only when every requirement is scored (D-112)
SERVICES = ["fastapi", "nestjs-realworld", "todoapp"]
SHORT = {"fastapi": "FastAPI", "nestjs-realworld": "NestJS", "todoapp": "TodoApp"}
TOOL_NAMES = {"b1_single_step": "B1 single-step", "b2_refinement": "B2 refinement",
              "b3_prompt_fixed": "B3 prompt-fixed", "b4_response_judge": "B4 response judge",
              "e_fuzzer": "E fuzzer", "ours": "Ours"}


def runs(group: str):
    """{run_id: (meta, [dirs])} for finished runs of a group."""
    out = defaultdict(lambda: [None, []])
    for m in sorted((RESULTS_ROOT / group).glob("*/run_meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("status") != "ok":
            continue
        rid = SUFFIX.sub("", meta["exp_id"])
        out[rid][0] = out[rid][0] or meta
        out[rid][1].append(m.parent)
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f}"


def write(name: str, header: list, rows: list, caption: str):
    out = RESULTS_ROOT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"<!-- {caption} -->", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
    tex = [r"\begin{table}[t]\centering\small", rf"\caption{{{esc(caption)}}}",
           r"\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", rf"\label{{tab:{name}}}", r"\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"\n{caption}\n" + "\n".join(md[1:]))


def main_table(groups):
    cells = defaultdict(lambda: defaultdict(list))      # (tool, variant) -> metric -> values
    for g in groups:
        for rid, (meta, dirs) in runs(g).items():
            s = summarize(dirs)
            key = (meta["tool"] if meta["tool"] != "ours" else f"ours:{(meta.get('agent_config') or {}).get('name', g)}",
                   meta["variant"])
            complete = {svc for svc, v in s["per_service"].items() if v["_n"] == N_REQ[svc]}
            for svc in SERVICES:
                if svc in complete:
                    cells[key][svc].append(s["per_service"][svc]["valid_pct"])
                elif svc in s["per_service"]:
                    print(f"  incomplete, left out: {g}/{rid} {svc} {s['per_service'][svc]['_n']}/{N_REQ[svc]}")
            if complete == set(SERVICES):          # pooled only over complete runs
                cells[key]["pooled"].append(s["pooled"]["valid_pct"])
                cells[key]["mutgen"].append(s["pooled"]["mutated_pct"])
                cells[key]["failok"].append(s["pooled"]["valid_tests_failing_pct"])
    rows = []
    for (tool, variant), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append([TOOL_NAMES.get(tool, tool), variant] + [fmt(c[s]) for s in SERVICES]
                    + [fmt(c["pooled"]), fmt(c["mutgen"]), fmt(c["failok"]), max(len(c[s]) for s in SERVICES)])
    write("main", ["Tool", "Wording"] + [SHORT[s] for s in SERVICES] + ["All", "Mutant-gen", "Fail-OK", "Runs"],
          rows, "Planted bugs caught (%), local Llama 3.1 8B (Q4_K_M); mean ± sd over runs")


def sweep_table():
    rs = runs("sweep")
    if not rs:
        print("\n(no sweep runs yet)")
        return
    base = None
    rows = []
    defaults = []
    for rid, (meta, dirs) in sorted(rs.items()):
        summ = summarize(dirs)
        n = summ["per_service"].get("fastapi", {}).get("_n", 0)
        if n != N_REQ["fastapi"]:
            print(f"  incomplete, left out: sweep/{rid} {n}/{N_REQ['fastapi']}")
            continue
        s = summ["pooled"]
        name = rid.replace("ours__", "").replace("sweep__", "")
        rows.append([name, meta["variant"], s["valid_pct"], s["mutated_pct"], s["valid_tests_failing_pct"]])
        if rid.startswith("ours__default"):
            defaults.append(s["valid_pct"])
    base = st.mean(defaults) if defaults else None          # mean of the default's repeats (D-112)
    out = [[r[0], r[1], f"{r[2]:.1f}", "–" if base is None else f"{r[2] - base:+.1f}",
            f"{r[3]:.1f}" if r[3] is not None else "–", f"{r[4]:.1f}"] for r in rows]
    write("sweep", ["Variant", "Wording", "Caught %", "Δ vs default", "Mutant-gen %", "Fail-OK %"], out,
          "Component sweep on FastAPI (vague), one run per variant")


def retrieval_table():
    f = RESULTS_ROOT / "offline/retrieval/summary_non_auth.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = [[k.split("/")[1], k.split("/")[0], f"{v['R@5']:.3f}", f"{v['C@5']:.3f}", f"{v['MRR']:.3f}", f"{v['nDCG@5']:.3f}"]
            for k, v in sorted(d.items(), key=lambda kv: (kv[0].split("/")[0], kv[0]))]
    write("retrieval", ["Variant", "Wording", "R@5", "All needed in top 5", "MRR", "nDCG@5"], rows,
          "Offline retrieval over 106 requirements (operations other than login/registration)")


def precision_table(version):
    f = RESULTS_ROOT / f"offline/precision/{version}/summary.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = [[k.split("/")[0], k.split("/")[1], f"{v['check_recall']:.3f}", f"{v['check_recall_soft']:.3f}",
             f"{v['contradiction_rate']:.3f}", f"{v['unsupported_rate']:.3f}", f"{v['n_checks']:.1f}"]
            for k, v in sorted(d.items())]
    write("precision", ["Variant", "Wording", "Recall", "Soft recall", "Contradiction", "Unsupported", "Checks/req"],
          rows, f"Offline precision layer, prompt {version}, scored by the Llama judge (to be validated by human labels)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["baselines", "finals"])
    ap.add_argument("--prompt-version", default="v2")
    a = ap.parse_args()
    main_table(a.groups)
    sweep_table()
    retrieval_table()
    precision_table(a.prompt_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
