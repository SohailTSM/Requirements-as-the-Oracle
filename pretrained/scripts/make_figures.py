#!/usr/bin/env python3
"""Figures for the final report, straight from the result files (D-128).

    .venv/bin/python scripts/make_figures.py      # writes ../Reports/Final/figures/*.pdf

adaptation.pdf  caught % on tests generated against the correct vs the buggy service, vague and precise
ablations.pdf   ablation effects (final minus one component) against the final's run-to-run range
Palette: validated categorical slots 1-2 (blue, orange) from the dataviz reference palette.
"""
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from common.wandb_log import summarize  # noqa: E402
from final_stats import collect  # noqa: E402
from make_tables import N_REQ, SERVICES, TOOL_NAMES, runs  # noqa: E402

OUT = ROOT.parent / "Reports" / "Final" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

SHORT = {"b2_refinement": "B2", "b3_prompt_fixed": "B3", "ours:final": "Ours\nfinal",
         "ours:runner_content": "Ours\n−SC", "ours:runner_selfcons": "Ours\n−trace",
         "ours:unrestricted": "Ours\nunrestr."}


def adaptation(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, wording in zip(axes, ("vague", "precise")):
        tools = [t for t in SHORT if (t, wording) in data]
        w = 0.36
        for i, t in enumerate(tools):
            rs = data[(t, wording)]["runs"]
            for j, (field, color) in enumerate((("valid", BLUE), ("mut", ORANGE))):
                vals = [r[field] for r in rs]
                m = st.mean(vals)
                x = i + (j - 0.5) * (w + 0.04)
                ax.bar(x, m, w, color=color, edgecolor=SURF, linewidth=1, zorder=2)
                if len(vals) > 1:
                    ax.errorbar(x, m, yerr=st.stdev(vals), color=INK2, lw=0.8, capsize=1.5, zorder=3)
                ax.text(x, m + (st.stdev(vals) if len(vals) > 1 else 0) + 0.6, f"{m:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK)
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels([SHORT[t] for t in tools], fontsize=7)
        ax.set_title(f"{wording.capitalize()} requirements", fontsize=9, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Planted bugs caught (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["test refined on the correct service (caught)", "test refined on the buggy service (mutant-gen)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "adaptation.pdf")


def ablations():
    final = [summarize(d)["pooled"]["valid_pct"] for rid, (m, d) in runs("finals").items()
             if (m.get("agent_config") or {}).get("name") == "final" and m["variant"] == "vague"]
    base = st.mean(final)
    rows = []
    for rid, (meta, dirs) in runs("ablations").items():
        s = summarize(dirs)
        if any(s["per_service"].get(svc, {}).get("_n") != N_REQ[svc] for svc in SERVICES):
            continue
        nm = (meta.get("agent_config") or {}).get("name", rid).replace("ablation_", "")
        rows.append((nm, s["pooled"]["valid_pct"] - base))
    label = {"no_executor": "no executor (raw pytest)", "no_spec_cleanup": "no spec clean-up / lint",
             "no_precision": "no precision layer", "no_retrieval": "no retrieval (full spec)",
             "no_format_relaxation": "no format relaxation", "no_judge": "no judge",
             "unrestricted_refinement": "unrestricted refinement"}
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    lo, hi = min(final) - base, max(final) - base
    ax.axvspan(lo, hi, color=GRID, zorder=0, lw=0)
    for i, (nm, d) in enumerate(rows):
        ax.barh(i, d, 0.55, color=BLUE, zorder=2)
        ax.text(d + (0.4 if d >= 0 else -0.4), i, f"{d:+.1f}", va="center", ha="left" if d >= 0 else "right", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label.get(n, n) for n, _ in rows], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Δ caught % vs final (vague, 1 run each);\ngrey band = range of the final's 3 runs", fontsize=7)
    ax.set_xlim(-17, 11)
    ax.set_ylim(-0.6, len(rows) + 0.4)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.pdf")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    adaptation(collect())
    ablations()
    print("figures in", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
