#!/usr/bin/env python3
"""Figures for the Paper-2 bounding-result manuscript (reads results only).

Fig 1  the dimensionality ladder: macro-AUCPR per model, with per-seed dots.
Fig 2  the two controls: (a) paired differences with 95 % CI; (b) mirror
       sensitivity of the SchNet pair.

Outputs: reports/figures/fig1_ladder.{png,pdf}, fig2_controls.{png,pdf}
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "reports", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})

GROUPS = [
    ("E-N3  GIN + flexibility", "results/r25/scalar_gin/seed*/metrics.json", "#2f855a"),
    ("R1  GIN (2D)", "results/r1/gin/seed*/metrics.json", "#2b6cb0"),
    ("R2.5  Edge-GIN", "results/r25/edge_gin/seed*/metrics.json", "#805ad5"),
    ("R2  Chiral-SchNet", "results/r2_sc/chiral_schnet/seed*/metrics.json", "#b7791f"),
    ("R2  SchNet", "results/r2_sc/schnet/seed*/metrics.json", "#c05621"),
    ("R2.5  Edge shuffled", "results/r25/edge_gin_shuffled/seed*/metrics.json", "#a0aec0"),
    ("R2  Angle-MPNN", "results/r2/angle_mpnn/seed*/metrics.json", "#c53030"),
]


def load(pat):
    vals = []
    for f in sorted(glob.glob(os.path.join(PROJ, pat))):
        vals.append(json.load(open(f))["test_macro_aucpr"])
    return np.array(vals)


def fig1():
    labels, means, sds, allvals, colors = [], [], [], [], []
    for name, pat, c in GROUPS:
        v = load(pat)
        labels.append(name); means.append(v.mean()); sds.append(v.std(ddof=1))
        allvals.append(v); colors.append(c)

    order = np.argsort(means)  # ascending -> horizontal bars, best on top
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(labels))
    for i, k in enumerate(order):
        ax.barh(i, means[k], color=colors[k], height=0.62, alpha=0.9)
        ax.errorbar(means[k], i, xerr=sds[k], fmt="none", ecolor="k", elinewidth=1, capsize=2)
        ax.scatter(allvals[k], np.full(len(allvals[k]), i), s=10, color="k", zorder=3, alpha=0.6)
        ax.text(0.338, i, f"{means[k]:.4f}", va="center", ha="right", fontsize=7)

    ax.set_yticks(y, [labels[k] for k in order])
    ax.set_xlabel("macro-AUCPR (test, 730 molecules, 3 seeds)")
    ax.set_xlim(0.24, 0.34)
    ax.set_xticks([0.24, 0.26, 0.28, 0.30, 0.32])
    ax.set_title("Dimensionality ladder: no 3D rung exceeds the 2D baseline")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_ladder.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig2():
    boot = json.load(open(os.path.join(PROJ, "results", "d7_bootstrap.json")))["comparisons"]
    keys = ["edge_gin - schnet", "gin - edge_gin", "scalar_gin - edge_gin", "edge_gin - shuffled"]
    nice = ["R2.5 vs static 3D", "2D vs R2.5", "flexibility vs R2.5", "R2.5 vs shuffled"]
    d = [boot[k]["delta"] for k in keys]
    lo = [boot[k]["delta"] - boot[k]["ci_low"] for k in keys]
    hi = [boot[k]["ci_high"] - boot[k]["delta"] for k in keys]

    mir = json.load(open(os.path.join(PROJ, "results", "r2_sc", "mirror_test.json")))
    short = {"results/r2_sc/schnet/seed0": "SchNet",
             "results/r2_sc/chiral_schnet/seed0": "Chiral-SchNet"}
    names = [short.get(n, n.split("/")[2]) for n in mir]
    dp = [mir[n]["mean_abs_dp"] for n in mir]
    dp = [0.0 if v < 1e-6 else v for v in dp]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    y = np.arange(len(keys))
    ax.axvline(0, color="k", lw=0.8)
    ax.errorbar(d, y, xerr=[lo, hi], fmt="o", ms=4, capsize=2,
                color="#2d3748", ecolor="#2d3748", elinewidth=1)
    ax.set_yticks(y, nice)
    ax.set_xlabel("Δ macro-AUCPR (95 % CI)")
    ax.set_title("A  Only the shuffle control separates")
    for i, dd in enumerate(d):
        ax.text(0.0475, i, f"{dd:+.4f}", ha="right", va="center", fontsize=6.5)
    ax.set_xlim(-0.03, 0.05)

    ax = axes[1]
    ax.bar(range(len(names)), dp, color=["#c05621", "#2f855a"][:len(names)], width=0.55)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("mean |Δp| under mirror reflection")
    ax.set_title("B  Chirality is representable")
    ax.set_ylim(0, max(max(dp) * 1.45, 2e-3))
    for i, v in enumerate(dp):
        ax.text(i, v, f"{v:.2e}" if v else "0 (exact)", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_controls.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("[done]", OUT)
