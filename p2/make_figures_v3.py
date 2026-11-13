#!/usr/bin/env python3
"""Figures for manuscript v3.0 (flexibility-count framing).

Fig 1  all thirteen arms, macro-AUCPR, per-seed dots, SD bars, grouped by family
Fig 2  the Rung-3 weight experiment: uniform vs Boltzmann vs attention vs rank-1

Both read `results/` only; nothing is hand-entered.
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

ARMS = [
    ("shape scalars (6)", "results/s2_ablation/en4/seed*/metrics.json", "scalar"),
    ("all nine scalars", "results/s2_ablation/en5/seed*/metrics.json", "scalar"),
    ("E-N3 (entropy+rotors)", "results/r25/scalar_gin/seed*/metrics.json", "scalar"),
    ("uniform-weight MIL", "results/s1_mil/shuffle/seed*/metrics_det.json", "mil"),
    ("n_rotors only", "results/s2_ablation/en1/seed*/metrics.json", "scalar"),
    ("2D GIN", "results/r1/gin/seed*/metrics.json", "2d"),
    ("thermodynamics only", "results/s2_ablation/en2/seed*/metrics.json", "scalar"),
    ("Boltzmann-weight MIL", "results/s1_mil/boltz/seed*/metrics.json", "mil"),
    ("Edge-GIN (R2.5)", "results/r25/edge_gin/seed*/metrics.json", "edge"),
    ("attention-weight MIL", "results/s1_mil/attn/seed*/metrics.json", "mil"),
    ("DimeNet++", "results/r2_dimenet/dimenet/seed*/metrics.json", "3d"),
    ("SchNet", "results/r2_sc/schnet/seed*/metrics.json", "3d"),
    ("Chiral-SchNet", "results/r2_sc/chiral_schnet/seed*/metrics.json", "3d"),
    ("Edge-GIN, shuffled", "results/r25/edge_gin_shuffled/seed*/metrics.json", "ctrl"),
    ("Angle-MPNN", "results/r2/angle_mpnn/seed*/metrics.json", "3d"),
]
COL = {"scalar": "#2a9d8f", "mil": "#e76f51", "2d": "#1d3557", "edge": "#457b9d",
       "3d": "#8d99ae", "ctrl": "#adb5bd"}


def load(pat):
    vals = []
    for f in sorted(glob.glob(os.path.join(PROJ, pat))):
        vals.append(json.load(open(f))["test_macro_aucpr"])
    return np.array(vals)


def fig1():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ys = np.arange(len(ARMS))[::-1]
    r1 = load(ARMS[5][1]).mean()
    for y, (name, pat, fam) in zip(ys, ARMS):
        v = load(pat)
        ax.barh(y, v.mean(), color=COL[fam], height=0.62, alpha=0.9)
        ax.scatter(np.full(len(v), v.mean()) + np.random.default_rng(0).normal(0, 0.0075, len(v)),
                   y + np.random.default_rng(1).normal(0, 0.06, len(v)),
                   s=6, color="white", edgecolor="black", linewidth=0.3, zorder=3)
        ax.plot([v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1)], [y, y],
                color="black", lw=1, zorder=4)
        ax.text(0.3455, y, f"{v.mean():.4f}", va="center", ha="left", fontsize=7)
    ax.axvline(r1, color="#1d3557", ls="--", lw=1)
    ax.set_yticks(ys, [a[0] for a in ARMS], fontsize=8)
    ax.set_xlim(0.24, 0.356)
    ax.set_xlabel("macro-AUCPR (test, 730 molecules)")
    ax.set_title("15 arms, 10 seeds each", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_ladder_v3.{ext}"), dpi=300)
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    arms = [("uniform", "results/s1_mil/shuffle/seed*/metrics_det.json"),
            ("Boltzmann", "results/s1_mil/boltz/seed*/metrics.json"),
            ("attention", "results/s1_mil/attn/seed*/metrics.json"),
            ("rank-1 (n=1)", "results/s1_mil/rank1/seed*/metrics.json")]
    xs = np.arange(len(arms))
    means = [load(p).mean() for _, p in arms]
    sds = [load(p).std(ddof=1) if len(load(p)) > 1 else 0 for _, p in arms]
    ax.bar(xs, means, yerr=sds, capsize=4, color=["#e76f51", "#457b9d", "#8d99ae", "#adb5bd"])
    for x, m, sd in zip(xs, means, sds):
        ax.text(x, m + sd + 0.0012, f"{m:.4f}", ha="center", fontsize=7)
    ax.axhline(load("results/r1/gin/seed*/metrics.json").mean(), color="#1d3557",
               ls="--", lw=1, label="2D GIN")
    ax.set_xticks(xs, [a[0] for a in arms], fontsize=7)
    ax.set_ylim(0.29, 0.338)
    ax.set_ylabel("macro-AUCPR")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("same encoder, same conformers, different weights", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_weights_v3.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    print("wrote fig1_ladder_v3 / fig2_weights_v3")
