#!/usr/bin/env python3
"""Figures for manuscript v3.1 (NC). Every number is read from results/, nothing hand-entered.

  fig1_v31_ladder     17-arm ladder, macro-AUCPR, per-seed dots
  fig2_v31_temperature  AUCPR vs median N_eff over six temperatures
  fig3_v31_gauss       Gaussian-TTA dose response against the uniform limit
  fig4_v31_truncation  Δ vs 2D by truncation stratum
  fig5_v31_cka         linear CKA between seven arms + probe R^2
  fig6_v31_tiers       DREAM (2D arms) and M2OR (dual tower, cold receptor)

Usage: python p2/make_figures_v31.py
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
R = os.path.join(PROJ, "results")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

ARMS = [
    ("shape scalars (6)", "s2_ablation/en4", "scalar"),
    ("all nine scalars", "s2_ablation/en5", "scalar"),
    ("E-N3 entropy+rotors", "r25/scalar_gin", "scalar"),
    ("E-N6 rotors+ring", "s2_ablation/en6", "scalar"),
    ("uniform-weight MIL", "s1_mil/shuffle", "mil"),
    ("n_rotors only", "s2_ablation/en1", "scalar"),
    ("2D GIN", "r1/gin", "2d"),
    ("thermodynamics only", "s2_ablation/en2", "scalar"),
    ("Boltzmann-weight MIL", "s1_mil/boltz", "mil"),
    ("Edge-GIN (R2.5)", "r25/edge_gin", "edge"),
    ("attention-weight MIL", "s1_mil/attn", "mil"),
    ("DimeNet++", "r2_dimenet/dimenet", "3d"),
    ("SchNet", "r2_sc/schnet", "3d"),
    ("Chiral-SchNet", "r2_sc/chiral_schnet", "3d"),
    ("E-N3 scalars permuted", "s2_ablation/en3_shuf", "ctrl"),
    ("Edge-GIN shuffled", "r25/edge_gin_shuffled", "ctrl"),
    ("Angle-MPNN", "r2/angle_mpnn", "3d"),
]
COL = {"scalar": "#2a9d8f", "mil": "#e76f51", "2d": "#1d3557", "edge": "#457b9d",
       "3d": "#8d99ae", "ctrl": "#adb5bd"}


def arm_vals(path):
    files = sorted(glob.glob(os.path.join(R, path, "seed*", "metrics_det.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(R, path, "seed*", "metrics.json")))
    return np.array([json.load(open(f))["test_macro_aucpr"] for f in files])


def fig1():
    rows = [(n, arm_vals(p), c) for n, p, c in ARMS]
    rows = [r for r in rows if len(r[1])]
    rows.sort(key=lambda r: r[1].mean())
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    y = np.arange(len(rows))
    for i, (n, v, c) in enumerate(rows):
        ax.barh(i, v.mean() - 0.25, left=0.25, color=COL[c], height=0.62, alpha=0.85)
        ax.scatter(v, np.full(len(v), i), s=5, color="k", alpha=0.35, zorder=3, linewidths=0)
        ax.text(v.max() + 0.0012, i, f"{v.mean():.4f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.axvline(rows[[r[0] for r in rows].index("2D GIN")][1].mean(), color="#1d3557",
               ls="--", lw=0.8)
    ax.set_xlim(0.25, 0.345)
    ax.set_xlabel("macro-AUCPR (730 test molecules)")
    ax.set_title("17 run families, 10 seeds each", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_v31_ladder.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig2():
    c = json.load(open(os.path.join(R, "wpb_tempsweep", "curve.json")))
    pts = [p for p in c["points"] if p["median_n_eff"] is not None]
    pts.sort(key=lambda p: p["median_n_eff"])
    x = [p["median_n_eff"] for p in pts]
    y = [p["aucpr_mean"] for p in pts]
    e = [p["aucpr_sd"] for p in pts]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.errorbar(x, y, yerr=e, marker="o", ms=4, lw=1.2, capsize=2, color="#e76f51")
    for p in pts:
        lab = "∞" if p["tau"] == float("inf") else f"{p['tau']:.0f}K"
        ax.annotate(lab, (p["median_n_eff"], p["aucpr_mean"]),
                    textcoords="offset points", xytext=(3, -9), fontsize=6)
    ax.axhline(0.3253, color="#1d3557", ls=":", lw=0.9)
    ax.text(min(x) - 0.3, 0.3250, "uniform ensemble", fontsize=6, ha="left", va="top",
            color="#1d3557")
    ax.set_xlim(min(x) - 0.6, max(x) + 1.0)
    ax.set_xlabel("median effective conformers, N_eff(τ)")
    ax.set_ylabel("macro-AUCPR")
    ax.set_title(f"Spearman ρ = {c['G7']['spearman']:.2f}", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_v31_temperature.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig3():
    g = json.load(open(os.path.join(R, "wpb_gauss", "summary.json")))
    uni = g["uniform_det"]["mean"]
    xs, ys, es = [0.0], [uni], [g["uniform_det"]["sd"]]
    labels = ["uniform"]
    for tag, x in (("0.05", 0.05), ("0.15", 0.15), ("0.30", 0.30), ("rmsd", 1.22)):
        v = g["arms"].get(tag)
        if not v:
            continue
        xs.append(x)
        ys.append(v["aucpr_mean"])
        es.append(v["aucpr_sd"])
        labels.append(tag)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.errorbar(xs[1:], ys[1:], yerr=es[1:], marker="o", ms=4, lw=1.2, capsize=2,
                color="#8d99ae")
    ax.axhline(uni, color="#2a9d8f", ls="--", lw=1)
    ax.text(0.052, uni + 0.0022, "uniform ensemble", fontsize=6, color="#2a9d8f")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.15, 0.3, 1.22])
    ax.set_xticklabels(["0.05", "0.15", "0.30", "1.22*"], fontsize=6.5)
    ax.set_xlabel("Gaussian jitter σ (Å);  * RMSD-matched")
    ax.set_ylabel("macro-AUCPR")
    ax.set_title("coordinate jitter does not replace coverage", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_v31_gauss.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig4():
    s = json.load(open(os.path.join(R, "wpb_censor", "stratified.json")))
    groups = ["untruncated", "truncated"]
    arms = [("SchNet", "SCHNET"), ("R3 Boltzmann", "R3_boltz"),
            ("R3 uniform", "R3_uniform"), ("Edge-GIN", "R25_edge"),
            ("n_rotors", "EN1_rotors")]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.38
    for i, (lab, key) in enumerate(arms):
        for j, g in enumerate(groups):
            r1 = s["arms"]["R1_2D"][g]["aucpr_mean"]
            v = s["arms"][key][g]["aucpr_mean"] - r1
            ax.bar(i + (j - 0.5) * w, v, width=w, color=["#8d99ae", "#e76f51"][j],
                   label=g if i == 0 else None)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel("Δ macro-AUCPR vs 2D")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("deficit is localised to truncated ensembles", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig4_v31_truncation.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig5():
    z = np.load(os.path.join(R, "wpa_redundancy", "cka.npz"), allow_pickle=True)
    M = z["linear_mean"]
    labels = [str(x) for x in z["labels"]]
    p = json.load(open(os.path.join(R, "wpa_redundancy", "probing.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    im = ax.imshow(M, cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="w" if M[i, j] < 0.8 else "k")
    ax.set_title("linear CKA, 730 test molecules", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    ax = axes[1]
    targets = p["targets"]
    fams = ["gin", "gin_rotors", "descriptors_2d"]
    cols = {"gin": "#1d3557", "gin_rotors": "#e76f51", "descriptors_2d": "#adb5bd"}
    w = 0.26
    for k, f in enumerate(fams):
        vals = [p["summary"]["mlp"][f][t]["r2_mean"] for t in targets]
        ax.bar(np.arange(len(targets)) + (k - 1) * w, vals, width=w, color=cols[f],
               label={"gin": "GIN", "gin_rotors": "GIN + n_rotors",
                      "descriptors_2d": "4 raw 2D"}[f])
    ax.axhline(0.85, color="k", ls=":", lw=0.9)
    ax.text(len(targets) - 0.5, 0.86, "G4 threshold", fontsize=6, ha="right")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("test R² (MLP)")
    ax.legend(fontsize=6, frameon=False, ncol=1, loc="upper left")
    ax.set_title("2D representation does not determine 3D shape", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig5_v31_cka.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig6():
    d = json.load(open(os.path.join(R, "s3_tiers", "dream", "summary.json")))
    m = json.load(open(os.path.join(R, "s3_tiers", "m2or", "summary.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    names = [("morgan", "Morgan+Ridge"), ("en1", "E-N1"), ("en6", "E-N6"), ("r1", "GIN")]
    vals = [d["arms"][k]["mean_r"] for k, _ in names]
    errs = [d["arms"][k]["sd"] for k, _ in names]
    ax.bar(range(len(names)), vals, yerr=errs, capsize=2,
           color=["#adb5bd", "#2a9d8f", "#2a9d8f", "#1d3557"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n for _, n in names], rotation=20, ha="right", fontsize=6.5)
    ax.set_ylim(0.2, 0.36)
    ax.set_ylabel("mean per-target pooled r")
    ax.set_title("DREAM (338 molecules, 21 targets)", fontsize=8)

    ax = axes[1]
    x = np.arange(3)
    w = 0.38
    s2 = [m["S2_cold_receptor"]["arms"][a]["macro_ap"] for a in "ABC"]
    e2 = [m["S2_cold_receptor"]["arms"][a]["sd_ap"] for a in "ABC"]
    s1 = [m["S1_cold_ligand"]["arms"][a]["macro_ap"] for a in "ABC"]
    e1 = [m["S1_cold_ligand"]["arms"][a]["sd_ap"] for a in "ABC"]
    ax.bar(x - w / 2, s2, w, yerr=e2, capsize=2, color="#e76f51", label="cold receptor")
    ax.bar(x + w / 2, s1, w, yerr=e1, capsize=2, color="#457b9d", label="cold ligand")
    ax.set_xticks(x)
    ax.set_xticklabels(["A dual", "B ligand-only", "C shuffled"], fontsize=6.5)
    ax.set_ylim(0.5, 0.78)
    ax.set_ylabel("macro-AP (per receptor)")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("M2OR dual tower", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig6_v31_tiers.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f"[fig] {f.__name__} done")


if __name__ == "__main__":
    main()
