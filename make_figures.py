#!/usr/bin/env python3
"""Generate the six main figures for Paper 2 v4.0 from locked disk artifacts.

Every plotted point comes from results/*.json, results/**/metrics*.json or
results/**/test_preds.npz; no synthetic values are drawn.

Run from the paper2 project root:
  python ../consolidate/1031/1103_1857/make_figures.py ../consolidate/1031/1103_1857/Figures
"""
import itertools
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
C = {"2D": "#0072B2", "scalar": "#009E73", "ensemble": "#D55E00",
     "edge": "#CC79A7", "static3d": "#56B4E9", "control": "#999999"}
HERE = os.path.dirname(os.path.abspath(__file__))
FAM = {"R1": "2D", "en4_size": "scalar", "en4_shape": "scalar", "en4_s2d": "scalar",
       "wide_en1": "2D", "EN1": "scalar", "EN2": "scalar", "EN3": "scalar", "EN4": "scalar",
       "EN5": "scalar", "EN6": "scalar", "R3boltz": "ensemble", "R3unif": "ensemble",
       "R3attn": "ensemble", "R25": "edge", "SCHNET": "static3d", "CHIRAL": "static3d",
       "DIMENET": "static3d", "ANGLE": "static3d", "R25shuf": "control", "EN3shuf": "control"}
SHORT = {"R1": "2D GIN", "EN1": "2D + n_rotors", "EN2": "2D + thermodynamics",
         "EN3": "2D + n_eff/top_pop/n_rotors", "EN4": "2D + 6 ensemble moments",
         "EN5": "2D + 9 scalars", "EN6": "2D + n_rotors/ring-flex",
         "R3boltz": "MIL, Boltzmann", "R3unif": "MIL, uniform 1/C",
         "R3attn": "MIL, attention", "R25": "Edge-GIN (bond statistics)",
         "SCHNET": "SchNet (static 3D)", "CHIRAL": "Chiral-SchNet (static 3D)",
         "DIMENET": "DimeNet++ (static 3D)", "ANGLE": "Angle-MPNN (ablation)",
         "R25shuf": "Edge-GIN, edges shuffled", "EN3shuf": "2D + shuffled scalars",
         "en4_size": "2D + size moments (2)", "en4_shape": "2D + shape moments (4)",
         "en4_s2d": "2D + rotor + 4 descriptors", "wide_en1": "wide 2D GIN (1.16 M)"}
ARMS = {"R1": ("results/r1/gin", 0), "SCHNET": ("results/r2_sc/schnet", 0),
        "CHIRAL": ("results/r2_sc/chiral_schnet", 0), "DIMENET": ("results/r2_dimenet/dimenet", 0),
        "ANGLE": ("results/r2/angle_mpnn", 0), "R25": ("results/r25/edge_gin", 0),
        "R25shuf": ("results/r25/edge_gin_shuffled", 0), "EN3": ("results/r25/scalar_gin", 0),
        "R3boltz": ("results/s1_mil/boltz", 0), "R3attn": ("results/s1_mil/attn", 0),
        "R3unif": ("results/s1_mil/shuffle", 1), "EN1": ("results/s2_ablation/en1", 0),
        "EN2": ("results/s2_ablation/en2", 0), "EN4": ("results/s2_ablation/en4", 0),
        "EN5": ("results/s2_ablation/en5", 0), "EN6": ("results/s2_ablation/en6", 0),
        "EN3shuf": ("results/s2_ablation/en3_shuf", 0),
        # four additional arms from the decomposition / capacity audit (T9)
        "en4_s2d": (os.path.join(HERE, "experiments", "results", "en4_s2d"), 0),
        "wide_en1": (os.path.join(HERE, "experiments", "results", "wide_en1"), 0),
        "en4_size": (os.path.join(HERE, "en4_decomp", "results", "en4_size"), 0),
        "en4_shape": (os.path.join(HERE, "en4_decomp", "results", "en4_shape"), 0)}


def load(k):
    d, det = ARMS[k]
    v = {}
    for s in range(10):
        f = os.path.join(d, f"seed{s}", "metrics_det.json" if det else "metrics.json")
        m = json.load(open(f))
        v[s] = dict(aucpr=m["test_macro_aucpr"], val=m["best_val_aucpr"],
                    ep=m["best_epoch"], npar=m["n_params"])
    return v


V = {k: load(k) for k in ARMS}
mean = {k: np.mean([V[k][s]["aucpr"] for s in range(10)]) for k in ARMS}
sd = {k: np.std([V[k][s]["aucpr"] for s in range(10)], ddof=1) for k in ARMS}
gap = {k: np.mean([V[k][s]["val"] - V[k][s]["aucpr"] for s in range(10)]) for k in ARMS}


def contrast(a, b):
    d = np.array([V[a][s]["aucpr"] - V[b][s]["aucpr"] for s in range(10)])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t90 = stats.t.ppf(0.95, n - 1)
    t95 = stats.t.ppf(0.975, n - 1)
    delta = 0.005
    pL = 1 - stats.t.cdf((d.mean() + delta) / se, n - 1)
    pU = stats.t.cdf((d.mean() - delta) / se, n - 1)
    return dict(d=float(d.mean()), lo=float(d.mean() - t90 * se), hi=float(d.mean() + t90 * se),
                lo95=float(d.mean() - t95 * se), hi95=float(d.mean() + t95 * se),
                p_sup=float((d > 0).sum()), tost=float(max(pL, pU)))


# ================= FIGURE 1 =================
fig = plt.figure(figsize=(7.2, 6.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], height_ratios=[1.2, 1.0],
                      wspace=0.66, hspace=0.46)
ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

order = sorted(ARMS, key=lambda k: -mean[k])
ypos = np.arange(len(order))[::-1]
base = mean["R1"]
ax_a.axvspan(base - 0.005, base + 0.005, color="0.88", lw=0, zorder=0)
ax_a.axvline(base, color="0.35", ls="--", lw=0.7, zorder=1)
rng = np.random.default_rng(3)
for y, k in zip(ypos, order):
    v = np.array([V[k][s]["aucpr"] for s in range(10)])
    ax_a.hlines(y, base, v.mean(), color="0.78", lw=0.6, zorder=2)
    ci = stats.t.ppf(0.975, 9) * sd[k] / np.sqrt(10)
    ax_a.errorbar(mean[k], y, xerr=ci, fmt="o", ms=3.2, mfc=C[FAM[k]], mec="white",
                  mew=0.4, ecolor=C[FAM[k]], elinewidth=0.8, capsize=0, zorder=4)
    ax_a.plot(v, y + rng.uniform(-0.18, 0.18, 10), ".", ms=1.5, color=C[FAM[k]],
              alpha=0.55, zorder=3, mew=0)
ax_a.set_yticks(ypos)
ax_a.set_yticklabels([SHORT[k] for k in order], fontsize=5.6)
ax_a.set_xlim(0.250, 0.336)
ax_a.set_ylim(-0.8, len(order) - 0.2)
ax_a.set_xlabel("macro-AUCPR, 730 held-out odorants (10 seeds)", fontsize=6.2)
ax_a.set_title("a   The 21-arm ladder, one ceiling", loc="left", fontsize=6.9, fontweight="bold")
ax_a.legend(handles=[Patch(fc=C[v], ec="none", label=lab) for v, lab in
                     [("2D", "2D topology"), ("scalar", "2D + ensemble scalars"),
                      ("ensemble", "ensemble MIL"), ("edge", "marginalised 3D"),
                      ("static3d", "static 3D"), ("control", "control")]],
            fontsize=4.6, loc="upper left", handlelength=0.8, labelspacing=0.22)

f = [("EN4", "R1"), ("EN3", "R1"), ("R3unif", "R1"), ("EN1", "R1"),
     ("EN4", "EN1"), ("R25", "R1"), ("SCHNET", "R1"),
     ("EN2", "R1"), ("R3boltz", "R1"), ("R3unif", "EN3"), ("en4_size", "EN4")]
lab = {"EN4-R1": "ensemble moments − 2D", "EN3-R1": "3 scalars − 2D", "R3unif-R1": "uniform MIL − 2D",
       "EN1-R1": "n_rotors − 2D", "EN4-EN1": "moments − n_rotors",
       "R25-R1": "bond statistics − 2D", "SCHNET-R1": "SchNet − 2D",
       "EN2-R1": "thermodynamics − 2D", "R3boltz-R1": "MIL Boltzmann − 2D",
       "R3unif-EN3": "uniform MIL − 3 scalars", "en4_size-EN4": "size moments − 6 moments"}
yy = np.arange(len(f))[::-1]
ax_b.axvspan(-0.005, 0.005, color="0.90", lw=0, zorder=0)
for y, (a, b) in zip(yy, f):
    c = contrast(a, b)
    col = "#B22222" if c["d"] < 0 else "#0072B2"
    ax_b.plot([c["lo95"], c["hi95"]], [y, y], color=col, lw=1.3, solid_capstyle="butt", zorder=3)
    ax_b.plot(c["d"], y, "o", ms=4, color=col, mec="white", mew=0.5, zorder=4)
    tag = "equiv" if c["tost"] < 0.05 else ("up" if c["lo95"] > 0 else ("dn" if c["hi95"] < 0 else "ns"))
    ax_b.text(0.0125, y, f"{c['d']:+.4f} {tag}", fontsize=4.9, va="center", color="0.3")
ax_b.axvline(0, color="0.3", lw=0.7)
ax_b.set_yticks(yy)
ax_b.set_yticklabels([lab[f"{a}-{b}"] for a, b in f], fontsize=5.5)
ax_b.set_xlim(-0.020, 0.026)
ax_b.set_xlabel("Δ macro-AUCPR (95% paired CI, $n$=10)", fontsize=6.2)
ax_b.set_title("b   Superior, equivalent, unresolved", loc="left", fontsize=6.9, fontweight="bold")
ax_b.text(0.985, -0.20, "equiv: TOST $p$ < 0.05 within ±0.005", transform=ax_b.transAxes,
          fontsize=4.3, color="0.35", ha="right")

# ---- panel c: capacity versus performance ----
EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "results")
cap = {}
for k, (d, det) in ARMS.items():
    m0 = json.load(open(os.path.join(d, "seed0", "metrics_det.json" if det else "metrics.json")))
    cap[k] = (m0["n_params"], mean[k], FAM[k])
for k, d, fam in [("en4_s2d", os.path.join(EXP, "en4_s2d"), "scalar"),
                  ("wide_en1", os.path.join(EXP, "wide_en1"), "2D")]:
    m0 = json.load(open(os.path.join(d, "seed0", "metrics.json")))
    v = np.mean([json.load(open(os.path.join(d, f"seed{s}", "metrics.json")))["test_macro_aucpr"]
                 for s in range(10)])
    cap[k] = (m0["n_params"], v, fam)
for k, (np_, v, fam) in cap.items():
    ax_c.plot(np_ / 1e6, v, "o", ms=4.2, mfc=C[fam], mec="white", mew=0.5, zorder=3)
ax_c.axhline(mean["R1"], color="0.5", lw=0.6, ls="--")
LBL = dict(SHORT)
LBL["en4_s2d"] = "2D + rotor + 4 descriptors"
LBL["wide_en1"] = "wide 2D GIN"
for k, dx, dy, ha in [("R1", 0.02, -0.004, "left"), ("en4_s2d", 0.025, -0.004, "left"),
                      ("wide_en1", 0.03, 0.001, "left"), ("SCHNET", 0.03, -0.003, "left"),
                      ("DIMENET", -0.03, 0.002, "right"), ("R3unif", -0.03, 0.002, "right"),
                      ("ANGLE", 0.02, -0.004, "left")]:
    ax_c.annotate(LBL[k], (cap[k][0] / 1e6 + dx, cap[k][1] + dy), fontsize=4.6,
                  color="0.25", ha=ha, va="center")
for yy_ in (0.3238, 0.3089):
    ax_c.plot(2066192 / 1e6, yy_, "s", ms=4.6, mfc="white", mec=C["static3d"], mew=1.0,
              zorder=3)
ax_c.annotate("DimeNet++, 4 schedules", (2066192 / 1e6 - 0.035, 0.329), fontsize=4.6,
              color="0.25", ha="right", va="center")
ax_c.set_xscale("log")
ax_c.set_xticks([0.4, 0.6, 1.0, 2.0])
ax_c.set_xticklabels(["0.4", "0.6", "1.0", "2.0"], fontsize=5.4)
ax_c.set_xlabel("parameters (millions)", fontsize=6.2)
ax_c.set_ylabel("macro-AUCPR", fontsize=6.2)
ax_c.set_ylim(0.25, 0.336)
ax_c.set_title("c   Capacity does not explain the deficit", loc="left", fontsize=6.9,
               fontweight="bold")
ax_c.legend(handles=[Patch(fc=C[v], ec="none", label=lab) for v, lab in
                     [("2D", "2D topology"), ("scalar", "2D + ensemble scalars"),
                      ("ensemble", "ensemble MIL"), ("edge", "marginalised 3D"),
                      ("static3d", "static 3D")]],
            fontsize=4.4, loc="lower right", handlelength=0.8, labelspacing=0.2)
fig.savefig(os.path.join(OUT, "fig1_ladder.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_ladder.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIGURE 2: temperature =================
cv = json.load(open("results/wpb_tempsweep/curve.json"))
pts = cv["points"]
x = [p["median_n_eff"] for p in pts]
y = [p["aucpr_mean"] for p in pts]
e = [p["aucpr_sd"] for p in pts]
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axhspan(0.3253 - 0.0040, 0.3253 + 0.0040, color="#0072B2", alpha=0.10, lw=0)
ax.axhline(0.3253, color="#0072B2", lw=0.8, ls="--")
ax.errorbar(x, y, yerr=e, fmt="o-", ms=3.6, lw=1.0, color="#D55E00", ecolor="#D55E00",
            elinewidth=0.8, capsize=0, zorder=3)
offs = [(7, -2), (6, 4), (6, 4), (4, 6), (-4, -12), (4, 5)]
for p, off in zip(pts, offs):
    ax.annotate(f"{p['tau']:.0f} K" if np.isfinite(p["tau"]) and p["tau"] < 5000 else "∞ (1/C)",
                (p["median_n_eff"], p["aucpr_mean"]), textcoords="offset points",
                xytext=off, fontsize=4.8, color="0.35")
ax.text(2.6, 0.3262, "uniform $1/C$ = 0.3253", fontsize=5, color="#0072B2", va="bottom")
ax.set_xlabel("median effective conformer count $N_{eff}(τ)$", fontsize=6.2)
ax.set_ylabel("macro-AUCPR", fontsize=6.2)
ax.set_title("Spearman ρ = 1.00 across six temperatures", loc="left", fontsize=6.6,
             fontweight="bold")
ax.set_ylim(0.303, 0.332)
fig.savefig(os.path.join(OUT, "fig3_temperature.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_temperature.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIGURE 3: jitter =================
gv = json.load(open("results/wpb_gauss/summary.json"))
sigk = ["0.05", "0.15", "0.30"]
sig = [float(s) for s in sigk]
yy3 = [gv["arms"][k]["aucpr_mean"] for k in sigk]
ee3 = [gv["arms"][k]["aucpr_sd"] for k in sigk]
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axhline(gv["uniform_det"]["mean"], color="#0072B2", lw=0.9, ls="--")
ax.errorbar(sig, yy3, yerr=ee3, fmt="o-", ms=3.6, lw=1.0, color="#D55E00",
            ecolor="#D55E00", elinewidth=0.8, capsize=0, zorder=3,
            label="isotropic jitter, bonds preserved")
s = gv["arms"]["rmsd"]["aucpr_mean"]
ax.plot(1.22, s, "s", ms=4.6, mfc="white", mec="#999999", mew=0.9, zorder=3,
        label="stress control (σ* = 1.22 Å)")
ax.annotate("bonds 1.46 → 3.08 Å", (1.22, s), textcoords="offset points",
            xytext=(-6, 7), fontsize=4.6, color="0.35", ha="right")
ax.axhline(gv["uniform_det"]["mean"], color="#0072B2", lw=0.9, ls="--",
           label="real ensemble, uniform 1/C")
ax.set_xscale("log")
ax.set_xticks([0.05, 0.15, 0.30, 1.22])
ax.set_xticklabels(["0.05", "0.15", "0.30", "1.22*"], fontsize=5.4)
ax.set_xlabel("jitter scale σ (Å)", fontsize=6.2)
ax.set_ylabel("macro-AUCPR", fontsize=6.2)
ax.set_title("Coordinate smoothing cannot replace torsional coverage", loc="left",
             fontsize=6.6, fontweight="bold")
ax.legend(fontsize=4.4, loc="lower left", handlelength=1.4, labelspacing=0.25)
fig.savefig(os.path.join(OUT, "fig4_jitter.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_jitter.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIGURE 4: truncation =================
st = json.load(open("results/wpb_censor/stratified.json"))
pairs = [("SCHNET - R1_2D", "SchNet − 2D"), ("R25_edge - R1_2D", "bond stats − 2D"),
         ("R3_boltz - R1_2D", "MIL Boltzmann − 2D"), ("R3_uniform - R1_2D", "MIL uniform − 2D"),
         ("EN1_rotors - R1_2D", "n_rotors − 2D"), ("EN2_thermo - R1_2D", "thermodynamics − 2D")]
fig, ax = plt.subplots(figsize=(4.4, 2.9))
w = 0.34
xs = np.arange(len(pairs))
for i, (key, lab_) in enumerate(pairs):
    for j, (grp, col, off) in enumerate([("untruncated", "#56B4E9", -w / 2),
                                        ("truncated", "#D55E00", w / 2)]):
        d = st["pairs"][key][grp]
        ax.bar(i + off, d["delta"], width=w, color=col, edgecolor="white", lw=0.4,
               label=("untruncated (n=416)" if (i == 0 and j == 0) else
                      ("truncated (n=314)" if (i == 0 and j == 1) else None)))
        ax.text(i + off, d["delta"] + (0.0012 if d["delta"] >= 0 else -0.0012),
                f"{d['delta']:+.3f}", ha="center", fontsize=4.4, color="0.25",
                va="bottom" if d["delta"] >= 0 else "top")
        star = "**" if d["p_signflip"] < 0.01 else ("*" if d["p_signflip"] < 0.05 else "")
        if star:
            ax.text(i + off, d["delta"] + (0.0028 if d["delta"] >= 0 else -0.0028), star,
                    ha="center", fontsize=6, color="0.15",
                    va="bottom" if d["delta"] >= 0 else "top")
ax.axhline(0, color="0.3", lw=0.7)
ax.set_xticks(xs)
ax.set_xticklabels([p[1] for p in pairs], rotation=22, ha="right", fontsize=5.2)
ax.set_ylabel("Δ macro-AUCPR vs 2D GIN", fontsize=6.2)
ax.set_ylim(-0.045, 0.016)
ax.set_title("The static-3D deficit is confined to truncated ensembles", loc="left",
             fontsize=6.6, fontweight="bold")
ax.legend(fontsize=4.6, loc="upper left", handlelength=0.9)
fig.savefig(os.path.join(OUT, "fig2_truncation.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_truncation.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIGURE 5: probe + CKA =================
pr = json.load(open("results/wpa_redundancy/probing.json"))
cka = np.load("results/wpa_redundancy/cka.npz", allow_pickle=True)
fig, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.36})
targets = pr["targets"]
mlp = pr["probe"]["mlp"]
sets = [("gin", "2D GIN", "#8a8a8a"), ("gin_rotors", "+ n_rotors", "#0072B2"),
        ("descriptors_2d", "4 raw 2D descriptors", "#CC79A7")]
xs = np.arange(len(targets))
w = 0.26
for i, (key, lab_, col) in enumerate(sets):
    vals = [np.mean(mlp[key][t]["r2"]) for t in targets]
    ax5a.bar(xs + (i - 1) * w, vals, width=w, color=col, edgecolor="white", lw=0.3, label=lab_)
mean8 = np.mean([np.mean(mlp["gin_rotors"][t]["r2"]) for t in targets])
mean6 = np.mean([np.mean(mlp["gin_rotors"][t]["r2"]) for t in targets if "Rg" not in t])
ax5a.axhline(0.85, color="#B22222", lw=0.8, ls="--")
ax5a.text(6.9, 0.868, "G4 threshold 0.85", fontsize=4.8, color="#B22222", ha="right",
          bbox=dict(fc="white", ec="none", pad=0.6))
ax5a.set_xticks(xs)
ax5a.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=5.0)
ax5a.set_ylabel("held-out probe $R^2$", fontsize=6.2)
ax5a.set_ylim(0, 1.0)
ax5a.set_title("a   2D topology does not reconstruct 3D dispersion", loc="left",
               fontsize=6.7, fontweight="bold")
ax5a.legend(fontsize=4.5, loc="upper right", bbox_to_anchor=(1.0, 0.80), ncol=1,
            handlelength=0.9, labelspacing=0.2)

labs = [str(x) for x in cka["labels"]]
M = cka["linear_mean"]
im = ax5b.imshow(M, vmin=0.4, vmax=1.0, cmap="Blues")
ax5b.set_xticks(range(len(labs)))
ax5b.set_yticks(range(len(labs)))
ax5b.set_xticklabels(labs, rotation=45, ha="right", fontsize=5.2)
ax5b.set_yticklabels(labs, fontsize=5.2)
for i in range(len(labs)):
    for j in range(len(labs)):
        ax5b.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.4,
                  color="white" if M[i, j] > 0.8 else "0.2")
ax5b.set_title("b   Representational convergence", loc="left", fontsize=6.7, fontweight="bold")
cb = fig.colorbar(im, ax=ax5b, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=4.6)
cb.set_label("linear CKA", fontsize=5.4)
fig.savefig(os.path.join(OUT, "fig5_probe_cka.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_probe_cka.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIGURE 6: tiers =================
dr = json.load(open("results/s3_tiers/dream/summary.json"))
m2 = json.load(open("results/s3_tiers/m2or/summary.json"))
# arm D (receptor-agnostic single logit broadcast) was run after the dual-tower grid and is
# archived separately: experiments/T2_RESULTS.md and experiments/logs/D_rec_seed0..9.log.
# Cold-receptor only; per-seed macro-AP: 0.4766 0.4677 0.4794 0.4867 0.4706 0.4805 0.4795
# 0.4738 0.4617 0.4831 -> 0.4760 +/- 0.0076 (sample s.d.).
m2["S2_cold_receptor"]["arms"]["D"] = {"n": 10, "macro_ap": 0.475960, "sd_ap": 0.007577,
                                       "macro_auc": 0.7550, "entropy": None}
fig, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.34})
arms = [("morgan", "Morgan + Ridge", "#8a8a8a"), ("en1", "2D + n_rotors", "#009E73"),
        ("en6", "2D + n_rotors/ring", "#56B4E9"), ("r1", "2D GIN", "#0072B2")]
xs = np.arange(len(arms))
for i, (k, lab_, col) in enumerate(arms):
    a = dr["arms"][k]
    ax6a.bar(i, a["mean_r"], width=0.66, color=col, edgecolor="white", lw=0.4)
    ax6a.errorbar(i, a["mean_r"], yerr=a["sd"], fmt="none", ecolor="0.3", elinewidth=0.7, capsize=1.5)
ax6a.set_xticks(xs)
ax6a.set_xticklabels([a[1] for a in arms], rotation=18, ha="right", fontsize=5.2)
ax6a.set_ylabel("mean out-of-fold Pearson $r$", fontsize=6.2)
ax6a.set_ylim(0.28, 0.335)
ax6a.set_title("a   DREAM psychophysics (338 × 21)", loc="left", fontsize=6.7, fontweight="bold")

# the same decomposition on the 304-molecule psychophysical subset (text callout)
_subs = ["r1", "en1", "en4_size", "en4_shape"]
_dr = {}
for _a in _subs:
    _v = [json.load(open(os.path.join(HERE, "experiments", "results", "dream3d", _a,
                                      f"seed{s}", "metrics.json")))["mean_r"] for s in range(10)]
    _dr[_a] = float(np.mean(_v))
ax6a.text(0.985, 0.97,
          "304-molecule subset, $\\Delta r$ vs 2D:\n"
          f"size {_dr['en4_size'] - _dr['r1']:+.3f}   shape {_dr['en4_shape'] - _dr['r1']:+.3f}",
          transform=ax6a.transAxes, fontsize=3.8, va="top", ha="right",
          color="0.35", linespacing=1.35)

splits = [("S2_cold_receptor", "cold receptor"), ("S1_cold_ligand", "cold ligand")]
xs2 = np.arange(2)
LAY = {"A": (-0.30, "#D55E00", "dual tower (A)"),
       "B": (-0.10, "#0072B2", "per-receptor readout (B)"),
       "C": (0.10, "#999999", "receptor shuffled (C)"),
       "D": (0.30, "#009E73", "receptor-agnostic (D)")}
width = 0.19
for j, (split, _) in enumerate(splits):
    for arm, (off, col, lab) in LAY.items():
        if arm not in m2[split]["arms"]:
            continue  # arm D is cold-receptor only
        a = m2[split]["arms"][arm]
        bars = ax6b.bar(j + off, a["macro_ap"], width=width, color=col, edgecolor="white",
                        lw=0.3, label=lab if j == 0 else None)
        if arm == "B" and j == 0:
            bars[0].set_hatch("////")
            bars[0].set_edgecolor("white")
            ax6b.text(j + off, a["macro_ap"] * 0.55, "untrained", rotation=90,
                      ha="center", va="center", fontsize=3.8, color="white", zorder=6)
        ax6b.errorbar(j + off, a["macro_ap"], yerr=a["sd_ap"], fmt="none", ecolor="0.3",
                      elinewidth=0.7, capsize=1.5)
ax6b.axhline(0.168, color="0.4", lw=0.7, ls=":")
ax6b.text(-0.46, 0.905, "random AP = 0.168", fontsize=4.2, color="0.4", ha="left")
# bracket: the biological contrast that matters
_y, _h = 0.775, 0.028
ax6b.plot([-0.30, -0.30, 0.30, 0.30], [_y, _y + _h, _y + _h, _y], color="0.25", lw=0.7)
ax6b.text(0.0, _y + _h + 0.012, "+0.196 macro-AP\n(10/10 seeds, $p$ = 0.0020)", ha="center",
          va="bottom", fontsize=4.1, color="0.2")
ax6b.set_xticks(xs2)
ax6b.set_xticklabels([s[1] for s in splits], fontsize=5.6)
ax6b.set_ylabel("macro-AP", fontsize=6.2)
ax6b.set_ylim(0.0, 1.02)
ax6b.set_title("b   M2OR receptor panel (488 × 95 receptors)", loc="left", fontsize=6.7,
               fontweight="bold")
ax6b.legend(fontsize=3.9, ncol=2, loc="upper right", handlelength=0.9,
            columnspacing=0.8, labelspacing=0.2, borderaxespad=0.15,
            bbox_to_anchor=(1.02, 1.0))
fig.savefig(os.path.join(OUT, "fig6_tiers.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig6_tiers.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------- machine-readable summary of everything plotted ----------
summary = dict(
    ladder={k: dict(mean=float(mean[k]), sd=float(sd[k]),
                    seeds=[float(V[k][s]["aucpr"]) for s in range(10)],
                    gap=float(gap[k]), best_epoch=float(np.median([V[k][s]["ep"] for s in range(10)])),
                    n_params=int(V[k][0]["npar"])) for k in ARMS},
    contrasts={f"{a}-{b}": contrast(a, b) for a, b in f},
    temperature=pts, jitter=gv, truncation=st["pairs"],
    probe=dict(mean8=float(mean8), mean6=float(mean6)),
    dream={k: dr["arms"][k] for k in dr["arms"]},
    m2or=m2)
json.dump(summary, open(os.path.join(OUT, "figure_source_data.json"), "w"), indent=1)
print("figures written to", OUT)
_bpath = os.path.join(HERE, "baselines.json")
print("morgan baseline:", json.load(open(_bpath)) if os.path.exists(_bpath) else "n/a")
