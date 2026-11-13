#!/usr/bin/env python3
"""Five main figures for the standalone Paper 3 (physical ensembles).

Every plotted number is read from a locked artefact under paper3/v3_results or
paper3/cache; nothing is typed into this file.  Run from the paper3 project root:

    python <this file> <output dir>

Panels and their sources
  Fig 1  cohort                       cache/ensemble_observables.csv, v3_results/*_molecules.csv
  Fig 2  saturation                   v3_results/exp18_figdata/sat_{curves,c95}.csv,
                                      v3_results/exp1_saturation/saturation_fits.csv
  Fig 3  curvature + dispersion       v3_results/exp12_table2/table2.csv,
                                      v3_results/exp8_curvature/curvature_by_observable.csv,
                                      cache/ensemble_observables.csv
  Fig 4  four probes                  exp3_alpb/solvation_summary.csv, exp9_pharm/model_compare.csv,
                                      exp18_figdata/cliffs.csv
  Fig 5  higher tiers                 exp14_tiers/m2or_200.json, exp16_m2or_human/m2or_human.json,
                                      exp15_dream/dream_200.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
V3 = "v3_results"

def _load_mols(name: str):
    """All representative conformers of one molecule, ordered by rank."""
    for f in ("pilot100_representatives.sdf", "pilot200_representatives.sdf",
              "fullrest_representatives.sdf"):
        buf, started = [], False
        path = Path(V3) / f
        if not path.exists():
            continue
        for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
            if m is None:
                continue
            if m.HasProp("name") and m.GetProp("name") == name:
                buf.append(m)
                started = True
            elif started:
                break
        if buf:
            buf.sort(key=lambda x: int(x.GetProp("rank")))
            return buf
    raise FileNotFoundError(name)


def _align(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Qc @ R


def _kabsch_rmsd(A, B):
    B2 = _align(A, B)
    A0 = A - A.mean(0)
    return float(np.sqrt(((A0 - B2) ** 2).sum() / len(A)))


def _hull(ax, pts, color, alpha, lw=0.0):
    from scipy.spatial import ConvexHull
    if len(pts) == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=min(1.0, alpha + 0.15), lw=11,
                solid_capstyle="round", zorder=1)
        return
    if len(pts) < 2:
        ax.add_patch(plt.Circle((pts[0, 0], pts[0, 1]), 0.30, color=color, alpha=alpha,
                                lw=lw, zorder=1))
        return
    try:
        h = ConvexHull(pts)
        ax.add_patch(Polygon(pts[h.vertices], closed=True, facecolor=color,
                             edgecolor=color, alpha=alpha, lw=lw, zorder=1))
    except Exception:
        pass


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
BLUE, ORANGE, GREY, RED, GREEN = "#0072B2", "#D55E00", "#8a8a8a", "#B22222", "#009E73"

# ---------------- shared data ----------------
e = pd.read_csv("cache/ensemble_observables.csv")
e["run1b"] = (e.run_pass == "1b").astype(float)
parts = [pd.read_csv(f"v3_results/{f}_molecules.csv")
         for f in ("pilot100", "pilot200", "fullrest")]
mm = pd.concat(parts, ignore_index=True)
mm = mm[mm.status == "ok"].groupby("name").agg(
    S200=("shannon_entropy", "first"), N200=("n_eff", "first")).reset_index()
d = e.merge(mm, on="name", how="inner")
print("uncapped molecules:", len(d))

# ================= FIG 1: cohort and what the ceiling did =================
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))

rng = np.random.default_rng(5)
for i in rng.choice(len(d), 160, replace=False):
    ax1.plot([0, 1], [np.log(d.N_eff.iloc[i]), np.log(d.N200.iloc[i])],
             color="0.7", lw=0.35, alpha=0.35, zorder=1)
ax1.plot([0, 1], [np.median(np.log(d.N_eff)), np.median(np.log(d.N200))],
         color=ORANGE, lw=1.8, marker="o", ms=4, zorder=3)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["$C_{max}$=20", "$C_{max}$=200"], fontsize=6)
ax1.set_ylabel("ln $N_{eff}$", fontsize=6.2)
ax1.set_title(f"a   {len(d):,} flexible odorants", loc="left", fontsize=6.8, fontweight="bold")
ax1.text(0.5, 4.25, f"median $N_{{eff}}$ {np.exp(np.median(np.log(d.N_eff))):.1f} → "
         f"{np.exp(np.median(np.log(d.N200))):.1f}", ha="center", fontsize=4.9, color=ORANGE)

dS = d.S200 - d.S_conf
ax2.hist(dS, bins=52, color=BLUE, alpha=0.9, edgecolor="white", lw=0.2)
ax2.axvline(0, color="0.4", lw=0.7)
ax2.axvline(np.median(dS), color=ORANGE, lw=1.0)
ax2.text(np.median(dS) + 0.07, ax2.get_ylim()[1] * 0.88,
         f"median +{np.median(dS):.3f} nats", fontsize=4.9, color=ORANGE)
ax2.text(0.03, 0.80, f"{(dS < 0).mean()*100:.1f}% decrease\n(re-embedding drift)",
         transform=ax2.transAxes, fontsize=4.5, color="0.35", va="top")
ax2.set_xlabel("$\\Delta S_{conf}$ (200 − 20), nats", fontsize=6.2)
ax2.set_ylabel("molecules", fontsize=6.2)
ax2.set_title(f"b   {(dS > 0).mean()*100:.1f}% gain entropy", loc="left",
              fontsize=6.8, fontweight="bold")

hb = ax3.hexbin(d.S_conf, d.S200, gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
lim = [min(d.S_conf.min(), d.S200.min()) - 0.05, max(d.S_conf.max(), d.S200.max()) + 0.05]
ax3.plot(lim, lim, "--", color="0.5", lw=0.7)
ax3.axhline(np.log(20), color=RED, lw=0.7, ls=":")
ax3.text(0.45, np.log(20) + 0.04, "ln 20", fontsize=4.8, color=RED)
rho = d[["S_conf", "S200"]].corr(method="spearman").iloc[0, 1]
rho_w = d[["S_conf", "S_top20"]].corr(method="spearman").iloc[0, 1] if "S_top20" in d else np.nan
ax3.text(0.03, 0.96, f"cross-run ρ = {rho:.3f}", transform=ax3.transAxes,
         fontsize=5.2, va="top", color=BLUE)
ax3.text(0.03, 0.86, "nested, same pool ρ = 0.841", transform=ax3.transAxes,
         fontsize=5.2, va="top", color=GREEN)
ax3.set_xlabel("$S_{conf}$ at $C_{max}$=20 (nats)", fontsize=6.2)
ax3.set_ylabel("$S_{conf}$ at $C_{max}$=200 (nats)", fontsize=6.2)
ax3.set_title("c   Rank order is only partly preserved", loc="left", fontsize=6.8,
              fontweight="bold")
plt.colorbar(hb, ax=ax3, fraction=0.045, pad=0.02).ax.tick_params(labelsize=4.4)

ax4.hist(e.L_sem, bins=np.arange(0.5, 22.5, 1), color=GREEN, alpha=0.9,
         edgecolor="white", lw=0.3)
ax4.axvline(e.L_sem.mean(), color="0.3", lw=0.9, ls="--")
ax4.text(e.L_sem.mean() + 0.4, ax4.get_ylim()[1] * 0.9, f"mean {e.L_sem.mean():.1f}",
         fontsize=4.9, color="0.3")
ax4.set_xlabel("semantic breadth $L_{sem}$ (labels)", fontsize=6.2)
ax4.set_ylabel("molecules", fontsize=6.2)
ax4.set_title("d   The outcome variable", loc="left", fontsize=6.8, fontweight="bold")

fig.savefig(os.path.join(OUT, "fig1_cohort.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_cohort.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIG 2: where the ensemble saturates =================
cur = pd.read_csv(f"{V3}/exp18_figdata/sat_curves.csv")
c95 = pd.read_csv(f"{V3}/exp18_figdata/sat_c95.csv")
fits = pd.read_csv(f"{V3}/exp1_saturation/saturation_fits.csv")
tau = float(fits.tau.median())
c95_param = float(fits.C95.median())

caps = np.arange(1, 101)
med = np.array([cur[cur.cap == c].frac.median() if (cur.cap == c).any() else np.nan for c in caps])
q1 = np.array([cur[cur.cap == c].frac.quantile(.25) if (cur.cap == c).any() else np.nan for c in caps])
q3 = np.array([cur[cur.cap == c].frac.quantile(.75) if (cur.cap == c).any() else np.nan for c in caps])

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.7),
                                 gridspec_kw={"width_ratios": [1.35, 0.95, 0.9], "wspace": 0.62})
a1.fill_between(caps, q1, q3, color=BLUE, alpha=0.15, lw=0)
a1.plot(caps, med, lw=1.1, color=BLUE, label=f"all {len(c95):,} flexible odorants")
tail = c95[c95.n_conf >= 100].name
cur_t = cur[cur.name.isin(tail)]
med_t = np.array([cur_t[cur_t.cap == c].frac.median() if (cur_t.cap == c).any() else np.nan
                  for c in caps])
a1.plot(caps, med_t, lw=0.9, ls="--", color=ORANGE,
        label=f"hyper-flexible (n = {len(tail)})")
fit = (1 - np.exp(-caps / tau))
a1.plot(caps, fit, ":", lw=1.2, color=GREY, label=f"exponential, τ = {tau:.1f}")
a1.axvline(20, color="0.4", lw=0.7, ls=":", label="conventional cap, 20")
_f20 = float(cur[cur.cap == 20].frac.median())
a1.text(20.9, 0.53, f"{_f20*100:.0f}% of $S(C_{{total}})$ at $C$ = 20",
          fontsize=4.3, color="0.25", rotation=90, va="bottom", ha="left",
          bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
a1.axvline(c95_param, color=RED, lw=0.8, ls="--", label=f"parametric $C_{{95}}$ = {c95_param:.1f}")
a1.axvline(float(c95.C95.median()), color=GREEN, lw=0.9, ls="-",
           label=f"model-free $C_{{95}}$ = {float(c95.C95.median()):.0f}")
a1.set_xlabel("conformer cap $C$", fontsize=6.2)
a1.set_ylabel("$S(C)\\,/\\,S(C_{total})$", fontsize=6.2)
a1.set_xlim(0, 101)
a1.set_ylim(0.45, 1.02)
a1.legend(fontsize=4.0, loc="lower right", handlelength=1.5, labelspacing=0.2)
a1.set_title("a   Saturation vs each molecule's own S", loc="left",
             fontsize=6.4, fontweight="bold")

surv = [(c95.C95 <= c).mean() for c in caps]
a2.plot(caps, np.array(surv) * 100, lw=1.1, color=BLUE)
a2.axhline(50, color="0.6", lw=0.6, ls=":")
m = float(c95.C95.median())
a2.axvline(m, color=GREEN, lw=0.9)
a2.text(m + 2, 55, f"median {m:.0f}", fontsize=4.8, color=GREEN)
a2.set_xlabel("conformer cap $C$", fontsize=6.2)
a2.set_ylabel("% of molecules at 95% of $S$", fontsize=6.2)
a2.set_xlim(0, 101)
a2.set_ylim(0, 102)
a2.set_title("b   Half saturated by 28", loc="left", fontsize=6.4, fontweight="bold")

a3.hist(c95.C95, bins=np.arange(0, 102, 2), color=BLUE, alpha=0.85, edgecolor="white", lw=0.3)
a3.axvline(c95_param, color=RED, lw=0.8, ls="--")
a3.text(c95_param + 1.5, a3.get_ylim()[1] * 0.92, f"fit\n{c95_param:.1f}", fontsize=4.4, color=RED)
a3.set_xlabel("model-free $C_{95}$ (conformers)", fontsize=6.2)
a3.set_ylabel("molecules", fontsize=6.2)
a3.set_xlim(0, 101)
a3.set_title(f"c   $C_{{95}}$ = {m:.0f}, not 75", loc="left", fontsize=6.4, fontweight="bold")
a3.text(0.97, 0.85, f"tail (n={len(tail)}):\nmedian {float(c95[c95.n_conf>=100].C95.median()):.0f}",
        transform=a3.transAxes, ha="right", fontsize=4.4, color=ORANGE)
fig.savefig(os.path.join(OUT, "fig2_saturation.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_saturation.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIG 3: no optimum in entropy, one in dispersion ==========
t2 = pd.read_csv(f"{V3}/exp12_table2/table2.csv")
show = ["A. Phase I primary (as published)", "B. Untruncated only",
        "C. Prospective, design-weighted", "D. Complete capped population",
        "E. Whole cohort, complete ensembles",
        "G. Paired difference, pre-registered scale",
        "H. Paired difference, scale-free (per-ceiling s.d.)"]
short = {"A. Phase I primary (as published)": "whole cohort, $C$=20",
         "B. Untruncated only": "untruncated only, $C$=20",
         "C. Prospective, design-weighted": "prospective n=200, $C$=200",
         "D. Complete capped population": "capped n=1,821, $C$=200",
         "E. Whole cohort, complete ensembles": "whole cohort, $C$=200",
         "G. Paired difference, pre-registered scale": "paired Δ, fixed scale",
         "H. Paired difference, scale-free (per-ceiling s.d.)": "paired Δ, scale-free"}
fam = {k: ("C20" if "=20" in v else ("diff" if "Δ" in v else "C200")) for k, v in short.items()}

fig = plt.figure(figsize=(7.2, 7.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.86], hspace=0.72, wspace=0.40)
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])
axd1 = fig.add_subplot(gs[2, 0])
axd2 = fig.add_subplot(gs[2, 1])

yy = np.arange(len(show))[::-1]
for y, k in zip(yy, show):
    r = t2[t2.row == k].iloc[0]
    col = {"C20": GREY, "C200": BLUE, "diff": ORANGE}[fam[k]]
    if np.isfinite(r.ci_lo):
        axa.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.3, solid_capstyle="butt")
    axa.plot(r.beta2, y, "D" if fam[k] == "diff" else "o", ms=5 if fam[k] == "diff" else 4,
             color=col, mec="white", mew=0.5)
    lab = f"{r.beta2:+.4f}"
    hi_ = r.ci_hi if np.isfinite(r.ci_hi) else r.beta2
    if hi_ < 0.02:
        axa.text(hi_ + 0.004, y, lab, fontsize=4.6, va="center", color="0.35")
    else:
        axa.text(r.beta2, y + 0.30, lab, fontsize=4.6, ha="center", color="0.35")
axa.axvline(0, color="0.3", lw=0.7)
need = [(-0.057 * 0.918917) / (2 * (np.log(ne) - 1.5972)) for ne in (10, 20, 50, 100)]
for i, (v, ne) in enumerate(zip(need, (10, 20, 50, 100))):
    axa.axvline(v, color=RED, lw=0.6, ls=(0, (2, 2)))
    axa.text(v, -0.78 - 0.34 * (i % 2), f"{ne}", fontsize=4.2, color=RED, ha="center")
axa.text(-0.104, -0.78, "dashed: β₂ required for a peak at $N_{eff}$ =", fontsize=4.4, color=RED)
axa.set_yticks(yy)
axa.set_yticklabels([short[k] for k in show], fontsize=5.4)
axa.set_xlim(-0.105, 0.075)
axa.set_ylim(-1.3, len(show) - 0.4)
axa.set_xlabel(r"$\beta_2$ (curvature of log breadth)", fontsize=6.2)
axa.set_title("a   Torsional entropy: every estimate, and where an optimum would sit",
              loc="left", fontsize=6.6, fontweight="bold")

geo = pd.read_csv(f"{V3}/exp8_curvature/curvature_by_observable.csv")
geo = geo[geo.spec == "common"].set_index(["observable", "ceiling"])["beta2"]
ci = {}
for lab, kee in (("rmsd_ab_20", ("rmsd_ab", 20)), ("rmsd_ab_200", ("rmsd_ab", 200))):
    j = json.load(open(f"{V3}/exp8_curvature/rmsd_ci.json"))[lab]
    ci[kee] = j["beta2_ci"]
t7 = pd.read_csv(f"{V3}/exp13_t2_t7/t7_observable_ci.csv")
for r in t7.itertuples():
    ci[(r.observable, int(r.ceiling))] = [r.ci_lo, r.ci_hi]

names = {"rmsd_ab": r"$\langle$RMSD$_{ab}\rangle$", "sigma2_rg": r"$\sigma^2(R_g)$",
         "tr_sigma_npr": r"tr$\Sigma$(NPR)", "sigma_sasa_polar": r"$\sigma$(polar SASA)"}
yy = np.arange(len(names))[::-1]
for y, ob in zip(yy, names):
    for ceil, col, off in ((20, GREY, 0.16), (200, BLUE, -0.16)):
        lo, hi = ci[(ob, ceil)]
        axb.plot([lo, hi], [y + off, y + off], color=col, lw=1.0)
        axb.plot(float(geo.loc[(ob, ceil)]), y + off, "o", ms=3.4, color=col,
                 mec="white", mew=0.4)
adj = json.load(open(f"{V3}/exp24_adjusted/adjusted.json"))
for y, ob in zip(yy, names):
    if ob in adj:
        r = adj[ob]
        axb.plot([r["ci"][0], r["ci"][1]], [y - 0.16, y - 0.16], color=RED, lw=1.0, alpha=0.8)
        axb.plot(r["beta2"], y - 0.16, "D", ms=3.6, mfc="white", mec=RED, mew=0.7)
axb.axvline(0, color="0.3", lw=0.7)
axb.set_yticks(yy)
axb.set_yticklabels([names[o] for o in names], fontsize=5.2)
axb.set_xlabel(r"$\beta_2$ of the dispersion observable", fontsize=6.2)
axb.set_xlim(-0.06, 0.01)
axb.set_title("b   Only polar-SASA survives the rotor control", loc="left", fontsize=6.4, fontweight="bold")
axb.plot([], [], "o", color=GREY, ms=3.4, label="unadjusted, $C$=20")
axb.plot([], [], "o", color=BLUE, ms=3.4, label="unadjusted, $C$=200")
axb.plot([], [], "D", mfc="white", mec=RED, ms=3.6, label="rotor-adjusted")
_ypol = yy[list(names).index("sigma_sasa_polar")]
axb.plot(-0.0264, _ypol + 0.02, "*", ms=7.0, color=GREEN, mec="white", mew=0.3, zorder=7)
axb.plot(-0.0463, _ypol + 0.02, "*", ms=7.0, mfc="white", mec=GREEN, mew=0.9, zorder=7)
axb.text(-0.0264, _ypol + 0.30, "joint", fontsize=3.9, color=GREEN, ha="center")
axb.text(-0.0463, _ypol + 0.30, "held-out", fontsize=3.9, color=GREEN, ha="center")
axb.plot([], [], "*", color=GREEN, ms=5.5, label="3-axis joint model")
axb.plot([], [], "*", mfc="white", mec=GREEN, ms=5.5, mew=0.8, label="held-out split")
axb.legend(fontsize=3.6, loc="upper center", bbox_to_anchor=(0.46, -0.30), ncol=3,
           handletextpad=0.3, columnspacing=0.9, labelspacing=0.2, frameon=False)

dd = e[e.name != "GS_LF_03198"].copy()
dd["nrot"] = dd["n_rotors"].astype(float)
q = pd.qcut(dd.nrot, 9, labels=False, duplicates="drop")
mu = dd.groupby(q).nrot.mean()
mv = dd.groupby(q).L_sem.mean()
axc.plot(mu.values, mv.values, "o-", ms=3.2, lw=1.2, color=ORANGE)
axc.axvline(9.2, color=RED, lw=0.8, ls="--")
axc.axvline(6.1, color=RED, lw=0.7, ls=(0, (2, 2)), alpha=0.65)
axc.text(8.95, 4.62, "9.2 bonds\n(pipeline count)", fontsize=4.2, color=RED,
         va="center", ha="right")
axc.text(5.95, 4.16, "6.1 bonds\n(strict RDKit)", fontsize=4.0, color=RED, va="bottom", ha="right")

# ---- panel d: what sigma(polar SASA) measures, promoted from Supplementary Fig. 1
obs = pd.read_csv(f"{V3}/exp7_obs200/observables_200.csv").set_index("name")
PICKS = [("GS_LF_04137", "coherent polar anchor"),
         ("GS_LF_01345", "intermittent polar burial")]
PCOL = {"O": "#D62728", "N": "#1F77B4", "S": "#E3B505"}
for ax, (nm, desc) in zip((axd1, axd2), PICKS):
    mols = _load_mols(nm)
    heavy = [a.GetIdx() for a in mols[0].GetAtoms() if a.GetAtomicNum() > 1]
    pos = {a: k for k, a in enumerate(heavy)}
    P = np.asarray([m.GetConformer(0).GetPositions()[heavy] for m in mols], float)
    ground = P[0]
    rms = np.asarray([_kabsch_rmsd(ground, P[i]) for i in range(1, len(P))])
    far = int(np.argmax(rms)) + 1
    Qal = _align(ground, P[far])
    Pc = ground - ground.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    proj = lambda X: (X - X.mean(0)) @ Vt[:2].T
    G2, F2 = proj(Pc), proj(Qal)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mols[0].GetBonds()
             if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1]
    for a, b in bonds:
        if a in pos and b in pos:
            i, j = pos[a], pos[b]
            ax.plot([F2[i, 0], F2[j, 0]], [F2[i, 1], F2[j, 1]], color=ORANGE, lw=0.85,
                    alpha=0.55, solid_capstyle="round", zorder=2)
    for a, b in bonds:
        if a in pos and b in pos:
            i, j = pos[a], pos[b]
            ax.plot([G2[i, 0], G2[j, 0]], [G2[i, 1], G2[j, 1]], color=GREY, lw=1.15,
                    solid_capstyle="round", zorder=3)
    sym = [a.GetSymbol() for a in mols[0].GetAtoms() if a.GetAtomicNum() > 1]
    pol = [k for k, el in enumerate(sym) if el in PCOL]
    if len(pol) >= 2:
        _hull(ax, G2[pol], "#17BECF", 0.30)
        _hull(ax, F2[pol], "#D62728", 0.30)
    for k in pol:
        ax.plot(G2[k, 0], G2[k, 1], "o", ms=4.0, mfc=PCOL[sym[k]], mec="white", mew=0.4, zorder=5)
        ax.plot(F2[k, 0], F2[k, 1], "o", ms=3.4, mfc="none", mec=PCOL[sym[k]], mew=0.8, zorder=5)
    ax.scatter(G2[:, 0], G2[:, 1], s=2.6, color="0.25", zorder=4, lw=0)
    ax.set_aspect("equal")
    allx = np.concatenate([G2[:, 0], F2[:, 0]])
    ally = np.concatenate([G2[:, 1], F2[:, 1]])
    cx, cy = (allx.max() + allx.min()) / 2, (ally.max() + ally.min()) / 2
    sx = allx.ptp() * 0.52 + 0.08
    sy = ally.ptp() * 0.50 + 0.06
    ax.set_xlim(cx - sx, cx + sx)
    ax.set_ylim(cy - sy, cy + sy)
    ax.set_box_aspect(0.62)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    r = obs.loc[nm]
    ax.set_title(("d   " if nm == PICKS[0][0] else "") +
                 f"$\\sigma$(polar SASA) = {r.sigma_sasa_polar:.1f} Å$^2$   {desc}",
                 fontsize=6.2, fontweight="bold", pad=3, loc="left")
    ax.text(0.5, -0.02, f"{nm}   $S_{{\\mathrm{{conf}}}}$ = {r.S_conf:.2f} nats   "
            f"RMSD = {rms[far-1]:.2f} Å", transform=ax.transAxes, fontsize=4.7,
            va="top", ha="center", color="0.3")

axd1.plot([], [], color=GREY, lw=1.3, label="ground state")
axd1.plot([], [], color=ORANGE, lw=1.1, label="most distant conformer")
axd1.plot([], [], "s", color="#17BECF", alpha=0.5, ms=4, label="polar surface, ground")
axd1.plot([], [], "s", color="#D62728", alpha=0.5, ms=4, label="polar surface, distant")
axd1.legend(fontsize=3.5, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2,
            handlelength=0.9, labelspacing=0.18, columnspacing=1.0, borderaxespad=0.0)
axc.set_xlabel("rotatable bonds (bin mean)", fontsize=6.2)
axc.set_ylabel("mean semantic breadth", fontsize=6.2)
axc.set_title("c   Breadth is concave in rotor count", loc="left", fontsize=6.4,
              fontweight="bold")
fig.savefig(os.path.join(OUT, "fig3_curvature.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_curvature.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIG 4: four orthogonal probes =================
sol = pd.read_csv(f"{V3}/exp20_hexane/solvation_all_corrected.csv")
ph = pd.read_csv(f"{V3}/exp9_pharm/model_compare.csv")
cl = pd.read_csv(f"{V3}/exp18_figdata/cliffs.csv")
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
(a, b), (cc, ddp) = axes

a.hist(sol.dS_water, bins=34, color=BLUE, alpha=0.72, label="water", edgecolor="white", lw=0.2)
a.hist(sol.dS_oct, bins=34, color=ORANGE, alpha=0.60, label="octanol", edgecolor="white", lw=0.2)
a.hist(sol.dS_hexane, bins=34, color=GREEN, alpha=0.55, label="n-hexane", edgecolor="white", lw=0.2)
a.axvline(0.514, color=RED, lw=1.0, ls="--")
a.text(0.53, a.get_ylim()[1] * 0.9, " uncapping\n +0.514", fontsize=4.6, color=RED)
a.set_xlim(-0.8, 1.6)
a.set_xlabel("$\\Delta S_{conf}$ under solvation (nats)", fontsize=6.2)
a.set_ylabel("molecules", fontsize=6.2)
a.legend(fontsize=4.4, loc="upper right")
for _v, _c in ((sol.dS_water.median(), BLUE), (sol.dS_oct.median(), ORANGE),
               (sol.dS_hexane.median(), GREEN)):
    a.plot([_v, _v], [0, a.get_ylim()[1] * 0.10], color=_c, lw=1.3, zorder=5)
for _i, (_lab, _v, _c) in enumerate((("water", sol.dS_water.median(), BLUE),
                                     ("1-octanol", sol.dS_oct.median(), ORANGE),
                                     ("n-hexane", sol.dS_hexane.median(), GREEN))):
    a.text(0.015, 0.92 - 0.075 * _i, f"{_lab}  median {_v:+.3f} nats",
           transform=a.transAxes, fontsize=4.1, color=_c, va="center")
a.text(0.015, 0.99, "median $\\Delta S_{conf}$", transform=a.transAxes,
       fontsize=4.3, color="0.3", va="center")
a.set_title("a   Solvation shifts entropy by <0.12 nats", loc="left", fontsize=6.4,
            fontweight="bold")

b.hist(sol.I_cham, bins=32, color=GREEN, alpha=0.85, edgecolor="white", lw=0.2)
b.axvline(0.60, color="0.35", lw=0.8, ls="--")
b.text(0.61, b.get_ylim()[1] * 0.9, "$k_BT$ = 0.60", fontsize=4.6, color="0.35")
b.set_xlabel("chameleonicity index (kcal mol$^{-1}$)", fontsize=6.2)
b.set_ylabel("molecules", fontsize=6.2)
b.set_title(f"b   Median {sol.I_cham.median():.3f} kcal mol$^{{-1}}$", loc="left",
            fontsize=6.4, fontweight="bold")

order = ["S_pharm_60", "S_pharm_p5", "S_pharm_p8", "S_pharm_p10", "S_pharm_p15",
         "S_pharm_ksg", "S_conf + S_pharm_60"]
labs = ["60-d", "5 PC", "8 PC", "10 PC", "15 PC", "KSG 5-NN", "+ torsional"]
vals = [float(ph.loc[ph.model == m, "dAIC_vs_Sconf"].iloc[0]) for m in order]
xs = np.arange(len(order))
cc.bar(xs, vals, color=ORANGE, width=0.62, edgecolor="white", lw=0.4)
cc.axhline(-10, color=RED, lw=0.8, ls="--")
cc.text(len(order) - 0.5, -9.2, "promotion threshold (−10)", fontsize=4.2, color=RED, ha="right")
for x, v in zip(xs, vals):
    cc.text(x, v + 0.5, f"+{v:.1f}", ha="center", fontsize=4.3, color="0.25")
cc.set_xticks(xs)
cc.set_xticklabels(labs, fontsize=4.6, rotation=30, ha="right")
cc.set_ylabel("ΔAIC vs torsional $S_{conf}$", fontsize=6.2)
cc.set_ylim(-12, max(vals) + 4)
cc.set_title("c   Pharmacophore entropy is worse, five ways", loc="left", fontsize=6.4,
             fontweight="bold")

strata = [("both200", "both complete\n(n=527)"), ("both20", "neither complete\n(n=257)")]
metrics = [("dS", r"$|\Delta S_{conf}|$"), ("dN", r"$|\Delta N_{eff}|$")]
yy = np.arange(len(metrics) * len(strata))[::-1]
k = 0
for sm, slab in strata:
    for mk, mlab in metrics:
        r = cl[(cl.stratum == sm) & (cl.metric == mk)].iloc[0]
        col = BLUE if sm == "both200" else GREY
        ddp.plot([r.ci_lo, r.ci_hi], [yy[k], yy[k]], color=col, lw=1.1)
        ddp.plot(r.auc, yy[k], "o", ms=3.8, color=col, mec="white", mew=0.4)
        ddp.text(0.60, yy[k], f"{r.auc:.3f}", fontsize=4.5, va="center", color="0.3")
        k += 1
ddp.axvline(0.5, color="0.4", lw=0.8, ls="--")
ddp.set_yticks(yy)
ddp.set_yticklabels([m[1] for m in metrics] * 2, fontsize=5.0)
ddp.text(0.345, yy[0] - 0.55, "both complete", fontsize=4.4, color=BLUE)
ddp.text(0.345, yy[2] - 0.55, "neither complete", fontsize=4.4, color="0.4")
ddp.set_xlim(0.34, 0.64)
ddp.set_xlabel("AUC (stratified by ceiling availability)", fontsize=6.2)
ddp.set_title("d   Cliffs: chance, in both strata", loc="left", fontsize=6.4,
              fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_rescue.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_rescue.pdf"), bbox_inches="tight")
plt.close(fig)

# ================= FIG 5: cross-tier =================
m2 = json.load(open(f"{V3}/exp14_tiers/m2or_200.json"))
hum = json.load(open(f"{V3}/exp16_m2or_human/m2or_human.json"))
dm = pd.read_csv(f"{V3}/exp15_dream/dream_200.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.38})

panels = [("all 3D-matched (n=294)", m2["all_20"], m2["all_200"], BLUE),
          ("≥50 receptors (n=42)", m2["wide50_20"], m2["wide50_200"], GREEN),
          ("human ORs (n=286)", hum["20"], hum["200"], ORANGE)]
xs = np.arange(len(panels))
w = 0.34
for i, (lab, r20_, r200, col) in enumerate(panels):
    for off, r, alpha in ((-w / 2, r20_, 0.45), (w / 2, r200, 1.0)):
        a.plot([i + off] * 2, [r["ci"][0], r["ci"][1]], color=col, lw=1.3, alpha=alpha,
               solid_capstyle="butt")
        a.plot(i + off, r["beta2"], "o", ms=4, color=col, alpha=alpha, mec="white", mew=0.4)
        a.text(i + off, r["ci"][1] + 0.012, f"{r['beta2']:+.3f}", ha="center", fontsize=4.2,
               color=col)
a.axhline(0, color="0.3", lw=0.7)
a.set_xticks(xs)
a.set_xticklabels([p[0] for p in panels], fontsize=5.0)
a.set_ylabel("$\\beta_2$ for receptor breadth", fontsize=6.2)
a.set_ylim(-0.22, 0.46)
a.plot([], [], "o", color="0.5", ms=4, alpha=0.45, label="$C_{max}$=20")
a.plot([], [], "o", color="0.5", ms=4, label="$C_{max}$=200")
a.legend(fontsize=4.4, loc="lower right", handlelength=1.2)
a.set_title("a   Receptor tier: positive in all six cells", loc="left", fontsize=6.4,
            fontweight="bold")

dm = dm.sort_values("delta_r2")
for i, r in enumerate(dm.itertuples()):
    col = BLUE if (r.delta_ci_lo > 0 or r.delta_ci_hi < 0) else GREY
    b.plot([r.delta_ci_lo, r.delta_ci_hi], [i, i], color=col, lw=0.9, alpha=0.9)
    b.plot(r.delta_r2, i, "o", ms=2.6, color=col, mec="white", mew=0.3)
b.axvline(0, color="0.3", lw=0.7)
b.set_yticks(range(len(dm)))
b.set_yticklabels([str(t).title()[:17] for t in dm.target], fontsize=4.1)
b.set_xlabel("$\\Delta R^2$ from adding 3D ensemble features", fontsize=6.2)
b.set_title(f"b   DREAM, complete ceiling: min $q$ = {dm.q_bh.min():.2f}",
            loc="left", fontsize=6.4, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig5_tiers.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_tiers.pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote five figures to", OUT)
