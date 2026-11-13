"""Rebuild Paper 3 figures from results/ (no hand-editing)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3.common import RESULTS
FIG = RESULTS / "figures"; FIG.mkdir(parents=True, exist_ok=True)

def err(ax, x, y, se, **kw):
    ax.errorbar(x, y, yerr=se, fmt="o-", capsize=3, **kw)

# 1 primary binned
d=pd.read_csv(RESULTS/"wp1_semantic"/"wp1_binned.csv")
x=np.sqrt(d.bin_left*d.bin_right)
fig,ax=plt.subplots(figsize=(5,3.5)); err(ax,x,d["mean"],d["se"],color="C0")
ax.set_xscale("log"); ax.set_xlabel("N_eff (bin geometric mean)"); ax.set_ylabel("L_sem (mean ± SE)")
ax.set_title("WP1 primary: no interior decline")
fig.tight_layout(); fig.savefig(FIG/"wp1_primary_binned.png",dpi=200); plt.close(fig)

# 2 rmsd binned
d=pd.read_csv(RESULTS/"wp1_semantic"/"wp1_rmsd_binned.csv")
x=(d.left+d.right)/2
fig,ax=plt.subplots(figsize=(5,3.5)); err(ax,x,d["mean"],d["se"],color="C1")
ax.set_xlabel("Boltzmann-weighted pairwise RMSD (bin midpoint, Å)"); ax.set_ylabel("L_sem (mean ± SE)")
ax.set_title("WP1 exploratory: RMSD inverted-U")
fig.tight_layout(); fig.savefig(FIG/"wp1_rmsd_binned.png",dpi=200); plt.close(fig)

# 3 WP4 delta R2
d=pd.read_csv(RESULTS/"wp4_dream"/"wp4_targets.csv").sort_values("delta_r2")
fig,ax=plt.subplots(figsize=(6,5))
ax.errorbar(d.delta_r2, np.arange(len(d)), xerr=[d.delta_r2-d.delta_ci_lo, d.delta_ci_hi-d.delta_r2], fmt="o", capsize=2, color="C2")
ax.axvline(0,color="k",lw=0.8)
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.target,fontsize=7)
ax.set_xlabel("ΔR² (ensemble − 2D), 95% bootstrap CI"); ax.set_title("WP4 DREAM: no target survives FDR")
fig.tight_layout(); fig.savefig(FIG/"wp4_dream_delta_r2.png",dpi=200); plt.close(fig)

# 4 WP3 AUC
s=json.load(open(RESULTS/"wp3_cliffs"/"wp3_summary.json"))
a=pd.DataFrame(s["auc_test"])
fig,ax=plt.subplots(figsize=(5,3.2))
ax.errorbar(a.auc, np.arange(len(a)), xerr=[a.auc-a.ci_lo,a.ci_hi-a.auc], fmt="o", capsize=3, color="C3")
ax.axvline(0.5,color="k",ls="--",lw=0.8); ax.set_yticks(np.arange(len(a))); ax.set_yticklabels(a.metric)
ax.set_xlabel("test-only AUC (95% CI)"); ax.set_xlim(0.3,0.7); ax.set_title("WP3 cliffs: no entropy separation")
fig.tight_layout(); fig.savefig(FIG/"wp3_auc.png",dpi=200); plt.close(fig)

# 5 WP2 forest
s=json.load(open(RESULTS/"wp2_m2or"/"wp2_summary.json"))
rows=[]
for k in ["primary","wide50","human_only"]:
    if s.get(k):
        rows.append((k,s[k]["beta2"]["beta"],s[k]["beta2"]["ci_lo"],s[k]["beta2"]["ci_hi"],s[k]["n"]))
fig,ax=plt.subplots(figsize=(5,2.8))
for i,(k,b,lo,hi,n) in enumerate(rows):
    ax.errorbar(b,i,xerr=[[b-lo],[hi-b]],fmt="o",capsize=3,color="C4")
    ax.text(hi+0.01,i,f"n={n}",va="center",fontsize=8)
ax.axvline(0,color="k",lw=0.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("β2 (S_conf²), 95% CI"); ax.set_title("WP2 M2OR: positive, not inverted-U")
fig.tight_layout(); fig.savefig(FIG/"wp2_forest.png",dpi=200); plt.close(fig)
print("wrote", FIG)

# 6 cohort overview
import pandas as pd
obs=pd.read_csv(RESULTS.parent/"cache"/"ensemble_observables.csv")
fig,axs=plt.subplots(2,2,figsize=(8,6))
axs[0,0].hist(obs.N_eff,bins=40,color="C0"); axs[0,0].set_xlabel("N_eff"); axs[0,0].set_ylabel("molecules"); axs[0,0].set_title("A  N_eff")
axs[0,1].hist(obs.L_sem,bins=range(1,23),color="C1"); axs[0,1].set_xlabel("L_sem"); axs[0,1].set_ylabel("molecules"); axs[0,1].set_title("B  semantic label count")
q=pd.qcut(obs.N_eff,10,duplicates="drop"); g=obs.groupby(q,observed=True).agg(cap=("capped20","mean"),mid=("N_eff","median"))
axs[1,0].plot(g.mid,g.cap*100,"o-",color="C2"); axs[1,0].set_xscale("log"); axs[1,0].set_xlabel("N_eff (bin median)"); axs[1,0].set_ylabel("% at 20-conformer cap"); axs[1,0].set_title("C  cap is concentrated at high N_eff")
axs[1,1].hexbin(obs.S_conf,obs.L_sem,gridsize=30,cmap="viridis",mincnt=1); axs[1,1].set_xlabel("S_conf"); axs[1,1].set_ylabel("L_sem"); axs[1,1].set_title("D  raw relationship")
fig.tight_layout(); fig.savefig(FIG/"fig1_cohort.png",dpi=200); plt.close(fig)

# 7 A2 controls
s=json.load(open(RESULTS/"wp5_bioactive"/"wp5_summary.json"))
fig,ax=plt.subplots(figsize=(4.5,3))
vals=[s["weighted_prop_cstar_not1"],s["mean_prop_prior_not1"],s["mean_prop_shuffle"]]
ax.bar(["A2 surrogate","Boltzmann prior","label shuffle"],[v*100 for v in vals],color=["C0","C1","C2"])
ax.set_ylabel("% c* ≠ 1"); ax.set_ylim(0,100); ax.set_title("A2: no label-specific selection")
for i,v in enumerate(vals): ax.text(i,v*100+2,f"{v*100:.1f}%",ha="center",fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"fig4_a2_controls.png",dpi=200); plt.close(fig)
