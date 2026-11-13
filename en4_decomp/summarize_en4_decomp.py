#!/usr/bin/env python3
"""Summarise the EN4 size-vs-shape decomposition against the locked reference arms."""
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "paper2"))
REF = {"R1": ("results/r1/gin", False), "EN1": ("results/s2_ablation/en1", False),
       "EN4": ("results/s2_ablation/en4", False), "EN3": ("results/r25/scalar_gin", False)}
NEW = {"en4_size": "results/en4_size", "en4_shape": "results/en4_shape"}

V = {}
for k, (d, det) in REF.items():
    V[k] = [json.load(open(os.path.join(PROJ, d, f"seed{s}", "metrics.json")))["test_macro_aucpr"]
            for s in range(10)]
for k, d in NEW.items():
    V[k] = [json.load(open(os.path.join(HERE, d, f"seed{s}", "metrics.json")))["test_macro_aucpr"]
            for s in range(10)]


def sf(d):
    d = np.asarray(d, float)
    obs = d.mean()
    cnt = sum(1 for b in itertools.product([-1, 1], repeat=len(d))
              if abs((np.array(b) * d).sum() / len(d)) >= abs(obs) - 1e-12)
    return obs, cnt / 2 ** len(d), int((d > 0).sum())


def ci(d, lvl=0.975):
    se = np.std(d, ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(lvl, len(d) - 1)
    return d.mean() - t * se, d.mean() + t * se


def tost(d, delta=0.005):
    se = np.std(d, ddof=1) / np.sqrt(len(d))
    pL = 1 - stats.t.cdf((d.mean() + delta) / se, len(d) - 1)
    pU = stats.t.cdf((d.mean() - delta) / se, len(d) - 1)
    return max(pL, pU)


print("arm            mean ± SD        best-epoch med   n_params")
for k in ["R1", "EN1", "EN3", "EN4", "en4_size", "en4_shape"]:
    ep = [json.load(open(os.path.join(
        PROJ if k in REF else HERE, REF[k][0] if k in REF else NEW[k],
        f"seed{s}", "metrics.json")))["best_epoch"] for s in range(10)]
    npar = json.load(open(os.path.join(
        PROJ if k in REF else HERE, REF[k][0] if k in REF else NEW[k],
        "seed0", "metrics.json")))["n_params"]
    a = np.array(V[k])
    print(f"{k:14s} {a.mean():.4f} ± {a.std(ddof=1):.4f}   {np.median(ep):5.0f}        {npar}")

rows = []
print("\ncontrast                     Δ       95% CI              seeds  p      TOST p  verdict")
pairs = [("en4_size", "R1"), ("en4_shape", "R1"), ("EN1", "R1"), ("EN4", "R1"),
         ("en4_size", "EN1"), ("en4_shape", "EN1"), ("en4_size", "en4_shape"),
         ("EN4", "en4_size"), ("EN4", "en4_shape")]
for a, b in pairs:
    d = np.array(V[a]) - np.array(V[b])
    obs, p, npos = sf(d)
    lo, hi = ci(d)
    pt = tost(d)
    verdict = ("equivalent" if pt < 0.05 else
               ("superior" if lo > 0 else ("inferior" if hi < 0 else "unresolved")))
    print(f"{a:9s}−{b:9s} {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {npos}/10  {p:.4f}  {pt:.3f}   {verdict}")
    rows.append(dict(a=a, b=b, delta=float(obs), lo=float(lo), hi=float(hi),
                     n_pos=npos, p=float(p), tost=float(pt), verdict=verdict))

add = (np.array(V["en4_size"]) - np.array(V["R1"])) + (np.array(V["en4_shape"]) - np.array(V["R1"]))
whole = np.array(V["EN4"]) - np.array(V["R1"])
obs, p, npos = sf(add - whole)
print(f"\nadditivity: (size−R1)+(shape−R1) − (EN4−R1) = {obs:+.4f} "
      f"[{ci(add-whole)[0]:+.4f}, {ci(add-whole)[1]:+.4f}], p = {p:.4f}")
print(f"sum of parts = {add.mean():+.4f}; EN4 − R1 = {whole.mean():+.4f}")

json.dump(dict(arms={k: dict(mean=float(np.mean(V[k])), sd=float(np.std(V[k], ddof=1)),
                            seeds=[float(x) for x in V[k]]) for k in V},
               contrasts=rows,
               additivity=dict(delta=float(obs), p=float(p),
                               sum_of_parts=float(add.mean()), whole=float(whole.mean()))),
          open(os.path.join(HERE, "en4_decomp_summary.json"), "w"), indent=1)
print("\nwrote en4_decomp_summary.json")
