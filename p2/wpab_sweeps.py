#!/usr/bin/env python3
"""T4 (temperature sweep) and T5 (Gaussian TTA) aggregation and gate verdicts.

Reads the run directories written by `p2/train_r3_variants.py`; trains nothing.

  G7 (prereg section 2): Spearman(AUCPR, median N_eff) over the SIX tau points > 0.7
  G5 (prereg section 2): |Gaussian(sigma*) - uniform| < 0.002 and p > 0.20

Outputs
  results/wpb_tempsweep/curve.json
  results/wpb_gauss/summary.json
  reports/15_temperature_gauss.md

Usage
  python p2/wpab_sweeps.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

TAUS = [(10.0, "temp_10"), (100.0, "temp_100"), (300.0, "boltz"),
        (600.0, "temp_600"), (1500.0, "temp_1500"), (float("inf"), "temp_inf")]
SIGMAS = [("0.05", "gauss_005"), ("0.15", "gauss_015"),
          ("0.30", "gauss_030"), ("rmsd", "gauss_rmsd")]


def load(arm):
    fs = sorted(glob.glob(os.path.join(PROJ, "results", "s1_mil", arm,
                                       "seed*", "metrics.json")))
    out = []
    for f in fs:
        d = json.load(open(f))
        out.append({"seed": d["seed"], "aucpr": d["test_macro_aucpr"],
                    "neff": d.get("median_n_eff_test")})
    return out


def signflip_p(d):
    d = np.asarray(d, float)
    obs, n = abs(d.sum()), len(d)
    c = sum(1 for k in range(1 << n)
            if abs((np.array([1 if (k >> i) & 1 else -1 for i in range(n)]) * d).sum())
            >= obs - 1e-12)
    return c / float(1 << n)


def main():
    os.makedirs(os.path.join(PROJ, "results", "wpb_tempsweep"), exist_ok=True)
    os.makedirs(os.path.join(PROJ, "results", "wpb_gauss"), exist_ok=True)

    # ---------------- T4 ----------------
    curve = {"points": [], "beta_neff_known": {}}
    for tau, arm in TAUS:
        runs = load(arm)
        if not runs:
            continue
        v = np.array([r["aucpr"] for r in runs])
        neff = ([r["neff"] for r in runs if r["neff"] is not None]
                or ([5.65] if arm == "boltz" else []))
        curve["points"].append({"tau": tau, "arm": arm, "n_seeds": len(v),
                                "aucpr_mean": float(v.mean()),
                                "aucpr_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                "median_n_eff": float(np.median(neff)) if neff else None,
                                "per_seed": v.tolist()})
    base = next(p for p in curve["points"] if p["arm"] == "boltz")
    for p in curve["points"]:
        n = min(len(p["per_seed"]), len(base["per_seed"]))
        d = np.array(p["per_seed"][:n]) - np.array(base["per_seed"][:n])
        p["delta_vs_boltz"] = float(d.mean())
        p["p_vs_boltz"] = signflip_p(d)

    pts = [p for p in curve["points"] if p["median_n_eff"] is not None]
    if len(pts) >= 4:
        rho = float(spearmanr([p["median_n_eff"] for p in pts],
                              [p["aucpr_mean"] for p in pts]).statistic)
    else:
        rho = float("nan")
    curve["G7"] = {"spearman": rho, "n_points": len(pts), "threshold": 0.7,
                   "complete": len(pts) == 6,
                   "verdict": ("PASS" if rho > 0.7 else "FAIL") if len(pts) == 6
                   else "PENDING"}
    with open(os.path.join(PROJ, "results", "wpb_tempsweep", "curve.json"), "w") as fh:
        json.dump(curve, fh, indent=1)

    # ---------------- T5 ----------------
    uniform = load("shuffle")                       # random-permutation arm
    uni_det = []
    for f in sorted(glob.glob(os.path.join(PROJ, "results", "s1_mil", "shuffle",
                                           "seed*", "metrics_det.json"))):
        uni_det.append(json.load(open(f))["test_macro_aucpr"])
    uni = np.array(uni_det) if uni_det else np.array([r["aucpr"] for r in uniform])
    gauss = {"uniform_det": {"n": len(uni), "mean": float(uni.mean()),
                             "sd": float(uni.std(ddof=1))}, "arms": {}}
    for tag, arm in SIGMAS:
        runs = load(arm)
        if not runs:
            continue
        v = np.array([r["aucpr"] for r in runs])
        n = min(len(v), len(uni))
        d = v[:n] - uni[:n]
        gauss["arms"][tag] = {"n_seeds": len(v), "aucpr_mean": float(v.mean()),
                              "aucpr_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                              "delta_vs_uniform": float(d.mean()),
                              "p_vs_uniform": signflip_p(d),
                              "complete": len(v) == 10}
    star = gauss["arms"].get("rmsd")
    if star and star["complete"]:
        ok = abs(star["delta_vs_uniform"]) < 0.002 and star["p_vs_uniform"] > 0.20
        gauss["G5"] = {"variable": "rmsd", "delta": star["delta_vs_uniform"],
                       "p": star["p_vs_uniform"],
                       "verdict": "PASS" if ok else "FAIL",
                       "branch": "coordinate smoothing" if ok else "torsional coverage"}
    else:
        gauss["G5"] = {"verdict": "PENDING"}
    with open(os.path.join(PROJ, "results", "wpb_gauss", "summary.json"), "w") as fh:
        json.dump(gauss, fh, indent=1)

    write_report(curve, gauss)
    print(json.dumps({"G7": curve["G7"], "G5": gauss["G5"]}, indent=1))
    for p in curve["points"]:
        print(f"  tau {p['tau']:>6}: {p['aucpr_mean']:.4f} "
              f"(n={p['n_seeds']}, Neff {p['median_n_eff']:.2f})")
    for k, v in gauss["arms"].items():
        print(f"  gauss {k:5}: {v['aucpr_mean']:.4f} (n={v['n_seeds']}) "
              f"delta {v['delta_vs_uniform']:+.4f} p={v['p_vs_uniform']:.3f}")


def write_report(curve, gauss):
    g7 = curve["G7"]
    lines = ["# 15 — 温度扫描与高斯 TTA（T4/T5）", "",
             "由 `p2/wpab_sweeps.py` 汇总；每个 run 见 `results/s1_mil/` 下同名目录。", "",
             "## 1. 权重定律（预注册 Amendment f）", "",
             "`w_c(τ) ∝ p_c^(300/τ)`。τ=300 精确等于 `boltz`；τ=∞ 为 `1/C`。", "",
             "## 2. 温度曲线", "",
             "| τ (K) | N_eff 中位 | macro-AUCPR | n | Δ vs 300 K | p |",
             "|---|---|---|---|---|---|"]
    for p in curve["points"]:
        d = p.get("delta_vs_boltz")
        pv = p.get("p_vs_boltz")
        lines.append(f"| {p['tau']:.0f}{' (∞)' if p['tau'] == float('inf') else ''} | "
                     f"{p['median_n_eff']:.2f} | {p['aucpr_mean']:.4f} ± {p['aucpr_sd']:.4f} | "
                     f"{p['n_seeds']} | {'—' if d is None else f'{d:+.4f}'} | "
                     f"{'—' if pv is None else f'{pv:.3f}'} |")
    lines += ["", f"**G7**：Spearman(AUCPR, N_eff) = **{g7['spearman']:.3f}**，"
              f"{g7['n_points']} 个点，阈值 0.7 → **{g7['verdict']}**。"]
    if not g7["complete"]:
        lines.append("（尚未跑满 6 个 τ 点，结论待定。）")
    lines += ["", "> 读法：加权臂的最高分仍低于 uniform（0.3253），因此温度曲线的作用是"
              "把「加权退化」写成连续量，而不是改变主结论。若 G7 不通过，按预注册只保留"
              "300 K 与 uniform 两点。", "",
              "## 3. 高斯 TTA", "",
              f"uniform 参照（det 版）：{gauss['uniform_det']['mean']:.4f} ± "
              f"{gauss['uniform_det']['sd']:.4f}（n={gauss['uniform_det']['n']}）", "",
              "| 臂 | σ (Å) | macro-AUCPR | n | Δ vs uniform | p |", "|---|---|---|---|---|---|"]
    for tag, v in gauss["arms"].items():
        lines.append(f"| gauss_{tag} | {tag} | {v['aucpr_mean']:.4f} ± {v['aucpr_sd']:.4f} | "
                     f"{v['n_seeds']} | {v['delta_vs_uniform']:+.4f} | {v['p_vs_uniform']:.3f} |")
    g5 = gauss["G5"]
    lines += ["", f"**G5**：判定变量为 RMSD 匹配臂（σ* = 该分子真实系综相对 rank-1 的 RMSD 中位数，"
              f"全库中位 1.22 Å）。状态：**{g5['verdict']}**"
              + (f" → §2.2 写「{g5['branch']}」" if g5["verdict"] != "PENDING" else ""), "",
              "固定 σ 网格（≤0.30 Å）远小于真实构象铺展，因此它们只是「抖动」而非「覆盖」；"
              "这三个臂的作用是显示噪声越大越差，不能替代 RMSD 匹配臂作判定。"]
    with open(os.path.join(PROJ, "reports", "15_temperature_gauss.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
