#!/usr/bin/env python3
"""Aggregate the DREAM runs: summary.json + reports/16_dream.md.

Primary metric (prereg Amendment (i)): mean over the 21 targets of each target's
pooled out-of-fold Pearson r. The single pooled-over-all r is written too, but is
reported as a diagnostic only -- it is dominated by between-target variance and is
0.860 for every arm including Morgan.

Usage: python p2/dream_summary.py
"""
import glob
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "s3_tiers", "dream")
ARMS = ["r1", "en1", "en6", "morgan"]
PAIRS = [("en1", "r1"), ("en6", "en1"), ("en6", "r1"), ("r1", "morgan"),
         ("en1", "morgan")]


def load(arm):
    fs = sorted(glob.glob(os.path.join(OUT, arm, "seed*", "metrics.json")),
                key=lambda f: int(f.split("seed")[-1].split("/")[0]))
    return [json.load(open(f)) for f in fs]


def sf(d):
    d = np.asarray(d, float)
    obs, n = abs(d.sum()), len(d)
    c = sum(1 for k in range(1 << n)
            if abs((np.array([1 if (k >> i) & 1 else -1 for i in range(n)]) * d).sum())
            >= obs - 1e-12)
    return c / float(1 << n)


def main():
    A = {a: load(a) for a in ARMS}
    res = {"arms": {}, "pairs": {}, "n": {a: len(A[a]) for a in ARMS}}
    for a in ARMS:
        v = np.array([x["mean_r"] for x in A[a]])
        res["arms"][a] = {"mean_r": float(v.mean()),
                          "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          "pooled_all": float(np.mean([x["pooled_r_all"] for x in A[a]])),
                          "mean_spearman": float(np.mean([x["mean_spearman"] for x in A[a]]))}
    for a, b in PAIRS:
        d = np.array([x["mean_r"] for x in A[a]]) - np.array([x["mean_r"] for x in A[b]])
        res["pairs"][f"{a}-{b}"] = {"delta": float(d.mean()),
                                    "sd": float(d.std(ddof=1)),
                                    "n_pos": int((d > 0).sum()),
                                    "p": float(sf(d))}
    t = list(A["r1"][0]["per_target_r"])
    res["per_target_delta_en1_r1"] = {
        c: float(np.mean([x["per_target_r"][c] for x in A["en1"]])
                 - np.mean([x["per_target_r"][c] for x in A["r1"]])) for c in t}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_report(res)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                      for k, v in res["pairs"].items()}, indent=1))


def write_report(res):
    a = res["arms"]
    p = res["pairs"]
    dt = sorted(res["per_target_delta_en1_r1"].items(), key=lambda kv: -kv[1])
    lines = ["# 16 — DREAM 层（T7，338 分子 × 21 目标，主表仅 2D）", "",
             "由 `p2/train_dream.py` + `p2/dream_summary.py` 生成。按 A-2，主表只用 2D 臂；"
             "3D 臂需精确 InChIKey join（304/338），未在本表运行。", "",
             "## 1. 主结果（per-target pooled OOF r 的 21 目标平均）", "",
             "| 臂 | mean r | SD | Spearman | 全块 pooled r（仅诊断） | n |",
             "|---|---|---|---|---|---|"]
    for k in ARMS:
        v = a[k]
        lines.append(f"| {k} | **{v['mean_r']:.3f}** | {v['sd']:.3f} | "
                     f"{v['mean_spearman']:.3f} | {v['pooled_all']:.3f} | {res['n'][k]} |")
    lines += ["", "全块 pooled r 对每个臂都是 0.860——它几乎全部来自 21 个目标之间的均值差异，"
              "不是预测能力，因此只作诊断，不用于比较（预注册 Amendment i）。", "",
              "## 2. 配对检验（10 seeds，符号翻转）", "",
              "| 比较 | Δ mean r | SD | 同向 seeds | p |", "|---|---|---|---|---|"]
    for k, v in p.items():
        lines.append(f"| {k} | {v['delta']:+.4f} | {v['sd']:.4f} | {v['n_pos']}/10 | {v['p']:.4f} |")
    lines += ["", "## 3. 逐目标（E-N1 − R1，只列两端）", "",
              "| 目标 | Δr |", "|---|---|"]
    for k, v in dt[:5] + dt[-3:]:
        lines.append(f"| {k} | {v:+.3f} |")
    lines += ["", "## 4. 读法", "",
              "- 在 DREAM 层，指纹 + Ridge 与最好的 GNN 打平（E-N1 − Morgan = "
              f"{p['en1-morgan']['delta']:+.4f}, p={p['en1-morgan']['p']:.2f}），"
              "而裸 GNN 明显低于 Morgan "
              f"（R1 − Morgan = {p['r1-morgan']['delta']:+.4f}, p={p['r1-morgan']['p']:.3f}）。",
              "- `n_rotors` 把 GNN 从落后拉到打平（+0.0098，p=0.13，未达显著），"
              "环柔性指数没有再增加（en6 − en1 = −0.0055）。方向与 GS-LF 层一致。",
              "- 按 A-1，这一层只是**次要检验**：不进主结论、不叫 G6。它支持的是"
              "「配体侧信息已足够」的复现，而不是新的门禁证据。"]
    with open(os.path.join(PROJ, "reports", "16_dream.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
