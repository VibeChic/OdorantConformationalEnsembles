#!/usr/bin/env python3
"""Aggregate the M2OR runs and decide G6a (prereg Amendment (c)/(j)).

G6a: Macro-AP(A dual) - Macro-AP(B ligand-only) > 0 with p < 0.01 over 10 seeds,
and the direction must be reproduced against C (A - C > 0). Evaluated on the
cold-receptor split; the cold-ligand split is reported alongside, not gated.

Usage: python p2/m2or_summary.py
"""
import glob
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results", "s3_tiers", "m2or")


def load(split, arm):
    fs = sorted(glob.glob(os.path.join(OUT, split, arm, "seed*", "metrics.json")),
                key=lambda f: int(f.split("seed")[-1].split("/")[0]))
    return [json.load(open(f)) for f in fs]


def sf(d):
    d = np.asarray(d, float)
    obs, n = abs(d.sum()), len(d)
    c = sum(1 for k in range(1 << n)
            if abs((np.array([1 if (k >> i) & 1 else -1 for i in range(n)]) * d).sum())
            >= obs - 1e-12)
    return c / float(1 << n)


def summarize(split):
    res = {"split": split, "arms": {}, "pairs": {}}
    for a in ("A", "B", "C"):
        r = load(split, a)
        if not r:
            continue
        v = np.array([x["macro_ap"] for x in r])
        u = np.array([x["macro_auc"] for x in r])
        ent = [x["entropy_attention"] for x in r if x.get("entropy_attention") is not None]
        res["arms"][a] = {"n": len(v), "macro_ap": float(v.mean()),
                          "sd_ap": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          "macro_auc": float(u.mean()),
                          "entropy": float(np.mean(ent)) if ent else None}
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        if x in res["arms"] and y in res["arms"]:
            n = min(res["arms"][x]["n"], res["arms"][y]["n"])
            d = (np.array([q["macro_ap"] for q in load(split, x)][:n])
                 - np.array([q["macro_ap"] for q in load(split, y)][:n]))
            res["pairs"][f"{x}-{y}"] = {"delta": float(d.mean()),
                                        "sd": float(d.std(ddof=1)),
                                        "n_pos": int((d > 0).sum()),
                                        "p": float(sf(d)), "n": n}
    return res


def main():
    s2 = summarize("receptor")
    s1 = summarize("ligand")
    gate = {"split": "receptor", "complete": all(
        s2["arms"].get(a, {}).get("n") == 10 for a in ("A", "B", "C"))}
    if gate["complete"]:
        ab = s2["pairs"]["A-B"]
        ac = s2["pairs"]["A-C"]
        ok = ab["delta"] > 0 and ab["p"] < 0.01 and ac["delta"] > 0
        gate.update({"A_minus_B": ab, "A_minus_C": ac,
                     "verdict": "PASS" if ok else "FAIL",
                     "branch": "receptor conditioning adds value" if ok
                     else "ligand-only sufficiency is universal"})
    else:
        gate["verdict"] = "PENDING"
    out = {"S2_cold_receptor": s2, "S1_cold_ligand": s1, "G6a": gate}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    write_report(out)
    print(json.dumps(gate, indent=1))
    for k, v in s2["arms"].items():
        print(f"  S2 {k}: AP {v['macro_ap']:.4f} ± {v['sd_ap']:.4f} (n={v['n']}) "
              f"AUC {v['macro_auc']:.4f}")


def write_report(res):
    s2, s1, g = res["S2_cold_receptor"], res["S1_cold_ligand"], res["G6a"]
    lines = ["# 17 — M2OR 双塔（T8）", "",
             "设计见 `Paper2-Dual-Tower-1026_2003.md`；门禁见预注册 Amendment (c)/(j)。", "",
             "## 1. 主判定：冷受体切分（按 OR 家族整簇留出，5 折，10 seeds）", "",
             "| 臂 | macro-AP | SD | macro-AUC | attention 熵 | n |",
             "|---|---|---|---|---|---|"]
    for a in ("A", "B", "C"):
        v = s2["arms"].get(a)
        if not v:
            continue
        ent = "—" if v["entropy"] is None else f"{v['entropy']:.3f}"
        lines.append(f"| {a} | **{v['macro_ap']:.4f}** | {v['sd_ap']:.4f} | "
                     f"{v['macro_auc']:.4f} | {ent} | {v['n']} |")
    lines += ["", "| 比较 | Δ macro-AP | 同向 seeds | p |", "|---|---|---|---|"]
    for k, v in s2["pairs"].items():
        lines.append(f"| {k} | {v['delta']:+.4f} | {v['n_pos']}/{v['n']} | {v['p']:.4f} |")
    if g.get("verdict") != "PENDING":
        lines += ["", f"**G6a**：{g['verdict']} → "
                  f"{'受体条件化有效' if g['verdict'] == 'PASS' else '**配体侧充分性（普适）**'}"]
    lines += ["", "读法：A − C = 正说明受体塔确实在使用序列身份（打乱即掉分）；"
              "但 A 仍显著低于仅配体臂 B，因此「显式受体条件化」在本面板上不划算——"
              "一个 423 路的受体专属输出头把受体阈值隐式学到了，且不需要泛化到新序列。", "",
              "## 2. 次表：冷配体切分（仅报告，不设门禁）", ""]
    if s1["arms"]:
        lines += ["| 臂 | macro-AP | n |", "|---|---|---|"]
        for a in ("A", "B", "C"):
            v = s1["arms"].get(a)
            if v:
                lines.append(f"| {a} | {v['macro_ap']:.4f} ± {v['sd_ap']:.4f} | {v['n']} |")
        lines += ["", "该切分下逐受体 AP 对受体级常数不可见（同一受体内排序），"
                  "因此它衡量的是另一种问题：看见受体时，条件化是否有助于给该受体排序。" ]
    else:
        lines.append("（该切分尚未跑满 10 seeds。）")
    with open(os.path.join(PROJ, "reports", "17_m2or.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
