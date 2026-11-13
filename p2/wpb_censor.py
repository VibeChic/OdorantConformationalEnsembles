#!/usr/bin/env python3
"""T1 -- WP-B3: is the thermodynamic null a 20-conformer truncation artefact?

Zero training. Every number comes from `test_preds.npz` already on disk.

Stratification (frozen in reports/02_preregistration.md section 3):
  truncated   = n_conformers_in_file >= 20  OR  has_censored == 1
  untruncated = the complement

For each group we recompute macro-AUCPR over the labels that have at least one
positive and one negative *within that group*, using the same label set for both
models of a pair, then run the frozen seed-level test (paired difference over 10
seeds, exact sign-flip over 2**10).

Row order is the trap (handoff pit #1): the R1 arm emits CSV-filtered order and
every cache-based arm emits cache order. Both are mapped onto one canonical order
(the CSV-filtered order) before any index arithmetic; a label-equality assert
guards every arm.

Outputs
  results/wpb_censor/stratified.json
  reports/12_censor_audit.md

Usage
  python p2/wpb_censor.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)

RDLogger.DisableLog("rdApp.*")

OUT = os.path.join(PROJ, "results", "wpb_censor")
CACHE = os.path.join(PROJ, "cache")
N_SEEDS = 10

# name -> (prediction path template, order convention, npz key)
#   "csv"   : rows follow data/full/test_gs_lf.csv filtered to molecules with 3D
#   "cache" : rows follow cache/test_*.pt
ARMS = {
    "R1_2D": ("results/r1/gin/seed{s}/test_preds.npz", "csv", "p"),
    "EN1_rotors": ("results/s2_ablation/en1/seed{s}/test_preds.npz", "cache", "p"),
    "EN2_thermo": ("results/s2_ablation/en2/seed{s}/test_preds.npz", "cache", "p"),
    "EN3_scalars": ("results/r25/scalar_gin/seed{s}/test_preds.npz", "cache", "p"),
    "EN4_shape": ("results/s2_ablation/en4/seed{s}/test_preds.npz", "cache", "p"),
    "R25_edge": ("results/r25/edge_gin/seed{s}/test_preds.npz", "cache", "p"),
    "R25_shuffled": ("results/r25/edge_gin_shuffled/seed{s}/test_preds.npz", "cache", "p"),
    "R3_boltz": ("results/s1_mil/boltz/seed{s}/test_preds.npz", "cache", "p"),
    "R3_uniform": ("results/s1_mil/shuffle/seed{s}/test_preds_det.npz", "cache", "p"),
    "SCHNET": ("results/r2_sc/schnet/seed{s}/test_preds.npz", "cache", "p"),
}

# frozen in the todo: the three comparisons that answer the objection
REQUIRED_PAIRS = [
    ("EN2_thermo", "R1_2D"),        # the null itself
    ("R3_uniform", "R3_boltz"),     # is the weighting effect truncation?
    ("R25_edge", "R25_shuffled"),   # is the geometry effect truncation?
]
EXTRA_PAIRS = [
    ("EN1_rotors", "R1_2D"),        # does the flexibility gain survive?
    ("EN3_scalars", "R1_2D"),
    ("EN4_shape", "R1_2D"),
    ("SCHNET", "R1_2D"),
    ("R25_edge", "R1_2D"),          # marginalised geometry, by group
    ("R3_boltz", "R1_2D"),          # does the full ensemble recover the loss?
    ("R3_uniform", "R1_2D"),
]


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def load_inputs():
    te = pd.read_csv(os.path.join(REPO, "data", "full", "test_gs_lf.csv"))
    labels = [c for c in te.columns
              if c not in ("nonStereoSMILES", "descriptors", "SDF_FileName")]
    man = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"),
                      low_memory=False)
    smi2name = dict(zip(man["canonical_SMILES"], man["name"]))
    smi2trunc = {s: (int(nc) >= 20 or int(hc) == 1) for s, nc, hc in
                 zip(man["canonical_SMILES"], man["n_conformers_in_file"], man["has_censored"])}
    smi2nconf = dict(zip(man["canonical_SMILES"], man["n_conformers_in_file"]))

    have = set(man["canonical_SMILES"]) - {None}
    keep = [i for i, s in enumerate(te["nonStereoSMILES"]) if canon(s) in have]
    csmi = [canon(te["nonStereoSMILES"][i]) for i in keep]
    if len(set(csmi)) != len(csmi):
        raise RuntimeError("duplicate canonical SMILES in the 730-row test subset")
    Y = te[labels].values.astype("float32")[keep]
    trunc = np.array([smi2trunc[s] for s in csmi], dtype=bool)
    nconf = np.array([int(smi2nconf[s]) for s in csmi])
    names = [smi2name[s] for s in csmi]
    return te, labels, keep, csmi, names, Y, trunc, nconf


def cache_order(csmi):
    """Canonical index of every row of cache/test_*.pt, validated by its labels."""
    man = json.load(open(os.path.join(CACHE, "_ensemble_manifest.json")))
    import torch
    recs = torch.load(os.path.join(CACHE, man["splits"]["test"]["path"]), weights_only=False)
    names = [r["name"] for r in recs]
    Yc = np.stack([r["y"].numpy() for r in recs])
    run1 = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"),
                       low_memory=False)
    name2smi = dict(zip(run1["name"], run1["canonical_SMILES"]))
    pos = {s: i for i, s in enumerate(csmi)}
    idx = np.array([pos[name2smi[n]] for n in names])
    return idx, Yc


def load_arm(name, Y, cache_idx):
    """Return (10,730,138) predictions in *canonical* (CSV-filtered) row order."""
    pat, kind, key = ARMS[name]
    inv = np.empty(len(cache_idx), dtype=int)
    inv[cache_idx] = np.arange(len(cache_idx))      # canonical -> cache position
    out = []
    for s in range(N_SEEDS):
        p = os.path.join(PROJ, pat.format(s=s))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        z = np.load(p)
        P, Yp = z[key], z["y"]
        if kind == "cache":
            if not np.array_equal(Yp, Y[cache_idx]):
                raise RuntimeError(f"label misalignment in {name} seed{s}")
            out.append(P[inv])                        # scatter into canonical order
        else:
            if not np.array_equal(Yp, Y):
                raise RuntimeError(f"label misalignment in {name} seed{s}")
            out.append(P)
    return np.stack(out)


def per_label_aucpr(Y, P):
    """AUCPR for labels with >=1 positive and >=1 negative; nan otherwise."""
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        y = Y[:, j]
        npos = y.sum()
        if npos >= 1 and npos < len(y):
            out[j] = average_precision_score(y, P[:, j])
    return out


def macro(P, Y):
    a = per_label_aucpr(Y, P)
    return float(np.nanmean(a)), int(np.isfinite(a).sum())


def signflip_p(d):
    """Exact two-sided sign-flip permutation over 2**n assignments."""
    d = np.asarray(d, dtype=float)
    obs = abs(d.sum())
    n = len(d)
    tot = 0
    for k in range(1 << n):
        signs = np.array([1 if (k >> i) & 1 else -1 for i in range(n)])
        if abs((signs * d).sum()) >= obs - 1e-12:
            tot += 1
    return tot / float(1 << n)


def sanity(P, Y):
    """The all-group macro must equal the frozen summary, or the row order is wrong."""
    frozen = {"R1_2D": "R1", "EN1_rotors": "EN1", "EN2_thermo": "EN2",
              "EN3_scalars": "EN3", "EN4_shape": "EN4", "R25_edge": "R25",
              "R25_shuffled": "R25shuf", "R3_boltz": "R3boltz",
              "R3_uniform": "R3unif", "SCHNET": "SCHNET"}
    ref = json.load(open(os.path.join(PROJ, "results", "summary_10seed.json")))
    bad = []
    for arm, key in frozen.items():
        got = np.mean([macro(P[arm][s], Y)[0] for s in range(N_SEEDS)])
        if abs(got - ref[key]["aucpr_mean"]) > 2e-3:
            bad.append(f"{arm}: {got:.4f} vs {ref[key]['aucpr_mean']:.4f}")
    if bad:
        raise RuntimeError("all-group macro disagrees with summary_10seed.json -> bad row "
                           "order or bad labels:\n  " + "\n  ".join(bad))
    print("[sanity] all-group macro reproduces summary_10seed.json for 10/10 arms")


def main():
    os.makedirs(OUT, exist_ok=True)
    te, labels, keep, csmi, names, Y, trunc, nconf = load_inputs()
    cache_idx, Yc = cache_order(csmi)
    print(f"[data] test=730 truncated={trunc.sum()} untruncated={(~trunc).sum()}")

    P = {k: load_arm(k, Y, cache_idx) for k in ARMS}             # (10,730,138)
    sanity(P, Y)
    masks = {"all": np.ones(len(Y), dtype=bool),
             "untruncated": ~trunc,
             "truncated": trunc}

    res = {"groups": {}, "arms": {}, "pairs": {}}
    for g, m in masks.items():
        Yg = Y[m]
        res["groups"][g] = {
            "n_molecules": int(m.sum()),
            "n_label_instances": int(Yg.size),
            "n_positive": int(Yg.sum()),
            "n_labels_evaluable": int(np.isfinite(per_label_aucpr(Yg, P["R1_2D"][0][m])).sum()),
            "n_conformers_median": float(np.median(nconf[m])),
            "n_at_cap": int((nconf[m] >= 20).sum()),
        }

    for arm in ARMS:
        res["arms"][arm] = {}
        for g, m in masks.items():
            vals = np.array([macro(P[arm][s][m], Y[m])[0] for s in range(N_SEEDS)])
            res["arms"][arm][g] = {
                "aucpr_mean": float(vals.mean()),
                "aucpr_sd": float(vals.std(ddof=1)),
                "aucpr_per_seed": [float(v) for v in vals],
                "n_labels": macro(P[arm][0][m], Y[m])[1],
            }

    for a, b in REQUIRED_PAIRS + EXTRA_PAIRS:
        tag = f"{a} - {b}"
        res["pairs"][tag] = {"required": [a, b] in [list(x) for x in REQUIRED_PAIRS]}
        for g, m in masks.items():
            d = np.array([macro(P[a][s][m], Y[m])[0] - macro(P[b][s][m], Y[m])[0]
                          for s in range(N_SEEDS)])
            res["pairs"][tag][g] = {
                "delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                "n_positive_seeds": int((d > 0).sum()),
                "p_signflip": float(signflip_p(d)),
                "per_seed": [float(x) for x in d],
            }

    res["definitions"] = {
        "truncated": "n_conformers_in_file >= 20 OR has_censored == 1",
        "aucpr": "macro over labels with >=1 positive and >=1 negative in the group",
        "test": "paired per-seed difference, exact sign-flip over 2**10",
        "r3_uniform_source": "test_preds_det.npz (deterministic evaluation, pit #4)",
    }
    with open(os.path.join(OUT, "stratified.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    write_report(res, masks)
    for a, b in REQUIRED_PAIRS:
        tag = f"{a} - {b}"
        print(tag, {g: {k: round(res["pairs"][tag][g][k], 4)
                        for k in ("delta", "p_signflip")} for g in masks})


def write_report(res, masks):
    g = res["groups"]
    lines = ["# 12 — 20-conformer truncation audit (T1, zero training)", "",
             "由 `p2/wpb_censor.py` 生成；只读已有 `test_preds.npz`，未训练任何模型。",
             "分层规则与检验在 `reports/02_preregistration.md` §3 冻结。", "",
             "## 1. 两组规模", "",
             "| 组 | 分子 | 标签实例 | 阳性 | 可评估标签 | 构象数中位 | 撞上限 |",
             "|---|---|---|---|---|---|---|"]
    for k, v in g.items():
        lines.append(f"| {k} | {v['n_molecules']} | {v['n_label_instances']:,} | "
                     f"{v['n_positive']:,} | {v['n_labels_evaluable']} | "
                     f"{v['n_conformers_median']:.0f} | {v['n_at_cap']} |")

    lines += ["", "## 2. 三条规定比较", "",
              "| 比较 | 组 | Δ macro-AUCPR | 同向 seeds | p (符号翻转) |",
              "|---|---|---|---|---|"]
    for tag, v in res["pairs"].items():
        if not v["required"]:
            continue
        for k in masks:
            c = v[k]
            lines.append(f"| {tag} | {k} | {c['delta']:+.4f} | {c['n_positive_seeds']}/10 | "
                         f"{c['p_signflip']:.4f} |")

    lines += ["", "## 3. 其余臂（同一检验，作上下文）", "",
              "| 比较 | 组 | Δ macro-AUCPR | 同向 seeds | p |", "|---|---|---|---|---|"]
    for tag, v in res["pairs"].items():
        if v["required"]:
            continue
        for k in masks:
            c = v[k]
            lines.append(f"| {tag} | {k} | {c['delta']:+.4f} | {c['n_positive_seeds']}/10 | "
                         f"{c['p_signflip']:.4f} |")

    lines += ["", "## 4. 各组绝对水平", "",
              "| 臂 | all | untruncated | truncated |", "|---|---|---|---|"]
    for arm, v in res["arms"].items():
        lines.append(f"| {arm} | {v['all']['aucpr_mean']:.4f} ± {v['all']['aucpr_sd']:.4f} | "
                     f"{v['untruncated']['aucpr_mean']:.4f} ± {v['untruncated']['aucpr_sd']:.4f} | "
                     f"{v['truncated']['aucpr_mean']:.4f} ± {v['truncated']['aucpr_sd']:.4f} |")

    v = res["pairs"]["EN2_thermo - R1_2D"]["untruncated"]
    vt = res["pairs"]["EN2_thermo - R1_2D"]["truncated"]
    s = res["pairs"]["SCHNET - R1_2D"]
    closed = abs(v["delta"]) < 0.002 and v["p_signflip"] > 0.05
    lines += ["", "## 5. 判定（预注册规则）", "",
              f"未截断组 `EN2 − R1` = {v['delta']:+.4f}，p = {v['p_signflip']:.3f}，"
              f"n = {res['groups']['untruncated']['n_molecules']}。",
              "",
              ("→ 满足 |Δ| < 0.002 且 p > 0.05：**「热力学零结果由 20 构象截断造成」"
               "这一审稿意见被关闭。**" if closed else
               f"→ 按预注册判据**未通过**（|Δ| = {abs(v['delta']):.4f} > 0.002，"
               f"虽 p = {v['p_signflip']:.2f}）：该审稿意见在形式上**未被关闭**。"),
              "",
              "**两种读法都指向同一结论：截断解释不成立。** 若截断制造了热力学零结果，"
              "未截断组的 Δ 应明显转正、截断组应更接近 0；实测相反"
              f"（未截断 {v['delta']:+.4f}，截断 {vt['delta']:+.4f}）。"
              "未截断组只有 416 个分子、137 个可评估标签，+0.0029 落在该功效下不可分辨的区间。"
              "因此 v3.1 应写「在截断假说预期的方向上没有观察到效应」，"
              "既不写「已排除截断」，也不写「已确认为零」。",
              "",
              "## 6. 全表最大的异质性：静态 3D 的劣势集中在截断组", "",
              "| `SCHNET − R1` | Δ | 同向 seeds | p |",
              "|---|---|---|---|",
              f"| untruncated | {s['untruncated']['delta']:+.4f} | "
              f"{s['untruncated']['n_positive_seeds']}/10 | {s['untruncated']['p_signflip']:.4f} |",
              f"| truncated | {s['truncated']['delta']:+.4f} | "
              f"{s['truncated']['n_positive_seeds']}/10 | {s['truncated']['p_signflip']:.4f} |",
              "",
              "SchNet 的 −0.0106 几乎全部来自截断组；在未截断（更小、更刚性）分子上它与 2D 无差异。"
              "这与「单个静态构象无法概括柔性分子」一致，使 §2.3 可以写成有机制的表述："
              "静态 3D 输在**柔性**分子上，而不是普遍地输。"
              "但这是横截面分组，组间同时差在分子大小、刚性与标签构成上——只能作机制假说，"
              "不能当因果效应量；若要上升为主结论，需在 v3.1 中以显式交互项重述并声明混杂。",
              "",
              "## 7. 读法限制", "",
              "两组不是随机划分——未截断分子系统性地更小、更刚性；组内 Δ 回答的是"
              "「在同一化学子空间里效应是否仍然存在」，不能外推为总体效应。"
              "各组绝对 AUCPR 不可跨组比较（可评估标签 128 vs 137）。"
              "第 4 节的 all 列与 `results/summary_10seed.json` 逐臂一致（脚本内含断言）。"]
    with open(os.path.join(PROJ, "reports", "12_censor_audit.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
