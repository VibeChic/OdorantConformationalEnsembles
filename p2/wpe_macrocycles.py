#!/usr/bin/env python3
"""T2 / WP-E -- macrocycle strata (descriptive) + the en6 arm test.

Two parts, deliberately separated (reports/02_preregistration.md, Amendment (d)):

  (a) descriptive -- three strata, group n, mean macro-AUCPR per arm, mean difference
      vs R1, **no p-value**. The flexible-macrocycle stratum has 48 molecules in the
      whole library and 7 in the test split; a p-value there would describe seven
      molecules.
  (b) en6 -- GIN + [n_rotors, n_ring_flex], 10 seeds, paired sign-flip test against
      en1 and en4 only. This is an arm comparison, not a subgroup claim.

Outputs
  results/wpe_macrocycles/strata.json
  reports/13_macrocycles.md

Usage
  python p2/wpe_macrocycles.py
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
OUT = os.path.join(PROJ, "results", "wpe_macrocycles")
CACHE = os.path.join(PROJ, "cache")
N_SEEDS = 10

ARMS = {
    "R1_2D": ("results/r1/gin/seed{s}/test_preds.npz", "csv"),
    "EN1_rotors": ("results/s2_ablation/en1/seed{s}/test_preds.npz", "cache"),
    "EN4_shape": ("results/s2_ablation/en4/seed{s}/test_preds.npz", "cache"),
    "EN6_ring": ("results/s2_ablation/en6/seed{s}/test_preds.npz", "cache"),
}


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def strata_table():
    d = pd.read_csv(os.path.join(CACHE, "exp2_scalars_v2.csv"))
    rigid = (d.n_rotors <= 1) & (d.n_eff < 1.5)
    macro = (d.n_rotors <= 1) & (d.max_ring_size >= 7) & (d.n_eff > 3)
    rest = ~(rigid | macro)
    d["stratum"] = np.select([rigid, macro], ["rigid", "flexible_macrocycle"], "rest")
    return d


def load_aligned():
    te = pd.read_csv(os.path.join(REPO, "data", "full", "test_gs_lf.csv"))
    labels = [c for c in te.columns
              if c not in ("nonStereoSMILES", "descriptors", "SDF_FileName")]
    man = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"),
                      low_memory=False)
    man["csmi"] = man.canonical_SMILES.map(canon)
    smi2name = dict(zip(man.csmi, man.name))
    have = set(man.csmi)
    keep = [i for i, s in enumerate(te.nonStereoSMILES) if canon(s) in have]
    csmi = [canon(te.nonStereoSMILES[i]) for i in keep]
    names = [smi2name[s] for s in csmi]
    Y = te[labels].values.astype("float32")[keep]

    import torch
    m = json.load(open(os.path.join(CACHE, "_ensemble_manifest.json")))
    recs = torch.load(os.path.join(CACHE, m["splits"]["test"]["path"]), weights_only=False)
    cache_names = [r["name"] for r in recs]
    Yc = np.stack([r["y"].numpy() for r in recs])
    pos = {n: i for i, n in enumerate(names)}
    scatter = np.array([pos[n] for n in cache_names])          # canonical <- cache
    inv = np.empty(len(scatter), dtype=int)
    inv[scatter] = np.arange(len(scatter))

    P = {}
    for arm, (pat, kind) in ARMS.items():
        ps = []
        for s in range(N_SEEDS):
            z = np.load(os.path.join(PROJ, pat.format(s=s)))
            if kind == "csv":
                if not np.array_equal(z["y"], Y):
                    raise RuntimeError(f"{arm} label misalignment")
                ps.append(z["p"])
            else:
                if not np.array_equal(z["y"], Yc):
                    raise RuntimeError(f"{arm} cache label misalignment")
                ps.append(z["p"][inv])
        P[arm] = np.stack(ps)
    return labels, names, Y, P


def macro_aucpr(P, Y):
    vals = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() >= 1 and y.sum() < len(y):
            vals.append(average_precision_score(y, P[:, j]))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def signflip_p(d):
    d = np.asarray(d, float)
    obs, n = abs(d.sum()), len(d)
    c = sum(1 for k in range(1 << n)
            if abs((np.array([1 if (k >> i) & 1 else -1 for i in range(n)]) * d).sum())
            >= obs - 1e-12)
    return c / float(1 << n)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = strata_table()
    labels, names, Y, P = load_aligned()
    stratum = d.set_index("name").loc[names, "stratum"].values
    n_ring_flex = d.set_index("name").loc[names, "n_ring_flex"].values

    res = {"library_counts": d.stratum.value_counts().to_dict(),
           "test_counts": {k: int((stratum == k).sum()) for k in
                           ("rigid", "flexible_macrocycle", "rest")},
           "descriptive": {}, "en6_tests": {}}

    for g in ("all", "rigid", "flexible_macrocycle", "rest"):
        m = np.ones(len(Y), bool) if g == "all" else (stratum == g)
        block = {"n_molecules": int(m.sum()),
                 "n_positive": int(Y[m].sum()),
                 "arms": {}}
        for arm in ARMS:
            vals, nl = [], None
            for s in range(N_SEEDS):
                v, nl = macro_aucpr(P[arm][s][m], Y[m])
                vals.append(v)
            block["arms"][arm] = {"aucpr_mean": float(np.mean(vals)),
                                  "aucpr_sd": float(np.std(vals, ddof=1)),
                                  "n_labels": nl,
                                  "delta_vs_R1": float(np.mean(vals) -
                                                       np.mean([macro_aucpr(P["R1_2D"][s][m],
                                                                            Y[m])[0]
                                                                for s in range(N_SEEDS)]))}
        res["descriptive"][g] = block

    for a in ("EN1_rotors", "EN4_shape"):
        dd = np.array([macro_aucpr(P["EN6_ring"][s], Y)[0] - macro_aucpr(P[a][s], Y)[0]
                       for s in range(N_SEEDS)])
        res["en6_tests"][f"EN6_ring - {a}"] = {
            "delta": float(dd.mean()), "sd": float(dd.std(ddof=1)),
            "n_positive_seeds": int((dd > 0).sum()), "p_signflip": signflip_p(dd),
            "per_seed": dd.tolist()}

    en6 = np.array([macro_aucpr(P["EN6_ring"][s], Y)[0] for s in range(N_SEEDS)])
    res["en6_overall"] = {"aucpr_mean": float(en6.mean()), "aucpr_sd": float(en6.std(ddof=1)),
                          "n_positive_with_ring_flex": int((n_ring_flex > 0).sum())}

    with open(os.path.join(OUT, "strata.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_report(res)
    print(json.dumps({"test_counts": res["test_counts"],
                      "library": res["library_counts"],
                      "en6": res["en6_overall"],
                      "tests": {k: {kk: round(vv, 4) for kk, vv in v.items()
                                    if kk in ("delta", "p_signflip")}
                                for k, v in res["en6_tests"].items()}}, indent=1))


def write_report(res):
    L = res["library_counts"]
    T = res["test_counts"]
    e = res["en6_overall"]
    lines = ["# 13 — 大环/柔性环分层与 en6（T2，A4 = 描述性）", "",
             "由 `p2/wpe_macrocycles.py` 生成。分层规则与「只描述、不设显著性」在 "
             "`reports/02_preregistration.md` Amendment (d) 冻结。", "",
             "## 1. 子空间规模", "",
             "| 层 | 定义 | 全库 | test |", "|---|---|---|---|",
             f"| 刚性笼/芳环 | `n_rotors ≤ 1`, `n_eff < 1.5` | {L.get('rigid',0)} | {T['rigid']} |",
             f"| 柔性大环 | `n_rotors ≤ 1`, `max_ring ≥ 7`, `n_eff > 3` | "
             f"{L.get('flexible_macrocycle',0)} | {T['flexible_macrocycle']} |",
             f"| 其余 | — | {L.get('rest',0)} | {T['rest']} |",
             "",
             "**结论性的规模事实：柔性大环全库 48 个、测试集 7 个；整个测试集里 `n_ring_flex > 0` 的"
             f"分子总共只有 {e['n_positive_with_ring_flex']} 个。** 这不足以支撑任何「大环例外」主张，"
             "因此本节没有 p 值，也不应被引用为效应。",
             "",
             "注：交接文档里给出的 513 / 42 / 339 是用 `n_rotors = 0` 算的，与它自己写的规则"
             "（`n_rotors ≤ 1`）不一致；按规则重算为 "
             f"{L.get('rigid',0)} / {L.get('flexible_macrocycle',0)} / {L.get('rest',0)}。"
             "本报告一律用规则版本。",
             "",
             "## 2. 各层描述性均值（macro-AUCPR，10 seeds 均值；无 p 值）", "",
             "| 层 | n | 阳性 | R1 | E-N1 | E-N4 | E-N6 | E-N6 − R1 |",
             "|---|---|---|---|---|---|---|---|"]
    for g in ("all", "rigid", "flexible_macrocycle", "rest"):
        b = res["descriptive"][g]
        a = b["arms"]
        lines.append(f"| {g} | {b['n_molecules']} | {b['n_positive']} | "
                     f"{a['R1_2D']['aucpr_mean']:.3f} | {a['EN1_rotors']['aucpr_mean']:.3f} | "
                     f"{a['EN4_shape']['aucpr_mean']:.3f} | {a['EN6_ring']['aucpr_mean']:.3f} | "
                     f"{a['EN6_ring']['delta_vs_R1']:+.3f} |")
    lines += ["", "⚠ 大环那一行只有 12 个可评估标签、19 个阳性，其 0.53 的绝对水平来自标签集变小，"
              "**不可与其它层比较、也不可解释为效应**，只能作为「该子空间规模不足」的证据。",
              "", "各层可评估标签数：" + "，".join(
        f"{g} {res['descriptive'][g]['arms']['R1_2D']['n_labels']}" for g in
        ("all", "rigid", "flexible_macrocycle", "rest")) + "。",
        "",
        "## 3. en6 臂（这是臂比较，按 A6 有 p 值）", "",
        f"`en6` = GIN + `[n_rotors, n_ring_flex]`，10 seeds，macro-AUCPR "
        f"**{e['aucpr_mean']:.4f} ± {e['aucpr_sd']:.4f}**。", "",
        "| 比较 | Δ | 同向 seeds | p |", "|---|---|---|---|"]
    for k, v in res["en6_tests"].items():
        lines.append(f"| {k} | {v['delta']:+.4f} | {v['n_positive_seeds']}/10 | "
                     f"{v['p_signflip']:.4f} |")
    lines += ["", "判定（预注册）：若 `en6 − en1` 不显著，v3.1 写「二维环柔性指数不能替代形状标量」；"
              "若显著为正，也只能写成「环柔性提供了一个额外、但很小的自由度」，不得升级为机制主张。"]
    with open(os.path.join(PROJ, "reports", "13_macrocycles.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
