#!/usr/bin/env python3
"""Exp 3.1 — zero-training analyses on existing predictions.

Two analyses, neither of which trains anything:

  (a) per-descriptor decomposition
      For each of the 138 labels: AUCPR(model) - AUCPR(R1), correlated with
      label prevalence and with intra-label chemical diversity (1 - mean
      pairwise Morgan Tanimoto). Tests whether 3D/flexibility gains concentrate
      on shape-convergent, well-powered descriptors.

  (b) MMP odour cliffs, test split only
      The 2,773-pair table is 90.6% training molecules, so only the 116 test
      pairs are used. For each pair, the flipped descriptor's true labels give
      the direction; we measure sign accuracy and the predicted separation
      |p_a - p_b| for every model.

Outputs
  results/s3_tiers/exp31_subgroups.json
  reports/09_exp31_subgroups_mmp.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(PROJ, "cache")
OUT = os.path.join(PROJ, "results", "s3_tiers")
# order_kind:
#   "csv"   -> rows of data/full/test_gs_lf.csv filtered to molecules with 3D,
#              in file order (this is what train_r1.py / train_r25.py emit)
#   "cache" -> order of cache/test_heavy.pt (what train_r2.py emits)
MODELS = {
    "r1_gin": ("results/r1/gin/seed{s}/test_preds.npz", "csv"),
    "en3_scalar": ("results/r25/scalar_gin/seed{s}/test_preds.npz", "cache"),
    "r25_edge": ("results/r25/edge_gin/seed{s}/test_preds.npz", "cache"),
    "r25_shuffled": ("results/r25/edge_gin_shuffled/seed{s}/test_preds.npz", "cache"),
    "r2_schnet": ("results/r2_sc/schnet/seed{s}/test_preds.npz", "cache"),
    "r2_chiral_schnet": ("results/r2_sc/chiral_schnet/seed{s}/test_preds.npz", "cache"),
    "r2_angle_mpnn": ("results/r2/angle_mpnn/seed{s}/test_preds.npz", "cache"),
}


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def label_names():
    te = pd.read_csv(os.path.join(REPO, "data", "full", "test_gs_lf.csv"))
    return [c for c in te.columns
            if c not in ("nonStereoSMILES", "descriptors", "SDF_FileName")], te


def have3d_smiles():
    run1 = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"))
    return set(run1["canonical_SMILES"].map(canon)) - {None}


def cache_names():
    man = json.load(open(os.path.join(CACHE, "_ensemble_manifest.json")))
    recs = torch_load(os.path.join(CACHE, man["splits"]["test"]["path"]))
    return [r["name"] for r in recs], np.stack([r["y"].numpy() for r in recs])


def order_indices(kind, te, labels, smi2row):
    """Return, for one prediction convention, the CSV row index of every row."""
    if kind == "csv":
        have = have3d_smiles()
        keep = [i for i, s in enumerate(te["nonStereoSMILES"]) if canon(s) in have]
        return np.array(keep)
    names, Yc = cache_names()
    run1 = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"))
    name2smi = dict(zip(run1["name"], run1["canonical_SMILES"]))
    idx = np.array([smi2row[canon(name2smi[n])] for n in names])
    if not np.array_equal(te[labels].values.astype("float32")[idx], Yc):
        raise RuntimeError("cache order does not match its stored labels")
    return idx


def torch_load(path):
    import torch
    return torch.load(path, weights_only=False)


def load_preds(pattern, n_seeds=3):
    ps, ys = [], None
    for s in range(n_seeds):
        p = os.path.join(PROJ, pattern.format(s=s))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        z = np.load(p)
        ps.append(z["p"])
        ys = z["y"] if ys is None else ys
    return np.mean(ps, axis=0), ys


def per_label_aucpr(Y, P):
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() >= 1:
            out[j] = average_precision_score(Y[:, j], P[:, j])
    return out


def label_diversity(labels, table):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in table["nonStereoSMILES"]:
        m = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(m) if m is not None else None)
    Y = table[labels].values
    div = np.full(len(labels), np.nan)
    for j in range(len(labels)):
        idx = np.where(Y[:, j] > 0)[0]
        if len(idx) < 5:
            continue
        sims = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                fa, fb = fps[idx[a]], fps[idx[b]]
                if fa is None or fb is None:
                    continue
                sims.append(DataStructs.TanimotoSimilarity(fa, fb))
        if sims:
            div[j] = 1.0 - float(np.mean(sims))
    return div


def mmp_analysis(models, orders):
    mmp = pd.read_csv(os.path.join(REPO, "results", "mmp_cliff_stratification.csv"))
    mmp = mmp[mmp.split == "test"].reset_index(drop=True)
    te = pd.read_csv(os.path.join(REPO, "data", "full", "test_gs_lf.csv"))
    labels = [c for c in te.columns
              if c not in ("nonStereoSMILES", "descriptors", "SDF_FileName")]
    smi2row = {canon(s): i for i, s in enumerate(te["nonStereoSMILES"])}
    Ycsv = te[labels].values

    rows = []
    for _, r in mmp.iterrows():
        d = r["descriptor"]
        ia, ib = smi2row.get(canon(r["smiles_a"])), smi2row.get(canon(r["smiles_b"]))
        if ia is None or ib is None or d not in labels:
            continue
        j = labels.index(d)
        ya, yb = Ycsv[ia, j], Ycsv[ib, j]
        if ya == yb:
            continue
        rows.append({"descriptor": d, "ia": int(ia), "ib": int(ib),
                     "sign": 1 if ya > yb else -1})
    used = pd.DataFrame(rows)
    res = {}
    for name, (P, _) in models.items():
        # translate csv-canonical pair positions into this model's own row order
        inv = {int(c): k for k, c in enumerate(orders[name])}
        ja = np.array([inv[int(c)] for c in used["ia"]])
        jb = np.array([inv[int(c)] for c in used["ib"]])
        cols = [labels.index(x) for x in used["descriptor"]]
        d = P[ja, cols] - P[jb, cols]
        correct = (np.sign(d) == used["sign"].values).astype(int)
        res[name] = {"n_pairs": int(len(used)),
                     "sign_accuracy": float(correct.mean()),
                     "mean_abs_separation": float(np.abs(d).mean()),
                     "median_abs_separation": float(np.median(np.abs(d))),
                     "_correct": correct.tolist()}
    # McNemar exact (binomial on the discordant pairs), vs r1_gin
    from scipy.stats import binomtest
    base = np.array(res["r1_gin"]["_correct"])
    for name, v in res.items():
        if name == "r1_gin":
            continue
        c = np.array(v["_correct"])
        b = int(((base == 1) & (c == 0)).sum())
        cc = int(((base == 0) & (c == 1)).sum())
        v["mcnemar_vs_r1"] = {"b_r1_only": b, "c_model_only": cc,
                              "p_exact": float(binomtest(cc, b + cc, 0.5).pvalue)
                              if b + cc else 1.0}
    return res, int(len(used)), int(len(mmp))


def main():
    os.makedirs(OUT, exist_ok=True)
    labels, te = label_names()

    smi2row = {canon(s): i for i, s in enumerate(te["nonStereoSMILES"])}
    Ycsv = te[labels].values.astype("float32")
    orders = {k: order_indices(kind, te, labels, smi2row)
              for k, (pat, kind) in MODELS.items()}

    models = {}
    for name, (pat, kind) in MODELS.items():
        P, Y = load_preds(pat)
        if not np.array_equal(Y, Ycsv[orders[name]]):
            raise RuntimeError(f"label misalignment for {name}")
        models[name] = (P, Y)

    auc = {k: per_label_aucpr(Y, P) for k, (P, Y) in models.items()}
    prev = Ycsv.mean(0)
    div = label_diversity(labels, te)

    base = auc["r1_gin"]
    delta = {k: (v - base) for k, v in auc.items() if k != "r1_gin"}
    ok = np.isfinite(base)

    corr = {}
    for k, dv in delta.items():
        m = ok & np.isfinite(dv)
        md = m & np.isfinite(div)
        corr[k] = {
            "pearson_vs_log_prevalence": float(np.corrcoef(np.log(prev[m] + 1e-4), dv[m])[0, 1]),
            "spearman_vs_log_prevalence": float(pd.Series(dv[m]).corr(
                pd.Series(np.log(prev[m] + 1e-4)), method="spearman")),
            "pearson_vs_diversity": (float(np.corrcoef(div[md], dv[md])[0, 1])
                                     if md.sum() > 3 else float("nan")),
            "n_labels": int(m.sum()),
            "n_diversity": int(md.sum()),
            "mean_delta": float(dv[m].mean()),
        }

    mmp_res, n_used, n_total = mmp_analysis(models, orders)

    # descriptor table for the report
    rows = []
    for j, lab in enumerate(labels):
        if not np.isfinite(base[j]):
            continue
        rows.append({"label": lab, "prevalence": float(prev[j]),
                     "diversity": float(div[j]) if np.isfinite(div[j]) else None,
                     "aucpr_r1": float(base[j]),
                     "delta_en3": float(delta["en3_scalar"][j]),
                     "delta_r25": float(delta["r25_edge"][j]),
                     "delta_schnet": float(delta["r2_schnet"][j])})
    tab = pd.DataFrame(rows).sort_values("prevalence", ascending=False)

    out = {"correlations": corr, "mmp": mmp_res,
           "mmp_pairs_used": n_used, "mmp_pairs_test": n_total,
           "per_descriptor": tab.to_dict(orient="records")}
    with open(os.path.join(OUT, "exp31_subgroups.json"), "w") as f:
        json.dump(out, f, indent=1)

    write_report(corr, mmp_res, n_used, n_total, tab)
    print(json.dumps({"corr": corr, "mmp_used": n_used}, indent=1))
    print(tab.head(10).to_string(index=False))


def write_report(corr, mmp, n_used, n_total, tab):
    lines = ["# Exp 3.1 — 子群分解与 MMP 悬崖（零训练）", "",
             "由 `p2/exp31_subgroups.py` 生成；数据源为已有 `test_preds.npz`，未训练任何模型。", "",
             "## 1. 每个描述符的 ΔAUCPR 与标签属性的关系", "",
             "| 模型 vs R1 | n labels | mean Δ | Pearson vs log(prevalence) | Spearman | Pearson vs diversity |",
             "|---|---|---|---|---|---|"]
    for k, v in corr.items():
        lines.append(f"| {k} | {v['n_labels']} | {v['mean_delta']:+.4f} | "
                     f"{v['pearson_vs_log_prevalence']:+.3f} | "
                     f"{v['spearman_vs_log_prevalence']:+.3f} | "
                     f"{v['pearson_vs_diversity']:+.3f} |")
    lines += ["", "> 解读规则（预注册）：若 3D/柔性增益集中在**低流行率**或**低多样性**标签上，"
              "则支持\"分辨率不足\"假说；若集中在高多样性（形状收敛）标签上，则支持\"几何有用但被平均掉\"。",
              "", "## 2. MMP 气味悬崖（仅 test 116 对）", "",
              f"可用对：**{n_used} / {n_total}**（两侧分子都在 730 个有 3D 的测试分子内，"
              "且翻转描述符的标签确实不同）", "",
              "| 模型 | sign accuracy | mean \\|Δp\\| | median \\|Δp\\| | McNemar b/c | p |",
              "|---|---|---|---|---|---|"]
    for k, v in mmp.items():
        mc = v.get("mcnemar_vs_r1")
        tail = (f" | {mc['b_r1_only']}/{mc['c_model_only']} | {mc['p_exact']:.3f}"
                if mc else " | — | —")
        lines.append(f"| {k} | {v['sign_accuracy']:.3f} | {v['mean_abs_separation']:.4f} | "
                     f"{v['median_abs_separation']:.4f}{tail} |")
    lines += ["", "随机基线 = 0.500。McNemar 两列分别为「仅 R1 对」与「仅该模型对」的配对数，"
              "最后一列为精确二项 p 值（对 R1）。", "",
              "## 3. 最高流行率的 10 个描述符", "",
              "| label | prevalence | diversity | AUCPR R1 | Δ E-N3 | Δ R2.5 | Δ SchNet |",
              "|---|---|---|---|---|---|---|"]
    for _, r in tab.head(10).iterrows():
        dv = "n/a" if r["diversity"] is None else f"{r['diversity']:.3f}"
        lines.append(f"| {r['label']} | {r['prevalence']:.3f} | {dv} | {r['aucpr_r1']:.4f} | "
                     f"{r['delta_en3']:+.4f} | {r['delta_r25']:+.4f} | {r['delta_schnet']:+.4f} |")
    lines += ["", "完整 138 行见 `results/s3_tiers/exp31_subgroups.json`。"]
    with open(os.path.join(PROJ, "reports", "09_exp31_subgroups_mmp.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
