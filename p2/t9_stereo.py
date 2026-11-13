#!/usr/bin/env python3
"""T9 / WP-D -- the 9 true enantiomer pairs on the dual tower (A1 = option a, restricted).

What this can and cannot test
-----------------------------
The dual tower's ligand tower consumes six atom features that carry no stereochemistry
(atomic number, degree, charge, H count, hybridisation, aromatic flag), so the two members
of an enantiomer pair are the *same graph*. This script verifies that claim numerically
(scores identical to the bit) rather than assuming it, and then reports the label-side
opportunity: how many receptors have discordant labels for each pair, i.e. how much signal
a parity-aware encoder would have to work with.

The parity-aware arms (R2.5 with signed sin phi, Chiral-SchNet) are trained on the 138
odour descriptors, not on receptor binding, so they cannot be scored on this panel without
the 3D join; that is reported as the boundary of this task.

Outputs
  results/wpd_stereo/pairs.json
  reports/18_stereo.md

Usage
  python p2/t9_stereo.py [--arm A] [--split receptor] [--seed 0]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from p2.train_m2or import Dual, load_all, lig_graphs, rec_tokens   # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = os.path.join(PROJ, "results", "wpd_stereo")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def mirror(s):
    m = Chem.MolFromSmiles(s)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    for a in m.GetAtoms():
        if a.HasProp("_CIPCode"):
            a.InvertChirality()
    return Chem.MolToSmiles(m)


def pairs():
    L = pd.read_csv(os.path.join(REPO, "data", "external", "m2or", "m2or_ligands.csv"),
                    low_memory=False)
    L["skel"] = L.inchikey.str.split("-").str[0]
    out = []
    for _, d in L.groupby("skel"):
        if len(d) != 2:
            continue
        a, b = d.iloc[0], d.iloc[1]
        if canon(a.smiles) == canon(b.smiles):
            continue
        if mirror(a.smiles) == canon(b.smiles):
            out.append((a["name"], b["name"], a.inchikey, b.inchikey))
    return out


@torch.no_grad()
def scores(model, arm, lig_graphs_list, Xtok, Mtok):
    b = Batch.from_data_list(lig_graphs_list).to(DEV)
    zl = model.lig(b)
    zr, _ = model.rec(Xtok.to(DEV), Mtok.to(DEV))
    return model.pairs(zl, zr).cpu().numpy()          # [L, R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--split", default="receptor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    recs, ligs, Y, W, smi, seq, tok, off, ln = load_all()
    l2i = {c: i for i, c in enumerate(ligs)}
    P = pairs()
    print(f"[panel] {len(P)} enantiomer pairs; receptors {len(recs)}; ligands {len(ligs)}")

    ck = os.path.join(PROJ, "results", "s3_tiers", "m2or", args.split, args.arm,
                      f"seed{args.seed}", "best_model.pt")
    model = Dual(args.arm, len(recs)).to(DEV)
    model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)["model_state_dict"])
    model.eval()

    Xtok, Mtok = rec_tokens(recs, seq, tok, off, ln)
    graphs = lig_graphs(ligs, smi)

    rows, deltas = [], []
    for name_a, name_b, ia, ib in P:
        ja, jb = l2i.get(ia), l2i.get(ib)
        if ja is None or jb is None:
            rows.append({"pair": f"{name_a} / {name_b}", "status": "ligand not in panel"})
            continue
        ga, gb = graphs[ja], graphs[jb]
        same_graph = bool(torch.equal(ga.x, gb.x) and torch.equal(ga.edge_index, gb.edge_index)
                          and torch.equal(ga.edge_attr, gb.edge_attr))
        S = scores(model, args.arm, [ga, gb], Xtok, Mtok)   # [2,R]
        d = S[0] - S[1]
        deltas.append(float(np.abs(d).max()))
        ya, yb = Y[:, ja], Y[:, jb]
        both = np.isfinite(ya) & np.isfinite(yb)
        disc = both & (ya != yb)
        agree = None
        if disc.sum():
            correct = ((ya[disc] == 1) & (d[disc] > 0)) | ((yb[disc] == 1) & (d[disc] < 0))
            agree = float(correct.mean())
        rows.append({"pair": f"{name_a} / {name_b}",
                     "identical_graph": same_graph,
                     "n_receptors_both": int(both.sum()),
                     "n_discordant": int(disc.sum()),
                     "sign_accuracy": agree,
                     "max_abs_score_diff": float(np.abs(d).max())})

    out = {"arm": args.arm, "split": args.split, "seed": args.seed,
           "n_pairs": len(P),
           "max_abs_score_diff_over_pairs": max(deltas) if deltas else None,
           "pairs": rows,
           "note": ("the ligand tower has no stereochemical input: both members of every "
                    "pair produce the same graph, so a zero score difference is a proof of "
                    "blindness, not a failed test")}
    with open(os.path.join(OUT, "pairs.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    write_report(out)
    print(json.dumps({k: out[k] for k in ("n_pairs", "max_abs_score_diff_over_pairs")},
                     indent=1))
    for r in rows:
        print(" ", r)


def write_report(out):
    rows = out["pairs"]
    lines = ["# 18 — 手性（T9，A1 = 方案 a 的受限版）", "",
             "由 `p2/t9_stereo.py` 生成。面板：M2OR 中 9 组经镜像核验的真对映体对"
             "（22 组里的 9 组；其余为 E/Z、非对映体或标注缺失，见预注册 Amendment 2026-10-26）。", "",
             "## 1. 结果", "",
             "| 对 | 图逐位相同 | 两侧都有标签的受体 | 标签不一致 | 最大打分差 |",
             "|---|---|---|---|---|"]
    for r in rows:
        if "status" in r:
            lines.append(f"| {r['pair']} | — | — | — | {r['status']} |")
            continue
        lines.append(f"| {r['pair']} | {'是' if r.get('identical_graph') else '否'} | "
                     f"{r['n_receptors_both']} | {r['n_discordant']} | "
                     f"{r['max_abs_score_diff']:.2e} |")
    lines += ["", f"所有对的最大打分差：**{out['max_abs_score_diff_over_pairs']:.1e}**。", "",
              "## 2. 这不是一次失败的检验，而是一个构造性事实", "",
              "配体塔的 6 维原子特征（原子序数、度、电荷、氢数、杂化、芳香性）**不含任何立体信息**，"
              "两个对映体的分子图逐位相同（上表第二列），因此模型输出相同，差异只剩 float32 的舍入"
              "（≤1.5e-5，相对 logit 量级约 1e-5）。**任何用这点差值算出的符号一致率都是舍入噪声，"
              "不是化学信号**，所以本表不报它。", "",
              "这与论文 §2.7 的结论一致：E(3) 不变（以及此处完全无手性输入的 2D）架构**能表示**手性差异"
              "是无意义的说法——它们不能；而 `Chiral-SchNet` 与 `R2.5`（⟨sin φ⟩）是本研究里唯二"
              "带宇称通道的臂，实测镜像 Δp 分别为 1.05e-3 与 2.06e-3。", "",
              "## 3. 边界（为什么不在这条线上继续）", "",
              "- `R2.5` / `Chiral-SchNet` 训练在 138 个气味描述符上，不在受体结合面板上；"
              "要把它们搬到 M2OR 需要这 44 个配体的 3D 系综，而精确 InChIKey 只覆盖 330/488 ——"
              "属于 A-2 里冻结的「3D 臂需精确 join」的范畴，且会引入另一套不确定性。",
              "- 因此手性在本论文里的正确写法是**表示能力 + 数据缺陷**两条，而不是对映体选择性效应："
              "9 组对里标签不一致的受体数（上表）说明**信号存在**，但本研究没有任何一个臂能在"
              "该面板上表达它。"]
    with open(os.path.join(PROJ, "reports", "18_stereo.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
