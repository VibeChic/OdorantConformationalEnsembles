#!/usr/bin/env python3
"""T7 / WP-C -- DREAM tier (338 molecules, 21 continuous targets), 2D arms only.

Per A-2 the main table is 2D (R1, E-N1, E-N6) plus a Morgan + Ridge baseline; 3D arms
are secondary and require the exact-InChIKey join, so they are not run here.

Protocol
  * folds: Butina clusters (Morgan2/2048, Tanimoto >= 0.6) grouped into 5 balanced folds,
    so no near-duplicate molecule straddles train/test;
  * one "run" = one (arm, seed) covering all 5 folds -> pooled out-of-fold predictions;
  * metric: per-target pooled OOF Pearson r, then the mean over the 21 targets
    (both this and the single pooled-over-all r are written; the mean-of-r is primary);
  * fixed 120 epochs, no early stopping (no validation fold exists inside a 5-fold CV).

Outputs
  results/s3_tiers/dream/{arm}/seed{s}/{metrics.json, oof.npz}

Usage
  python p2/train_dream.py --arm r1 --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from torch_geometric.loader import DataLoader as PyGLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from p2.models.r25 import ScalarGIN                      # noqa: E402
from src.data.featurization import featurize_mol        # noqa: E402
from src.models import get_model                        # noqa: E402
from src.utils.seed import set_seed                     # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = os.path.join(PROJ, "results", "s3_tiers", "dream")
DATA = os.path.join(REPO, "data", "external", "dream", "dream_dataset.csv")
TARGETS = None          # filled from the CSV


def ring_flex(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    bonds = set()
    for r in mol.GetRingInfo().BondRings():
        if len(r) >= 7 and not all(mol.GetBondWithIdx(b).GetIsAromatic() for b in r):
            bonds.update(r)
    return len(bonds)


def load_dream():
    df = pd.read_csv(DATA)
    tcols = [c for c in df.columns if c not in ("cid", "SMILES", "inchikey")]
    df["n_rotors"] = df.SMILES.map(lambda s: Chem.rdMolDescriptors.CalcNumRotatableBonds(
        Chem.MolFromSmiles(s)))
    df["n_ring_flex"] = df.SMILES.map(ring_flex)
    return df, tcols


def folds(df, k=5):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df.SMILES]
    dist = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dist, len(fps), 0.4, isDistData=True, reordering=True)
    sizes = [len(c) for c in clusters]
    order = np.argsort(sizes)[::-1]
    load = [0] * k
    assign = np.zeros(len(fps), dtype=int)
    for ci in order:
        f = int(np.argmin(load))
        assign[list(clusters[ci])] = f
        load[f] += sizes[ci]
    return assign


def graphs(df, cols):
    gs = []
    for i, r in df.iterrows():
        x, ei, ea = featurize_mol(Chem.MolFromSmiles(r.SMILES))
        s = torch.tensor([r[c] for c in cols], dtype=torch.float32).view(1, -1) \
            if cols else torch.zeros(1, 0)
        gs.append((x, ei, ea, s, i))
    return gs


class DS(torch.utils.data.Dataset):
    def __init__(self, items, idx, Y, cols):
        self.items = [items[i] for i in idx]
        self.Y = Y
        self.cols = cols

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        x, ei, ea, s, i = self.items[k]
        from torch_geometric.data import Data
        return Data(x=x, edge_index=ei, edge_attr=ea, scalars=s,
                    y=torch.tensor(self.Y[i], dtype=torch.float32).view(1, -1))


def build_model(arm, n_out, n_scalars):
    if arm == "r1":
        return get_model("gin")(num_classes=n_out, hidden_dim=256, num_layers=3,
                                dropout=0.15, eps=0.0, train_eps=True)
    return ScalarGIN(num_classes=n_out, n_scalars=max(1, n_scalars), dropout=0.15)


def train_predict(arm, tr_idx, te_idx, items, Y, Ymu, Ysd, cols, seed, epochs=120):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    mu = Y[tr_idx].mean(0)
    sd = Y[tr_idx].std(0) + 1e-6
    Yz = (Y - mu) / sd
    model = build_model(arm, Y.shape[1], len(cols)).to(dev)
    dl = PyGLoader(DS(items, tr_idx, Yz, cols), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        for b in dl:
            b = b.to(dev)
            out = model(b)
            loss = F.mse_loss(out, b.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    outs = []
    with torch.no_grad():
        for b in PyGLoader(DS(items, te_idx, Yz, cols), batch_size=64, shuffle=False):
            b = b.to(dev)
            outs.append(model(b).cpu().numpy())
    return np.concatenate(outs) * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["r1", "en1", "en6", "morgan"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    df, tcols = load_dream()
    Y = df[tcols].values.astype("float64")
    fold = folds(df)
    arm_cols = {"r1": [], "en1": ["n_rotors"], "en6": ["n_rotors", "n_ring_flex"],
                "morgan": []}[args.arm]
    items = graphs(df, arm_cols) if args.arm != "morgan" else None

    t0 = time.time()
    P = np.full_like(Y, np.nan)
    for f in range(5):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if args.arm == "morgan":
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            X = np.stack([np.array(gen.GetFingerprint(Chem.MolFromSmiles(s)))
                          for s in df.SMILES])
            m = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            P[te] = m.predict(X[te])
        else:
            P[te] = train_predict(args.arm, tr, te, items, Y, None, None, arm_cols,
                                  args.seed, args.epochs)
        print(f"  fold {f}: train {len(tr)} test {len(te)}", flush=True)

    rs, rhos = [], []
    for j, c in enumerate(tcols):
        r = pearsonr(Y[:, j], P[:, j]).statistic
        rs.append(float(r))
        rhos.append(float(spearmanr(Y[:, j], P[:, j]).statistic))
    pooled = float(np.corrcoef(Y.ravel(), P.ravel())[0, 1])
    out_dir = os.path.join(OUT, args.arm, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "oof.npz"), p=P, y=Y, fold=fold,
             targets=np.array(tcols))
    m = {"arm": args.arm, "seed": args.seed, "n_molecules": int(len(df)),
         "n_targets": len(tcols), "folds": int(fold.max() + 1),
         "mean_r": float(np.mean(rs)), "median_r": float(np.median(rs)),
         "pooled_r_all": pooled,
         "mean_spearman": float(np.mean(rhos)),
         "per_target_r": dict(zip(tcols, rs)),
         "per_target_spearman": dict(zip(tcols, rhos)),
         "minutes": round((time.time() - t0) / 60, 3),
         "config": vars(args)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    print(f"[DREAM {args.arm} seed{args.seed}] mean r {m['mean_r']:.3f} "
          f"(pooled {pooled:.3f}) | {m['minutes']} min")


if __name__ == "__main__":
    main()
