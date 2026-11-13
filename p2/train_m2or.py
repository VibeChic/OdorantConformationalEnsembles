#!/usr/bin/env python3
"""T8 / WP-C -- M2OR dual tower (ligand GIN x receptor ESM-2 tokens), stage 1.

Design: `Paper2-Dual-Tower-1026_2003.md`. Gates: prereg Amendment (c).

  arm A  dual tower; ligand tower initialised from the GS-LF R1 checkpoint
  arm B  ligand-only: same ligand tower, one output unit per receptor (fixed head)
  arm C  dual tower with the receptor embeddings permuted across receptors (control)
  arm D  Morgan + logistic regression -- separate script (p2/m2or_morgan.py)

Split: cold-ligand 5-fold, Butina clusters (Morgan2/2048, Tanimoto >= 0.6), so no
near-duplicate ligand straddles train/test and every pair is predicted once.

Receptors without a resolved sequence (13 of 436) are excluded and counted.

Outputs
  results/s3_tiers/m2or/{arm}/seed{s}/{metrics.json, per_receptor.json, oof.npz}
  results/s3_tiers/m2or/{arm}/seed{s}/config_used.yaml

Usage
  python p2/train_m2or.py --arm A --seed 0 --epochs 25
  python p2/train_m2or.py --arm A --seed 0 --folds 2 --epochs 3   # smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from src.data.featurization import featurize_mol      # noqa: E402
from src.models.gin import GIN                        # noqa: E402
from src.utils.seed import set_seed                   # noqa: E402

RDLogger.DisableLog("rdApp.*")
CACHE = os.path.join(PROJ, "cache")
OUT = os.path.join(PROJ, "results", "s3_tiers", "m2or")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REC_BATCH = 16


# ------------------------------------------------------------------ data
def load_all():
    mat = pd.read_csv(os.path.join(REPO, "data", "external", "m2or",
                                   "m2or_receptor_ligand.csv"), index_col=0)
    lig = pd.read_csv(os.path.join(REPO, "data", "external", "m2or",
                                   "m2or_ligands.csv")).set_index("inchikey")
    seq = pd.read_csv(os.path.join(CACHE, "m2or_receptor_sequences.csv"))
    seq = seq[seq.sequence.notna() & (seq.sequence != "")].set_index("entry")

    recs = [r for r in mat.index if r in seq.index]
    ligs = [c for c in mat.columns if c in lig.index]
    Y = mat.loc[recs, ligs].values.astype("float32")
    smi = {k: lig.loc[k, "smiles"] for k in ligs}

    z = np.load(os.path.join(CACHE, "esm2_m2or_tokens.npz"), allow_pickle=True)
    ent = list(z["entry"])
    off = dict(zip(ent, z["offsets"][:-1]))
    ln = dict(zip(ent, z["lengths"]))
    tok = z["tokens"]

    W = np.ones_like(Y, dtype="float32")
    a = pd.read_csv(os.path.join(REPO, "data", "external", "m2or", "m2or_assays.csv"),
                    low_memory=False)
    pos = a[(a.responsive == 1) & (a.parameter != "ec50")]
    ec50 = a[(a.responsive == 1) & (a.parameter == "ec50")]
    r2i = {r: i for i, r in enumerate(recs)}
    l2j = {c: j for j, c in enumerate(ligs)}
    for _, row in ec50.iterrows():
        i, j = r2i.get(row.receptor), l2j.get(row.inchikey)
        if i is not None and j is not None and Y[i, j] == 1:
            W[i, j] = 1.0
    for _, row in pos.iterrows():
        i, j = r2i.get(row.receptor), l2j.get(row.inchikey)
        if i is not None and j is not None and Y[i, j] == 1:
            W[i, j] = 0.6
    return recs, ligs, Y, W, smi, seq, tok, off, ln


def folds_for(ligs, smi, k=5):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(smi[c])) for c in ligs]
    dist = []
    for i in range(1, len(fps)):
        dist.extend([1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    cl = Butina.ClusterData(dist, len(fps), 0.4, isDistData=True, reordering=True)
    load = [0] * k
    assign = np.zeros(len(fps), dtype=int)
    for ci in np.argsort([len(c) for c in cl])[::-1]:
        f = int(np.argmin(load))
        assign[list(cl[ci])] = f
        load[f] += len(cl[ci])
    return assign


def lig_graphs(ligs, smi):
    gs = []
    for c in ligs:
        x, ei, ea = featurize_mol(Chem.MolFromSmiles(smi[c]))
        gs.append(Data(x=x, edge_index=ei, edge_attr=ea))
    return gs


def rec_tokens(recs, seq, tok, off, ln):
    T = max(int(ln[r]) for r in recs)
    X = np.zeros((len(recs), T, tok.shape[1]), dtype="float32")
    M = np.zeros((len(recs), T), dtype="float32")
    for i, r in enumerate(recs):
        L = min(int(ln[r]), T)
        X[i, :L] = tok[off[r]:off[r] + L]
        M[i, :L] = 1.0
    return torch.from_numpy(X), torch.from_numpy(M)


# ------------------------------------------------------------------ model
class ReceptorTower(nn.Module):
    def __init__(self, d_in=1280, d=128, d_out=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.SiLU())
        self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.out = nn.Sequential(nn.Linear(d, d_out), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, X, M):
        H = self.proj(X)
        s = self.att(H).squeeze(-1).masked_fill(M == 0, -1e4)
        a = torch.softmax(s, 1)
        z = (H * a.unsqueeze(-1)).sum(1)
        ent = float((-(a.clamp_min(1e-9).log() * a).sum(1)
                     / torch.log(M.sum(1).clamp_min(2))).mean())
        return self.out(z), ent


class Dual(nn.Module):
    def __init__(self, arm, n_rec, dropout=0.1, init_r1=None):
        super().__init__()
        self.arm = arm
        self.gin = GIN(num_classes=138, hidden_dim=256, num_layers=3, dropout=dropout,
                       eps=0.0, train_eps=True)
        if init_r1:
            sd = torch.load(init_r1, map_location="cpu", weights_only=False)["model_state_dict"]
            sd = {k: v for k, v in sd.items() if k.startswith("convs")}
            missing = self.gin.load_state_dict(sd, strict=False)
            print(f"[init] R1 transfer: {len(sd)} tensors, missing {len(missing.missing_keys)}")
        if arm == "B":
            self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(),
                                      nn.Dropout(dropout), nn.Linear(256, n_rec))
        else:
            self.rec = ReceptorTower(d_out=256)
            self.fuse = nn.Sequential(nn.Linear(256 + 256 + 256, 256), nn.SiLU(),
                                      nn.Dropout(dropout), nn.Linear(256, 128), nn.SiLU(),
                                      nn.Linear(128, 1))

    def lig(self, batch):
        x, b = self.gin.encode(batch)
        return global_mean_pool(x, b)

    def pairs(self, zl, zr):
        L, R = zl.shape[0], zr.shape[0]
        a = zl[:, None, :].expand(L, R, -1)
        b = zr[None, :, :].expand(L, R, -1)
        return self.fuse(torch.cat([a, b, a * b], -1)).squeeze(-1)


# ------------------------------------------------------------------ train
def family_folds(recs, seq, k=5):
    """Cold-receptor folds: whole OR families are held out, never split.

    Family = the digits of the gene name (OR51E2 -> OR51); mutants inherit their
    parent gene. Entries without a family become their own singleton group. Folds
    are filled greedily by size so each holds ~1/k of the receptors.
    """
    import re
    a = pd.read_csv(os.path.join(REPO, "data", "external", "m2or", "m2or_assays.csv"),
                    low_memory=False)
    g2 = dict(zip(a.uniprot.astype(str), a.gene.astype(str)))
    fam_of = {}
    for r in recs:
        m = re.match(r"^h?(OR[0-9]+[A-Z]{1,2}[0-9]+)", str(r))
        gene = m.group(1) if m else g2.get(str(r))
        m2 = re.match(r"OR([0-9]+)", str(gene)) if gene else None
        fam_of[r] = f"OR{m2.group(1)}" if m2 else f"singleton:{r}"
    groups = {}
    for r, f in fam_of.items():
        groups.setdefault(f, []).append(r)
    load = [0] * k
    assign = np.zeros(len(recs), dtype=int)
    for f in sorted(groups, key=lambda x: -len(groups[x])):
        j = int(np.argmin(load))
        for r in groups[f]:
            assign[recs.index(r)] = j
        load[j] += len(groups[f])
    return assign, fam_of


# ------------------------------------------------------------------ train
def train_fold(model, arm, rec_tr, lig_tr, Y, W, graphs, Xtok, Mtok, epochs, seed,
               perm=None):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    torch.manual_seed(seed)
    Ytr = Y[np.ix_(rec_tr, lig_tr)]
    Wtr = W[np.ix_(rec_tr, lig_tr)]
    n_rec, n_lig = Ytr.shape
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n_rec)
        for s in range(0, n_rec, REC_BATCH):
            loc = order[s:s + REC_BATCH].numpy()
            rec_glob = rec_tr[loc]
            cols = np.where(~np.isnan(Ytr[loc]).all(0))[0]
            if len(cols) == 0:
                continue
            sub = Ytr[np.ix_(loc, cols)]
            mask = torch.from_numpy(~np.isnan(sub)).to(DEV)
            if mask.sum() == 0:
                continue
            y = torch.from_numpy(np.nan_to_num(sub)).to(DEV)
            w = torch.from_numpy(Wtr[np.ix_(loc, cols)]).to(DEV)
            b = Batch.from_data_list([graphs[lig_tr[c]] for c in cols]).to(DEV)
            zl = model.lig(b)
            if arm == "B":
                logits = model.head(zl)[:, rec_glob]              # [L,R]
            else:
                src = perm[rec_glob] if arm == "C" else torch.from_numpy(rec_glob)
                zr, _ = model.rec(Xtok[src].to(DEV), Mtok[src].to(DEV))
                logits = model.pairs(zl, zr)
            loss = (F.binary_cross_entropy_with_logits(logits, y.t(), reduction="none")
                    * mask.t() * w.t()).sum() / (mask * w).sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def predict(model, arm, rec_te, lig_te, graphs, Xtok, Mtok, perm=None):
    model.eval()
    rec_te = np.asarray(rec_te)
    lig_te = np.asarray(lig_te)
    gbatch = Batch.from_data_list([graphs[c] for c in lig_te]).to(DEV)
    zl = model.lig(gbatch)
    out = np.full((len(rec_te), len(lig_te)), np.nan, dtype="float32")
    ents = []
    for s in range(0, len(rec_te), REC_BATCH):
        loc = slice(s, min(s + REC_BATCH, len(rec_te)))
        rec_glob = rec_te[loc]
        if arm == "B":
            out[loc] = model.head(zl)[:, rec_glob].t().cpu().numpy()
        else:
            src = perm[rec_glob] if arm == "C" else torch.from_numpy(rec_glob)
            zr, ent = model.rec(Xtok[src].to(DEV), Mtok[src].to(DEV))
            ents.append(ent)
            out[loc] = model.pairs(zl, zr).t().cpu().numpy()
    return out, (float(np.mean(ents)) if ents else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--split", default="ligand", choices=["ligand", "receptor"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache_dir", default=CACHE)
    args = ap.parse_args()
    tag = f"{args.split}"
    d = os.path.join(OUT, tag, args.arm, f"seed{args.seed}")
    os.makedirs(d, exist_ok=True)
    set_seed(args.seed)
    t0 = time.time()

    recs, ligs, Y, W, smi, seq, tok, off, ln = load_all()
    print(f"[data] {len(recs)} receptors x {len(ligs)} ligands | observed "
          f"{int(np.isfinite(Y).sum())} | positives {int(np.nansum(Y))} | device {DEV}")
    graphs = lig_graphs(ligs, smi)
    Xtok, Mtok = rec_tokens(recs, seq, tok, off, ln)
    print(f"[tokens] {tuple(Xtok.shape)}")

    perm = None
    if args.arm == "C":
        perm = torch.randperm(len(recs), generator=torch.Generator().manual_seed(7919))

    all_rec = np.arange(len(recs))
    all_lig = np.arange(len(ligs))
    if args.split == "ligand":
        fold = folds_for(ligs, smi, args.folds)
    else:
        fold, fam_of = family_folds(recs, seq, args.folds)
        print(f"[split] families {len(set(fam_of.values()))}, fold sizes "
              f"{np.bincount(fold).tolist()}")

    init = (os.path.join(PROJ, "results", "r1", "gin", f"seed{args.seed}", "best_model.pt")
            if args.arm in ("A", "C") else None)
    model = Dual(args.arm, len(recs), dropout=0.1, init_r1=init).to(DEV)
    print(f"[model] arm {args.arm} params {sum(p.numel() for p in model.parameters()):,}")

    P = np.full_like(Y, np.nan)
    ents = []
    for f in range(args.folds):
        if args.split == "ligand":
            lig_te = np.where(fold == f)[0]
            lig_tr = np.where(fold != f)[0]
            train_fold(model, args.arm, all_rec, lig_tr, Y, W, graphs, Xtok, Mtok,
                       args.epochs, args.seed * 100 + f, perm)
            p_te, ent = predict(model, args.arm, all_rec, lig_te, graphs, Xtok, Mtok, perm)
            P[:, lig_te] = p_te
        else:
            rec_te = np.where(fold == f)[0]
            rec_tr = np.where(fold != f)[0]
            train_fold(model, args.arm, rec_tr, all_lig, Y, W, graphs, Xtok, Mtok,
                       args.epochs, args.seed * 100 + f, perm)
            p_te, ent = predict(model, args.arm, rec_te, all_lig, graphs, Xtok, Mtok, perm)
            P[rec_te] = p_te
        ents.append(ent)
        print(f"  fold {f}: {len(lig_tr) if args.split == 'ligand' else len(rec_tr)} train / "
              f"{len(lig_te) if args.split == 'ligand' else len(rec_te)} test "
              f"[{time.time()-t0:.0f}s]", flush=True)

    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)},
               os.path.join(d, "best_model.pt"))

    per, aps, aucs = {}, [], []
    for i, r in enumerate(recs):
        m = np.isfinite(Y[i]) & np.isfinite(P[i])
        y, p = Y[i][m], P[i][m]
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue
        ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
        per[r] = {"n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                  "ap": float(ap), "auc": float(auc)}
        aps.append(ap)
        aucs.append(auc)
    met = {"arm": args.arm, "split": args.split, "seed": args.seed,
           "n_receptors_used": len(recs), "n_receptors_scored": len(aps),
           "macro_ap": float(np.mean(aps)), "macro_auc": float(np.mean(aucs)),
           "folds": args.folds, "epochs": args.epochs,
           "minutes": round((time.time() - t0) / 60, 2),
           "entropy_attention": (None if not ents or ents[0] is None
                                 else float(np.mean(ents))),
           "config": vars(args)}
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump(met, fh, indent=1)
    with open(os.path.join(d, "per_receptor.json"), "w") as fh:
        json.dump(per, fh, indent=1)
    np.savez(os.path.join(d, "oof.npz"), p=P, y=Y)
    with open(os.path.join(d, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(vars(args), fh, sort_keys=False)
    print(f"[TEST] {args.split} arm {args.arm} seed{args.seed} macro-AP "
          f"{met['macro_ap']:.4f} macro-AUC {met['macro_auc']:.4f} (n={len(aps)}) | "
          f"entropy {met['entropy_attention']} | {met['minutes']} min")


if __name__ == "__main__":
    main()
