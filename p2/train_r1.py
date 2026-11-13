#!/usr/bin/env python3
"""Rung-1 baseline under the Paper-2 protocol (matched-budget 2D GIN).

Why this file exists: the published R1 numbers (`results/all_results.json`) were
trained with 60 epochs / lr 1e-4 / no warmup on the full 3,488-molecule split.
Paper 2's R2 uses 100 epochs / lr 3e-4 / warmup+cosine on the 3,363-molecule
subset that has 3D coverage. Comparing across those two setups compares training
budgets, not architectures -- so R1 must be re-run here, on the same molecules
with the same schedule.

Outputs (paper2 only):
  cache/subset2d_{split}.csv
  results/r1/gin/seed<N>/{metrics.json,config_used.yaml,best_model.pt,test_preds.npz}
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)
sys.path.insert(0, PROJ)

from src.data.datasets import OdorDataset   # noqa: E402
from src.models import get_model             # noqa: E402
from src.utils.seed import set_seed          # noqa: E402
from p2.metrics import macro_metrics         # noqa: E402

RDLogger.DisableLog("rdApp.*")


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def build_subsets():
    # the 3D-covered subset == molecules present in the run1 3D manifest
    man = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"))
    have3d = set(man["canonical_SMILES"].map(canon)) - {None}
    keep_by_split = {s: have3d for s in ("train", "val", "test")}
    out = {}
    for split, keep in keep_by_split.items():
        df = pd.read_csv(os.path.join(REPO, "data", "full", f"{split}_gs_lf.csv"))
        df["_csmi"] = df["nonStereoSMILES"].map(canon)
        sub = df[df["_csmi"].isin(keep)].drop(columns=["_csmi"]).reset_index(drop=True)
        p = os.path.join(PROJ, "cache", f"subset2d_{split}.csv")
        sub.to_csv(p, index=False)
        out[split] = (p, len(sub))
    print("[subset]", {k: v[1] for k, v in out.items()})
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    for b in loader:
        b = b.to(device)
        out = model(b)
        ps.append(torch.sigmoid(out).cpu().numpy())
        ys.append(b.y.view(out.size(0), -1).cpu().numpy())
    P, Y = np.concatenate(ps), np.concatenate(ys)
    return macro_metrics(Y, P), P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gin", choices=["gin", "gcn", "gatv2", "mpnn", "graphsage"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_epochs", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--smooth_val", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = build_subsets()
    cache = os.path.join(PROJ, "cache", "graph2d")
    ds = {}
    for s in ("train", "val", "test"):
        ds[s] = OdorDataset(paths[s][0], os.path.join(REPO, "data", f"{s}-sdf"),
                            use_3d=False, cache_dir=cache)
    print(f"[data] train {len(ds['train'])} val {len(ds['val'])} test {len(ds['test'])}")

    cfg = {"hidden_dim": 256, "num_layers": 3, "dropout": args.dropout,
           "eps": 0.0, "train_eps": True}
    model = get_model(args.model)(num_classes=len(ds["train"].label_names), **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} params={n_params:,}")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.BCEWithLogitsLoss()
    best, best_ep, hist, pctr, t0 = -1.0, 0, [], 0, __import__("time").time()

    for ep in range(1, args.epochs + 1):
        lr = args.lr * (ep / args.warmup if ep <= args.warmup else
                        0.5 * (1 + math.cos(math.pi * (ep - args.warmup) /
                                            max(1, args.epochs - args.warmup))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        tot, nb = 0.0, 0
        for b in train_loader:
            b = b.to(device)
            out = model(b)
            loss = crit(out, b.y.view(out.size(0), -1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        vm, _, _ = evaluate(model, val_loader, device)
        hist.append({"epoch": ep, "train_loss": tot / max(1, nb), "lr": lr,
                     **{f"val_{k}": v for k, v in vm.items()}})
        smooth = float(np.mean([h["val_macro_aucpr"] for h in hist[-args.smooth_val:]]))
        print(f"E{ep:03d} loss {tot/max(1,nb):.4f} val AUCPR {vm['macro_aucpr']:.4f} "
              f"(smooth {smooth:.4f}) micro {vm['micro_auroc']:.4f}", flush=True)
        if smooth > best:
            best, best_ep, pctr = smooth, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "config": vars(args),
                        "best_val_aucpr": best, "epoch": ep, "n_params": n_params},
                       os.path.join(args.out, "best_model.pt"))
        elif ep >= args.min_epochs:
            pctr += 1
        if pctr >= args.patience and ep >= args.min_epochs:
            print(f"[early stop] {ep}")
            break

    ck = torch.load(os.path.join(args.out, "best_model.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    tm, P, Y = evaluate(model, test_loader, device)
    metrics = {"rung": "R1", "model": args.model, "seed": args.seed, "n_params": n_params,
               "best_epoch": best_ep, "best_val_aucpr": best,
               "train_min": round((__import__("time").time() - t0) / 60, 2),
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"], "history": hist,
               "config": vars(args), "subset": {k: v[1] for k, v in paths.items()}}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    np.savez(os.path.join(args.out, "test_preds.npz"), p=P, y=Y)
    print(f"[TEST] {args.model} macro-AUCPR {tm['macro_aucpr']:.4f} | "
          f"micro-AUROC {tm['micro_auroc']:.4f} | {metrics['train_min']} min")


if __name__ == "__main__":
    main()
