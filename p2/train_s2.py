#!/usr/bin/env python3
"""Experiment 2.1 — the 2^3 flexibility / shape scalar ablation.

Arms (all share the identical GIN backbone; only the appended scalars differ):

  r1        none  (kept for code symmetry only: the manuscript reuses the
            existing results/r1/gin baseline instead, because this head has a
            32-dim zero branch and is therefore not parameter-identical)
  en1       n_rotors                                    (2D only)
  en2       n_eff, top_pop, has_censored                (300 K xTB)
  en3       n_eff, top_pop, n_rotors                    (current best)
  en4       NPR1/NPR2/Rg mean+sd (6)                    (global 3D shape)
  en5       all 9
  en3_shuf  en3, permuted across molecules in the batch (control)

Scalars are z-scored with training-set statistics, computed once and stored in
the output directory so the transform is auditable.

Usage:
  python p2/train_s2.py --arm en3 --seed 0 --out results/s2_ablation/en3/seed0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)
sys.path.insert(0, PROJ)

from p2.metrics import macro_metrics          # noqa: E402
from p2.models.r25 import ScalarGIN           # noqa: E402
from src.utils.seed import set_seed           # noqa: E402

ARMS = {
    "r1": [],
    "en1": ["n_rotors"],
    "en2": ["n_eff", "top_pop", "has_censored"],
    "en3": ["n_eff", "top_pop", "n_rotors"],
    "en4": ["NPR1_mean", "NPR2_mean", "Rg_mean", "NPR1_sd", "NPR2_sd", "Rg_sd"],
    "en5": ["n_eff", "top_pop", "n_rotors", "NPR1_mean", "NPR2_mean", "Rg_mean",
            "NPR1_sd", "NPR2_sd", "Rg_sd"],
    "en3_shuf": ["n_eff", "top_pop", "n_rotors"],
}


def load_split(split, cache_dir, scal, cols):
    man = json.load(open(os.path.join(cache_dir, "_ensemble_manifest.json")))
    recs = torch.load(os.path.join(cache_dir, man["splits"][split]["path"]),
                      weights_only=False)
    out = []
    for r in recs:
        s = scal.loc[r["name"], cols].values.astype("float32") if cols else np.zeros(0, "float32")
        out.append(Data(x=r["x"], edge_index=r["edge_index"], edge_attr=r["edge_attr"],
                        scalars=torch.tensor(s, dtype=torch.float).unsqueeze(0),
                        y=r["y"].unsqueeze(0)))
    return out


@torch.no_grad()
def evaluate(model, loader, device, shuffle_scalars=False):
    model.eval()
    ps, ys = [], []
    for b in loader:
        b = b.to(device)
        if shuffle_scalars and b.scalars.numel():
            b.scalars = b.scalars[torch.randperm(b.scalars.size(0), device=b.scalars.device)]
        ps.append(torch.sigmoid(model(b)).cpu().numpy())
        ys.append(b.y.cpu().numpy())
    P, Y = np.concatenate(ps), np.concatenate(ys)
    return macro_metrics(Y, P), P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
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
    ap.add_argument("--cache_dir", default=os.path.join(PROJ, "cache"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cols = ARMS[args.arm]
    sc = pd.read_csv(os.path.join(PROJ, "cache", "exp2_scalars.csv")).set_index("name")
    tr = sc[sc.split == "train"]

    mu = tr[cols].mean() if cols else None
    sd = tr[cols].std().replace(0, 1.0) if cols else None
    if cols:
        sc[cols] = (sc[cols] - mu) / sd

    train = load_split("train", args.cache_dir, sc, cols)
    val = load_split("val", args.cache_dir, sc, cols)
    test = load_split("test", args.cache_dir, sc, cols)
    print(f"[data] {args.arm} cols={cols} train {len(train)} val {len(val)} test {len(test)}")

    g = torch.Generator().manual_seed(args.seed)
    tl = PyGLoader(train, batch_size=args.batch_size, shuffle=True, generator=g)
    vl = PyGLoader(val, batch_size=args.batch_size, shuffle=False)
    sl = PyGLoader(test, batch_size=args.batch_size, shuffle=False)

    model = ScalarGIN(num_classes=138, n_scalars=max(1, len(cols)),
                      dropout=args.dropout).to(device)
    if not cols:                      # r1 arm: zero the scalar branch at init
        for p in list(model.scalar_mlp.parameters()):
            torch.nn.init.zeros_(p)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.arm} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.BCEWithLogitsLoss()
    shuf = args.arm == "en3_shuf"

    t0 = time.time()
    best, best_ep, hist, pctr = -1.0, 0, [], 0
    for ep in range(1, args.epochs + 1):
        lr = args.lr * (ep / args.warmup if ep <= args.warmup else
                        0.5 * (1 + math.cos(math.pi * (ep - args.warmup) /
                                            max(1, args.epochs - args.warmup))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        tot, nb = 0.0, 0
        for b in tl:
            b = b.to(device)
            if shuf and b.scalars.numel():
                b.scalars = b.scalars[torch.randperm(b.scalars.size(0), device=b.scalars.device)]
            out = model(b)
            loss = crit(out, b.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        vm, _, _ = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_loss": tot / max(1, nb), "lr": lr,
                     **{f"val_{k}": v for k, v in vm.items()}})
        smooth = float(np.mean([h["val_macro_aucpr"] for h in hist[-args.smooth_val:]]))
        print(f"E{ep:03d} loss {tot/max(1,nb):.4f} val AUCPR {vm['macro_aucpr']:.4f} "
              f"(smooth {smooth:.4f}) micro {vm['micro_auroc']:.4f} lr {lr:.1e} "
              f"[{time.time()-t0:.0f}s]", flush=True)
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
    tm, P, Y = evaluate(model, sl, device, shuffle_scalars=shuf)
    metrics = {"suite": "2.1", "arm": args.arm, "seed": args.seed, "columns": cols,
               "n_params": n_params, "best_epoch": best_ep, "best_val_aucpr": best,
               "train_min": round((time.time() - t0) / 60, 2),
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"], "history": hist,
               "config": vars(args)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    if cols:
        with open(os.path.join(args.out, "scalar_transform.json"), "w") as f:
            json.dump({"mean": mu.astype(float).to_dict(),
                       "sd": sd.astype(float).to_dict()}, f, indent=1)
    np.savez(os.path.join(args.out, "test_preds.npz"), p=P, y=Y)
    print(f"[TEST] {args.arm} macro-AUCPR {tm['macro_aucpr']:.4f} | "
          f"micro-AUROC {tm['micro_auroc']:.4f} | {metrics['train_min']} min")


if __name__ == "__main__":
    main()
