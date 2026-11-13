#!/usr/bin/env python3
"""Train R2.5 (ensemble edge statistics) and its E-N3 control.

Usage:
  python p2/train_r25.py --model edge_gin   --seed 0 --out results/r25/edge_gin/seed0
  python p2/train_r25.py --model scalar_gin --seed 0 --out results/r25/scalar_gin/seed0
  python p2/train_r25.py --model edge_gin --shuffle_edges --seed 0 \
      --out results/r25/edge_gin_shuffled/seed0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)
sys.path.insert(0, PROJ)

from p2.models.r25 import build      # noqa: E402
from p2.metrics import macro_metrics  # noqa: E402
from src.utils.seed import set_seed   # noqa: E402


def load_split(split, cache_dir, shuffle_edges=False, scal_mean=None, scal_std=None):
    recs = torch.load(os.path.join(cache_dir, f"r25_{split}.pt"), weights_only=False)
    out = []
    for r in recs:
        ef = r["edge_feat"].clone()
        if shuffle_edges:
            g = torch.Generator().manual_seed(hash(r["name"]) % 2**31)
            ef = ef[torch.randperm(ef.shape[0], generator=g)]
        sc = r["scalars"].view(1, -1).clone()
        if scal_mean is not None:
            sc = (sc - scal_mean) / scal_std
        out.append(Data(x=r["x"], edge_index=r["edge_index"], edge_attr=r["edge_attr"],
                        edge_feat=ef, scalars=sc, y=r["y"].unsqueeze(0)))
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    for b in loader:
        b = b.to(device)
        ps.append(torch.sigmoid(model(b)).cpu().numpy())
        ys.append(b.y.cpu().numpy())
    P, Y = np.concatenate(ps), np.concatenate(ys)
    return macro_metrics(Y, P), P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["edge_gin", "scalar_gin"])
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
    ap.add_argument("--shuffle_edges", action="store_true",
                    help="control: permute edge statistics within each molecule")
    ap.add_argument("--cache_dir", default=os.path.join(PROJ, "cache"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_split("train", args.cache_dir, args.shuffle_edges)
    sc = torch.cat([d.scalars for d in train], 0)
    mean, std = sc.mean(0), sc.std(0).clamp_min(1e-6)
    train = load_split("train", args.cache_dir, args.shuffle_edges, mean, std)
    val = load_split("val", args.cache_dir, False, mean, std)
    test = load_split("test", args.cache_dir, False, mean, std)
    print(f"[data] train {len(train)} val {len(val)} test {len(test)} | device {device}")

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g)
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    sl = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    model = build(args.model, dropout=args.dropout,
                  edge_dim=int(train[0].edge_feat.shape[1])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.BCEWithLogitsLoss()
    t0, best, best_ep, hist, pctr = time.time(), -1.0, 0, [], 0

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
              f"(smooth {smooth:.4f}) micro {vm['micro_auroc']:.4f} [{time.time()-t0:.0f}s]",
              flush=True)
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
    tm, P, Y = evaluate(model, sl, device)
    metrics = {"rung": "R2.5", "model": args.model, "seed": args.seed,
               "shuffle_edges": args.shuffle_edges, "n_params": n_params,
               "best_epoch": best_ep, "best_val_aucpr": best,
               "train_min": round((time.time() - t0) / 60, 2),
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"], "history": hist,
               "config": vars(args)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    np.savez(os.path.join(args.out, "test_preds.npz"), p=P, y=Y)
    print(f"[TEST] {args.model}{' (shuffled)' if args.shuffle_edges else ''} "
          f"macro-AUCPR {tm['macro_aucpr']:.4f} | micro-AUROC {tm['micro_auroc']:.4f} | "
          f"{metrics['train_min']} min")


if __name__ == "__main__":
    main()
