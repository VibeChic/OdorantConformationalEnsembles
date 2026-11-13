#!/usr/bin/env python3
"""en6 -- E-N6: GIN + [n_rotors, n_ring_flex]  (WP-E, A4 option a).

Identical backbone, protocol and metrics to `p2/train_s2.py`; the only difference
is the scalar table: this arm reads `cache/exp2_scalars_v2.csv` because the v1
table is frozen and must not gain columns (todo section 4).

Usage:
  python p2/train_en6.py --seed 0 --out results/s2_ablation/en6/seed0
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
from torch_geometric.loader import DataLoader as PyGLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from p2.models.r25 import ScalarGIN                    # noqa: E402
from p2.train_s2 import evaluate, load_split          # noqa: E402
from src.utils.seed import set_seed                   # noqa: E402

COLS = ["n_rotors", "n_ring_flex"]


def main():
    ap = argparse.ArgumentParser()
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

    sc = pd.read_csv(os.path.join(PROJ, "cache", "exp2_scalars_v2.csv")).set_index("name")
    tr = sc[sc.split == "train"]
    mu, sd = tr[COLS].mean(), tr[COLS].std().replace(0, 1.0)
    sc[COLS] = (sc[COLS] - mu) / sd

    train = load_split("train", args.cache_dir, sc, COLS)
    val = load_split("val", args.cache_dir, sc, COLS)
    test = load_split("test", args.cache_dir, sc, COLS)
    print(f"[data] en6 cols={COLS} train {len(train)} val {len(val)} test {len(test)}")

    g = torch.Generator().manual_seed(args.seed)
    tl = PyGLoader(train, batch_size=args.batch_size, shuffle=True, generator=g)
    vl = PyGLoader(val, batch_size=args.batch_size, shuffle=False)
    sl = PyGLoader(test, batch_size=args.batch_size, shuffle=False)

    model = ScalarGIN(num_classes=138, n_scalars=len(COLS), dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] en6 params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.BCEWithLogitsLoss()

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
              f"(smooth {smooth:.4f}) [{time.time()-t0:.0f}s]", flush=True)
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
    metrics = {"suite": "2.1", "arm": "en6", "seed": args.seed, "columns": COLS,
               "n_params": n_params, "best_epoch": best_ep, "best_val_aucpr": best,
               "train_min": round((time.time() - t0) / 60, 2),
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"], "history": hist,
               "config": vars(args)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    with open(os.path.join(args.out, "scalar_transform.json"), "w") as f:
        json.dump({"mean": mu.astype(float).to_dict(),
                   "sd": sd.astype(float).to_dict()}, f, indent=1)
    np.savez(os.path.join(args.out, "test_preds.npz"), p=P, y=Y)
    print(f"[TEST] en6 macro-AUCPR {tm['macro_aucpr']:.4f} | micro {tm['micro_auroc']:.4f} "
          f"| {metrics['train_min']} min")


if __name__ == "__main__":
    main()
