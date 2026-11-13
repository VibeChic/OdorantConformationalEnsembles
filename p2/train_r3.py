#!/usr/bin/env python3
"""Train Rung 3 (ensemble-level MIL) on the rank-all conformer cache.

Protocol is identical to `p2/train_r2.py` (AdamW, lr 3e-4, 3-epoch warmup +
cosine, 100-epoch cap, batch 32 *molecules*, clip 1.0, dropout 0.15, early stop
on the 3-epoch moving average of val macro-AUCPR, patience 15, min 15 epochs).
The only difference is that a "batch item" is a molecule with C conformers.

Usage:
  python p2/train_r3.py --pool attn --seed 0 --out results/s1_mil/attn/seed0
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
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)
sys.path.insert(0, PROJ)

from p2.metrics import macro_metrics          # noqa: E402
from p2.models.r3 import SchNetMIL            # noqa: E402
from src.utils.seed import set_seed           # noqa: E402


def load_split(split, cache_dir):
    man = json.load(open(os.path.join(cache_dir, "_ensemble_manifest.json")))
    recs = torch.load(os.path.join(cache_dir, man["splits"][split]["path"]),
                      weights_only=False)
    return recs


def collate(mols):
    graphs, conf_mol, pis, ys = [], [], [], []
    for m, r in enumerate(mols):
        C = r["pos"].shape[0]
        for c in range(C):
            graphs.append(Data(x=r["x"], edge_index=r["edge_index"],
                               edge_attr=r["edge_attr"], pos=r["pos"][c]))
            conf_mol.append(m)
            pis.append(float(r["pi"][c]))
        ys.append(r["y"])
    b = Batch.from_data_list(graphs)
    b.bs = b.batch
    b.conf_mol = torch.tensor(conf_mol, dtype=torch.long)
    b.pi = torch.tensor(pis, dtype=torch.float)
    b.n_mol = len(mols)
    b.y = torch.stack(ys).float()
    return b


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    for b in loader:
        b = b.to(device)
        out = model(b)
        ps.append(torch.sigmoid(out).cpu().numpy())
        ys.append(b.y.cpu().numpy())
    P, Y = np.concatenate(ps), np.concatenate(ys)
    return macro_metrics(Y, P), P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    choices=["boltz", "attn", "shuffle", "rank1"])
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
    ap.add_argument("--limit_mols", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.path.join(PROJ, "cache"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_split("train", args.cache_dir)
    val = load_split("val", args.cache_dir)
    test = load_split("test", args.cache_dir)
    if args.limit_mols:
        train = train[: args.limit_mols]
    n_conf = sum(r["pos"].shape[0] for r in train)
    print(f"[data] train {len(train)} mol / {n_conf} conf | val {len(val)} | "
          f"test {len(test)} | device {device}")

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                    generator=g, collate_fn=collate)
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    sl = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SchNetMIL(num_classes=138, pool=args.pool, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] schnet_mil/{args.pool} params={n_params:,}")

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
    tm, P, Y = evaluate(model, sl, device)
    metrics = {"rung": "R3", "pool": args.pool, "model": f"schnet_mil_{args.pool}",
               "seed": args.seed, "n_params": n_params, "best_epoch": best_ep,
               "best_val_aucpr": best, "train_min": round((time.time() - t0) / 60, 2),
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"], "history": hist,
               "config": vars(args)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    np.savez(os.path.join(args.out, "test_preds.npz"), p=P, y=Y)
    print(f"[TEST] {args.pool} macro-AUCPR {tm['macro_aucpr']:.4f} | "
          f"micro-AUROC {tm['micro_auroc']:.4f} | {metrics['train_min']} min")


if __name__ == "__main__":
    main()
