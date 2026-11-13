#!/usr/bin/env python3
"""T4 (temperature sweep) and T5 (Gaussian TTA) on the R3 MIL backbone.

Both variants only change the *weights* and, for T5, the *coordinates*; the
architecture, protocol and metrics are identical to `p2/train_r3.py`.

  --mode temp  --tau T
      w_c(T) ∝ p_c ** (300 / T), renormalised per molecule  (see prereg
      Amendment (f): the stored populations are Boltzmann at 300 K over a
      thermochemical energy, not over the cache's `dE`).
      T = inf  ->  w_c = 1/C  (deterministic; distinct from the `shuffle` arm).

  --mode gauss --sigma S
      replaces the ensemble with C copies of the rank-1 conformer plus
      N(0, S^2 I) noise, weights 1/C.  S = "rmsd" uses the per-molecule
      median RMSD of the real ensemble against its rank-1 conformer.

Outputs (same schema as train_r3 plus variant fields)
  results/s1_mil/temp_{T}/seed{s}/      or   results/s1_mil/gauss_{S}/seed{s}/

Usage
  python p2/train_r3_variants.py --mode temp --tau 100 --seed 0 --out results/s1_mil/temp_100/seed0
  python p2/train_r3_variants.py --mode gauss --sigma rmsd --seed 0 --out results/s1_mil/gauss_rmsd/seed0
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from p2.models.r3 import SchNetMIL                        # noqa: E402
from p2.train_r3 import collate, evaluate, load_split     # noqa: E402
from src.utils.seed import set_seed                       # noqa: E402

T0 = 300.0
CACHE = os.path.join(PROJ, "cache")


def temp_weights(pi, tau):
    p = torch.as_tensor(pi, dtype=torch.float64).clamp_min(1e-12)
    if math.isinf(tau):
        return torch.full_like(p, 1.0 / len(p)).float()
    w = p ** (T0 / tau)
    return (w / w.sum()).float()


def n_eff(pi):
    p = np.asarray(pi, dtype="float64")
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def kabsch_rmsd(a, b):
    """RMSD between two (N,3) arrays after optimal rotation."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    c = b.T @ a
    v, s, wt = np.linalg.svd(c)
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1, 1, d]) @ wt
    b = b @ rot
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def sigma_table(recs, split):
    """Per-molecule median RMSD of the ensemble against its rank-1 conformer."""
    p = os.path.join(CACHE, "wpb_gauss_sigma.npz")
    tab = {}
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        tab = dict(zip(z["name"], z["sigma"]))
    missing = [r for r in recs if r["name"] not in tab]
    if not missing:
        return tab
    for r in missing:
        pos = r["pos"].numpy()
        if pos.shape[0] < 2:
            tab[r["name"]] = 0.0
            continue
        tab[r["name"]] = float(np.median([kabsch_rmsd(pos[0], pos[c])
                                          for c in range(1, pos.shape[0])]))
    np.savez(p, name=np.array(list(tab)), sigma=np.array(list(tab.values())))
    return tab


def build_variant(split, mode, tau, sigma, cache_dir=CACHE, rng=None):
    recs = load_split(split, cache_dir)
    out = []
    neffs = []
    st = sigma_table(load_split(split, cache_dir), split) if (mode == "gauss"
                                                              and sigma == "rmsd") else None
    for r in recs:
        r = copy.copy(r)
        if mode == "temp":
            r["pi"] = temp_weights(r["pi"], tau)
        else:
            C = r["pos"].shape[0]
            sig = st[r["name"]] if st is not None else float(sigma)
            base = r["pos"][0].numpy()
            noise = rng.normal(0.0, sig, size=(C,) + base.shape).astype("float32")
            r["pos"] = torch.from_numpy(base[None] + noise)
            r["pi"] = torch.full((C,), 1.0 / C)
        neffs.append(n_eff(r["pi"].numpy()))
        out.append(r)
    return out, float(np.median(neffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["temp", "gauss"])
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--sigma", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_epochs", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--smooth_val", type=int, default=3)
    ap.add_argument("--cache_dir", default=CACHE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "temp" and args.tau is None:
        raise SystemExit("--tau required for --mode temp")
    if args.mode == "gauss" and args.sigma is None:
        raise SystemExit("--sigma required for --mode gauss")
    tau = math.inf if (args.mode == "temp" and str(args.tau).lower() in ("inf", "infinity")) \
        else (float(args.tau) if args.mode == "temp" else None)

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    train, n_tr = build_variant("train", args.mode, tau, args.sigma, args.cache_dir, rng)
    val, n_va = build_variant("val", args.mode, tau, args.sigma, args.cache_dir, rng)
    test, n_te = build_variant("test", args.mode, tau, args.sigma, args.cache_dir, rng)
    tag = (f"tau={tau}" if args.mode == "temp" else f"sigma={args.sigma}")
    print(f"[data] {args.mode} {tag} | train {len(train)} | val {len(val)} | test {len(test)} "
          f"| median N_eff train {n_tr:.2f} test {n_te:.2f}")

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g,
                    collate_fn=collate)
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    sl = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SchNetMIL(num_classes=138, pool="boltz", dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] schnet_mil/boltz (variant weights) params={n_params:,}")

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
            loss = crit(model(b), b.y)
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
    metrics = {"suite": "2.2", "arm": (f"temp_{tau}" if args.mode == "temp"
                                       else f"gauss_{args.sigma}"),
               "mode": args.mode, "tau": (None if tau is None or math.isinf(tau) else tau),
               "tau_inf": bool(tau is not None and math.isinf(tau)),
               "sigma": args.sigma, "seed": args.seed,
               "median_n_eff_train": n_tr, "median_n_eff_test": n_te,
               "n_params": n_params, "best_epoch": best_ep, "best_val_aucpr": best,
               "train_min": round((time.time() - t0) / 60, 2),
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"], "history": hist,
               "config": vars(args)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    np.savez(os.path.join(args.out, "test_preds.npz"), p=P, y=Y)
    print(f"[TEST] {metrics['arm']} macro-AUCPR {tm['macro_aucpr']:.4f} | "
          f"micro {tm['micro_auroc']:.4f} | N_eff {n_te:.2f} | {metrics['train_min']} min")


if __name__ == "__main__":
    main()
