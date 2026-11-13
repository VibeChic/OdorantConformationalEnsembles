#!/usr/bin/env python3
"""Mirror-sensitivity test on trained R2 checkpoints (inference only).

For each molecule we reflect the conformer through a plane (z -> -z) and compare
predictions. An E(3)-invariant model must be *exactly* invariant; a model that
consumes the signed chiral volume must change.

This is the cheapest direct evidence for the "3D-WL" claim: it needs no
retraining and no labels.

Usage:
  python p2/mirror_test.py --runs results/r2/egnn/seed0 results/r2/chiral_egnn/seed0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from p2.models.r2 import build  # noqa: E402
from p2.train_r2 import load_split  # noqa: E402


@torch.no_grad()
def predict(model, loader, device, reflect=False, flip_v=False):
    model.eval()
    out = []
    for b in loader:
        b = b.to(device)
        if reflect:
            pos = b.pos.clone()
            pos[:, 2] = -pos[:, 2]
            b.pos = pos
            if flip_v:
                b.v = -b.v
        out.append(torch.sigmoid(model(b)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache_dir", default=os.path.join(PROJ, "cache"))
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "r2", "mirror_test.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = load_split("test", args.cache_dir)
    loader = DataLoader(test, batch_size=32, shuffle=False)

    res = {}
    for run in args.runs:
        ck = torch.load(os.path.join(run, "best_model.pt"), map_location=device,
                        weights_only=False)
        cfg = ck["config"]
        model = build(cfg["model"], dropout=cfg["dropout"],
                      layer_norm=cfg.get("layer_norm", False),
                      norm_feats=cfg.get("norm_feats", False)).to(device)
        model.load_state_dict(ck["model_state_dict"])
        p0 = predict(model, loader, device)
        p_ref = predict(model, loader, device, reflect=True, flip_v=True)
        dp = float(np.abs(p0 - p_ref).mean())
        corr = float(np.mean([np.corrcoef(p0[:, j], p_ref[:, j])[0, 1]
                              for j in range(p0.shape[1]) if p0[:, j].std() > 1e-9]))
        res[run] = {"model": cfg["model"], "mean_abs_dp": dp,
                    "mean_per_label_corr": corr,
                    "max_abs_dp": float(np.abs(p0 - p_ref).max())}
        print(f"{cfg['model']:12s} mean|Δp| {dp:.6f}  max {res[run]['max_abs_dp']:.4f}  "
              f"corr {corr:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("[done]", args.out)


if __name__ == "__main__":
    main()
