#!/usr/bin/env python3
"""Mirror test for R2.5 (Edge-GIN), inference only.

R2.5's 66-dim edge vector is [RBF32(<d>), RBF32(sd d), <sin phi>, <cos phi>]. `sin phi`
is a pseudoscalar: under a reflection every conformer's sign flips, so the *input*
changes sign in one channel. Whether the trained model *uses* it is an empirical
question, and it decides whether R2.5 may enter the A1 stereo panel as a
chirality-sensitive architecture (reports/02_preregistration.md, Amendment 2026-10-26).

Four passes per checkpoint:
  orig      features as built by p2/data/build_r25_cache.py
  reflect   z -> -z on every conformer, features rebuilt
  rotate    random SO(3) applied to every conformer, features rebuilt (invariance control)
  sinflip   orig features with the <sin phi> column negated (usage-only probe)

Output: results/r25/mirror_test.json

Usage:
  python p2/mirror_test_r25.py --seeds 0 1 2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)
sys.path.insert(0, REPO)

from p2.data.build_r25_cache import edge_stats            # noqa: E402
from p2.models.r25 import build                           # noqa: E402

SIN_COL = 64


def load_test_recs():
    man = json.load(open(os.path.join(PROJ, "cache", "_ensemble_manifest.json")))
    return torch.load(os.path.join(PROJ, "cache", man["splits"]["test"]["path"]),
                      weights_only=False)


def neighbours(ei, n):
    nbrs = [[] for _ in range(n)]
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        nbrs[a].append(b)
    return nbrs


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


@torch.no_grad()
def predict(model, data_list, device, batch=32):
    model.eval()
    out = []
    for b in DataLoader(data_list, batch_size=batch, shuffle=False):
        b = b.to(device)
        out.append(torch.sigmoid(model(b)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "r25", "mirror_test.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = load_test_recs()
    print(f"[data] {len(recs)} test molecules")

    R = random_rotation(0)
    variants = {"orig": [], "reflect": [], "rotate": [], "sinflip": []}
    sin_abs, sin_signed = [], []
    for r in recs:
        ei, pos = r["edge_index"], r["pos"]
        nbrs = neighbours(ei, r["x"].shape[0])
        f0 = edge_stats(ei, pos, nbrs)
        pr = pos.clone(); pr[..., 2] = -pr[..., 2]
        fr = edge_stats(ei, pr, nbrs)
        prot = (pos.numpy().reshape(-1, 3) @ R.T).reshape(pos.shape)
        fo = edge_stats(ei, torch.from_numpy(prot), nbrs)
        fs = f0.copy(); fs[:, SIN_COL] = -fs[:, SIN_COL]
        for k, f in (("orig", f0), ("reflect", fr), ("rotate", fo), ("sinflip", fs)):
            variants[k].append(Data(x=r["x"], edge_index=ei, edge_attr=r["edge_attr"],
                                    edge_feat=torch.from_numpy(f), y=r["y"].unsqueeze(0)))
        sin_abs.append(np.abs(f0[:, SIN_COL]))
        sin_signed.append(f0[:, SIN_COL])

    sin_all = np.concatenate(sin_abs)
    print(f"[sin] |sin phi| median {np.median(sin_all):.3f}, "
          f"frac>0.1 {np.mean(sin_all > 0.1):.2f}")

    res = {}
    for s in args.seeds:
        ck = torch.load(os.path.join(PROJ, "results", "r25", "edge_gin", f"seed{s}",
                                     "best_model.pt"), map_location=device, weights_only=False)
        model = build("edge_gin", dropout=0.15, edge_dim=66).to(device)
        model.load_state_dict(ck["model_state_dict"])
        P = {k: predict(model, v, device) for k, v in variants.items()}
        d = {k: np.abs(P["orig"] - P[k]) for k in ("reflect", "rotate", "sinflip")}
        res[f"seed{s}"] = {
            "mean_abs_dp_reflect": float(d["reflect"].mean()),
            "max_abs_dp_reflect": float(d["reflect"].max()),
            "mean_abs_dp_rotate": float(d["rotate"].mean()),
            "mean_abs_dp_sinflip": float(d["sinflip"].mean()),
            "max_abs_dp_sinflip": float(d["sinflip"].max()),
            "per_label_corr_reflect": float(np.corrcoef(P["orig"].ravel(),
                                                        P["reflect"].ravel())[0, 1]),
        }
        print(f"[seed{s}] reflect {res[f'seed{s}']['mean_abs_dp_reflect']:.2e} "
              f"(max {res[f'seed{s}']['max_abs_dp_reflect']:.3f}) | "
              f"rotate {res[f'seed{s}']['mean_abs_dp_rotate']:.2e} | "
              f"sinflip {res[f'seed{s}']['mean_abs_dp_sinflip']:.2e}")

    m = {k: float(np.mean([v[k] for v in res.values()]))
         for k in ("mean_abs_dp_reflect", "mean_abs_dp_rotate", "mean_abs_dp_sinflip")}
    out = {"model": "edge_gin (R2.5)", "n_test": len(recs), "seeds": args.seeds,
           "sin_abs_median": float(np.median(sin_all)),
           "sin_frac_abs_gt_0.1": float(np.mean(sin_all > 0.1)),
           "mean_over_seeds": m, "per_seed": res,
           "criterion": "eligible for the A1 stereo panel iff mean_abs_dp_reflect >= 1e-3",
           "verdict": "ELIGIBLE" if m["mean_abs_dp_reflect"] >= 1e-3 else "NOT ELIGIBLE"}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"mean_over_seeds": m, "verdict": out["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
