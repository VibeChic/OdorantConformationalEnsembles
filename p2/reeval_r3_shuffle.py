#!/usr/bin/env python3
"""Re-evaluate the conformer-shuffle arm with a *deterministic* permutation.

Why: the `shuffle` arm in `p2/models/r3.py` permutes the population weights with
`torch.randperm` on every forward pass, including evaluation. Test predictions
therefore differ between calls (measured spread 0.0019 AUCPR on seed 0, i.e.
about 0.4 x the 10-seed SD). The control is still meaningful, but the reported
number must come from a fixed rule.

This script keeps training exactly as it was (random permutation per epoch, a
form of augmentation that forces weight-invariance) and re-evaluates every
checkpoint under one fixed rule:

    reverse    the population vector is reversed inside each molecule, so the
               largest weight goes to the lowest-population conformer. Same
               multiset of weights, maximally wrong assignment.

It writes `metrics_det.json` and `test_preds_det.npz` next to the originals and
never overwrites them.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_sum

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from p2.models.r3 import SchNetMIL            # noqa: E402
from p2.train_r3 import collate, load_split   # noqa: E402
from p2.metrics import macro_metrics          # noqa: E402


@torch.no_grad()
def evaluate_fixed(model, loader, device):
    model.eval()
    ps, ys = [], []
    for b in loader:
        b = b.to(device)
        z = model.enc(b.x[:, 0].long(), b.pos, b.bs)
        raw = z @ model.u.t()
        pi = b.pi.clone()
        cm = b.conf_mol
        for k in range(b.n_mol):
            sel = (cm == k).nonzero(as_tuple=False).view(-1)
            if sel.numel() > 1:
                pi[sel] = pi[sel].flip(0)
        out = scatter_sum(raw * pi.unsqueeze(-1), cm, dim=0, dim_size=b.n_mol) + model.b
        ps.append(torch.sigmoid(out).cpu().numpy())
        ys.append(b.y.cpu().numpy())
    P, Y = np.concatenate(ps), np.concatenate(ys)
    return macro_metrics(Y, P), P, Y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = load_split("test", os.path.join(PROJ, "cache"))
    sl = DataLoader(test, batch_size=32, shuffle=False, collate_fn=collate)
    out = []
    for d in sorted(glob.glob(os.path.join(PROJ, "results", "s1_mil", "shuffle", "seed*"))):
        ck = torch.load(os.path.join(d, "best_model.pt"), map_location=device,
                        weights_only=False)
        m = SchNetMIL(num_classes=138, pool="shuffle").to(device)
        m.load_state_dict(ck["model_state_dict"])
        tm, P, Y = evaluate_fixed(m, sl, device)
        old = json.load(open(os.path.join(d, "metrics.json")))
        rec = {**{k: v for k, v in old.items() if k != "test"},
               "test": tm, "test_macro_aucpr": tm["macro_aucpr"],
               "test_micro_auroc": tm["micro_auroc"],
               "eval_permutation": "reverse (deterministic)",
               "eval_note": "training used a random per-epoch permutation; this is a fixed re-evaluation"}
        json.dump(rec, open(os.path.join(d, "metrics_det.json"), "w"), indent=2)
        np.savez(os.path.join(d, "test_preds_det.npz"), p=P, y=Y)
        out.append((int(os.path.basename(d).replace("seed", "")), old["test_macro_aucpr"],
                    tm["macro_aucpr"]))
    print(f"{'seed':>4s} {'random-eval':>11s} {'fixed-eval':>10s} {'diff':>7s}")
    for s, a, b in out:
        print(f"{s:4d} {a:11.4f} {b:10.4f} {b-a:+7.4f}")
    arr = np.array([x[2] for x in out])
    print(f"fixed-eval mean {arr.mean():.4f} ± {arr.std(ddof=1):.4f} (n={len(arr)})")


if __name__ == "__main__":
    main()
