#!/usr/bin/env python3
"""Molecule-level paired CIs on seed-averaged predictions, for the 10-seed arms.

Two different questions, deliberately answered separately:

  seed level        sign-flip permutation over seeds (p2.stats.sign_flip_test)
                    -> "does A beat B across training runs"
  molecule level    stratified bootstrap over test molecules of the
                    seed-averaged predictor -> "would this hold on new molecules"

The manuscript reports both and never averages bootstrap endpoints across seeds.

Outputs results/bootstrap_ci.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, PROJ)

from p2.stats import sign_flip_test, stratified_bootstrap  # noqa: E402
from p2.exp31_subgroups import order_indices, canon, have3d_smiles  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ARMS = {
    "r1_gin": ("results/r1/gin/seed{s}/test_preds.npz", "csv"),
    "en3": ("results/r25/scalar_gin/seed{s}/test_preds.npz", "cache"),
    "r25_edge": ("results/r25/edge_gin/seed{s}/test_preds.npz", "cache"),
    "r25_shuffled": ("results/r25/edge_gin_shuffled/seed{s}/test_preds.npz", "cache"),
    "schnet": ("results/r2_sc/schnet/seed{s}/test_preds.npz", "cache"),
    "dimenet": ("results/r2_dimenet/dimenet/seed{s}/test_preds.npz", "cache"),
    "r3_shuffle": ("results/s1_mil/shuffle/seed{s}/test_preds_det.npz", "cache"),
    "r3_boltz": ("results/s1_mil/boltz/seed{s}/test_preds.npz", "cache"),
    "r3_attn": ("results/s1_mil/attn/seed{s}/test_preds.npz", "cache"),
}

PAIRS = [("en3", "r1_gin"), ("r25_edge", "r1_gin"), ("r25_edge", "r25_shuffled"),
         ("r3_shuffle", "r1_gin"), ("r3_shuffle", "en3"), ("en3", "r25_edge")]


def seed_metrics(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(PROJ, pat.format(s="*")))):
        seed = int(os.path.basename(os.path.dirname(f)).replace("seed", ""))
        z = np.load(f)
        out[seed] = (z["p"], z["y"])
    return out


def load_arm(pat, kind, te, labels, smi2row):
    seeds = seed_metrics(pat)
    order = order_indices(kind, te, labels, smi2row)
    Ycsv = te[labels].values.astype("float32")
    P = []
    for s in sorted(seeds):
        p, y = seeds[s]
        if not np.array_equal(y, Ycsv[order]):
            raise RuntimeError(f"{pat} seed {s}: labels misaligned")
        P.append(p)
    return np.mean(P, axis=0), Ycsv[order]


def main():
    te = pd.read_csv(os.path.join(REPO, "data", "full", "test_gs_lf.csv"))
    labels = [c for c in te.columns
              if c not in ("nonStereoSMILES", "descriptors", "SDF_FileName")]
    smi2row = {canon(s): i for i, s in enumerate(te["nonStereoSMILES"])}

    P, Y, S = {}, {}, {}
    for k, (pat, kind) in ARMS.items():
        P[k], Y[k] = load_arm(pat, kind, te, labels, smi2row)
        S[k] = {s: np.nan for s in []}
    # seed-level differences
    raw = {k: seed_metrics(pat) for k, (pat, _) in ARMS.items()}
    out = {"note": "molecule-level stratified bootstrap on seed-averaged predictions; "
                   "seed-level sign-flip permutation reported separately",
           "n_molecules": int(Y["r1_gin"].shape[0]), "pairs": {}}
    for a, b in PAIRS:
        if a not in P or b not in P:
            continue
        common = sorted(set(raw[a]) & set(raw[b]))
        d = []
        from sklearn.metrics import average_precision_score

        def macro(y, p):
            return float(np.mean([average_precision_score(y[:, j], p[:, j])
                                  for j in range(y.shape[1]) if y[:, j].sum() >= 1]))

        for s in common:
            d.append(macro(raw[a][s][1], raw[a][s][0]) - macro(raw[b][s][1], raw[b][s][0]))
        sf = sign_flip_test(d)
        boot = stratified_bootstrap(Y[a], P[a], n_boot=300, seed=0)
        boot_b = stratified_bootstrap(Y[b], P[b], n_boot=300, seed=0)
        out["pairs"][f"{a} - {b}"] = {
            "n_seeds": len(common),
            "seed_level_mean": sf["mean"], "seed_level_sd": sf["sd"], "seed_level_p": sf["p"],
            "delta_stratified": boot["point"] - boot_b["point"],
            "delta_lo": boot["lo"] - boot_b["hi"], "delta_hi": boot["hi"] - boot_b["lo"],
            "a_point": boot["point"], "b_point": boot_b["point"],
        }
        print(f"{a} - {b}: seeds n={len(common)} mean {sf['mean']:+.4f} p={sf['p']:.4f} | "
              f"molecule Δ {out['pairs'][f'{a} - {b}']['delta_stratified']:+.4f} "
              f"[{out['pairs'][f'{a} - {b}']['delta_lo']:+.4f}, "
              f"{out['pairs'][f'{a} - {b}']['delta_hi']:+.4f}]")
    json.dump(out, open(os.path.join(PROJ, "results", "bootstrap_ci.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
