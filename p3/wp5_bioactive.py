"""WP5 (A2 appendix): model-agnostic surrogate for the "bioactive conformer".

For each of the 138 semantic labels, train a conformer-level ridge model on
training molecules and select the highest-scoring conformer (c*) for each
positive test molecule. Report the fraction c* != 1 (index 0 = lowest energy),
the Boltzmann-prior baseline (argmax pi), and a label-shuffle control.

This is explicitly an exploratory appendix, not a biophysical measurement.
Outputs under paper3/results/wp5_bioactive/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import CACHE, RESULTS, load_cache  # noqa: E402

OUT = RESULTS / "wp5_bioactive"
OUT.mkdir(parents=True, exist_ok=True)

FEATS = ["npr1", "npr2", "rg", "sasa_polar"]


def load_labels() -> pd.DataFrame:
    recs, _ = load_cache()
    rows = []
    for r in recs:
        y = np.asarray(r["y"].numpy(), dtype=int)
        rows.append([str(r["name"])] + y.tolist())
    cols = ["name"] + [f"label_{i}" for i in range(138)]
    return pd.DataFrame(rows, columns=cols)


def evaluate_label(feat: pd.DataFrame, labels: pd.DataFrame, label: str, n_splits=5, seed=20261025, n_shuffle=1):
    # molecule-level table
    mol = labels[["name", label]].copy()
    mol = mol.merge(feat.groupby("name").agg(n_conf=("conf_idx", "size")).reset_index(), on="name")
    y = mol[label].to_numpy(int)
    n_pos = int(y.sum())
    if n_pos < 20 or len(np.unique(y)) < 2:
        return None
    names = mol["name"].to_numpy()
    groups = feat.groupby("name").indices
    feat_names = feat["name"].to_numpy()
    Xall = feat[FEATS].to_numpy(float)
    conf = feat["conf_idx"].to_numpy()
    dE = feat["dE"].to_numpy()
    pi = feat["pi"].to_numpy()

    def run(y_vec, seed_):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_)
        chosen = []
        de = []
        pos_by_name = pd.Series(y_vec == 1, index=names)
        feat_pos = pos_by_name.reindex(feat_names).to_numpy()
        for tr, te in skf.split(names, y_vec):
            tr_names = names[tr]
            te_pos_names = [names[i] for i in te if y_vec[i] == 1]
            if not te_pos_names:
                continue
            tr_idx = np.concatenate([groups[n] for n in tr_names])
            Xtr = Xall[tr_idx]
            ytr = feat_pos[tr_idx].astype(float)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            model = Ridge(alpha=1.0)
            model.fit((Xtr - mu) / sd, ytr)
            for n in te_pos_names:
                ii = groups[n]
                s = model.predict((Xall[ii] - mu) / sd)
                j = int(np.argmax(s))
                chosen.append(int(conf[ii[j]]))
                de.append(float(dE[ii[j]] - dE[ii[0]]))
        return np.asarray(chosen), np.asarray(de)

    chosen, de = run(y, seed)
    prop = float(np.mean(chosen != 0)) if len(chosen) else float("nan")
    # Boltzmann prior baseline: argmax pi per molecule, positive molecules only
    prior_choice = []
    for n in names[y == 1]:
        ii = groups[n]
        j = int(np.argmax(pi[ii]))
        prior_choice.append(int(conf[ii[j]]))
    prior_prop = float(np.mean(np.asarray(prior_choice) != 0)) if prior_choice else float("nan")

    shuf_props = []
    rng = np.random.default_rng(seed)
    for s in range(n_shuffle):
        ys = y.copy()
        rng.shuffle(ys)
        ch, _ = run(ys, seed + 100 + s)
        shuf_props.append(float(np.mean(ch != 0)) if len(ch) else float("nan"))

    return {
        "label": label,
        "n_pos": n_pos,
        "n_eval": int(len(chosen)),
        "prop_cstar_not1": prop,
        "prop_prior_not1": prior_prop,
        "prop_shuffle_mean": float(np.nanmean(shuf_props)) if shuf_props else float("nan"),
        "mean_dE_bio": float(np.nanmean(de)) if len(de) else float("nan"),
        "median_dE_bio": float(np.nanmedian(de)) if len(de) else float("nan"),
    }


def main(n_shuffle: int = 1) -> None:
    feat = pd.read_csv(CACHE / "conformer_features.csv")
    labels = load_labels()
    rows = []
    for i in range(138):
        r = evaluate_label(feat, labels, f"label_{i}", n_shuffle=n_shuffle)
        if r is not None:
            rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/138", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wp5_labels.csv", index=False)
    summary = {
        "n_labels": int(len(df)),
        "n_labels_with_ge20_pos": int((df.n_pos >= 20).sum()),
        "mean_prop_cstar_not1": float(df.prop_cstar_not1.mean()),
        "median_prop_cstar_not1": float(df.prop_cstar_not1.median()),
        "weighted_prop_cstar_not1": float(np.average(df.prop_cstar_not1, weights=df.n_eval)),
        "mean_prop_prior_not1": float(df.prop_prior_not1.mean()),
        "mean_prop_shuffle": float(df.prop_shuffle_mean.mean()),
        "mean_dE_bio": float(df.mean_dE_bio.mean()),
        "n_labels_prop_gt_prior_0p1": int((df.prop_cstar_not1 > df.prop_prior_not1 + 0.1).sum()),
        "n_labels_prop_gt_shuffle_0p1": int((df.prop_cstar_not1 > df.prop_shuffle_mean + 0.1).sum()),
        "n_shuffle": int(n_shuffle),
        "note": "A2 appendix only; c* is a model-based choice, not a biophysical measurement. Index 0 is the lowest-energy conformer.",
    }
    (OUT / "wp5_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffle", type=int, default=1)
    args = ap.parse_args()
    main(n_shuffle=args.n_shuffle)
