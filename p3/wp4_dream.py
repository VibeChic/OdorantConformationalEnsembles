"""WP4: DREAM psychophysics.

Tests whether the six ensemble observables add predictive value over 2D
descriptors for 21 DREAM targets (338 labelled molecules; 304 with 3D).
Primary targets: INTENSITY/STRENGTH and VALENCE/PLEASANTNESS. The other 19
are exploratory with BH-FDR.

Outputs under paper3/results/wp4_dream/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import CACHE, DATA, RESULTS  # noqa: E402

OUT = RESULTS / "wp4_dream"
OUT.mkdir(parents=True, exist_ok=True)

BASE = ["mw", "clogp", "n_ha", "tpsa", "n_rotors"]
ENS = ["S_conf", "rmsd_ab", "sigma2_rg", "tr_sigma_npr", "sigma_sasa_polar", "N_eff"]
ALPHAS = np.logspace(-3, 3, 13)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        i = n - rank + 1
        val = min(prev, p[idx] * n / i)
        q[idx] = val
        prev = val
    return q


def cv_predict(X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]):
    pred = np.full(len(y), np.nan, dtype=float)
    for tr, te in folds:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def r2(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def delta_ci(y, pb, pf, n_boot=1000, seed=20261025):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(y[idx]) == 0:
            continue
        vals.append(r2(y[idx], pf[idx]) - r2(y[idx], pb[idx]))
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0)) if len(vals) else np.nan
    return float(lo), float(hi), float(p), len(vals)


def run_target(df: pd.DataFrame, target: str, folds, n_boot=1000) -> dict:
    y = df[target].to_numpy(float)
    Xb = df[BASE].to_numpy(float)
    Xf = df[BASE + ENS].to_numpy(float)
    pb = cv_predict(Xb, y, folds)
    pf = cv_predict(Xf, y, folds)
    rb, rf = r2(y, pb), r2(y, pf)
    lo, hi, p, nb = delta_ci(y, pb, pf, n_boot=n_boot)
    return {
        "target": target, "n": int(len(df)),
        "r2_base": rb, "r2_full": rf, "delta_r2": rf - rb,
        "delta_ci_lo": lo, "delta_ci_hi": hi, "p_boot": p, "n_boot_valid": nb,
        "pred_base": pb, "pred_full": pf, "y": y,
    }


def main(n_boot: int = 1000) -> None:
    obs = pd.read_csv(CACHE / "ensemble_observables.csv")
    mp = pd.read_csv(CACHE / "inchikey_map.csv")
    obs = obs.merge(mp, on="name", how="left")
    dm = pd.read_csv(DATA / "external" / "dream" / "dream_dataset.csv")
    df = dm.merge(obs, on="inchikey", how="inner", suffixes=("", "_obs"))
    print(f"DREAM labelled {len(dm)}; with 3D {len(df)}", flush=True)

    targets = [c for c in dm.columns if c not in ("cid", "SMILES", "inchikey")]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=20261025).split(df))
    rows = []
    preds = []
    for t in targets:
        r = run_target(df, t, folds, n_boot=n_boot)
        rows.append({k: v for k, v in r.items() if k not in ("pred_base", "pred_full", "y")})
        preds.append(pd.DataFrame({
            "inchikey": df["inchikey"].to_numpy(),
            "target": t,
            "y": r["y"],
            "pred_base": r["pred_base"],
            "pred_full": r["pred_full"],
        }))
        print(f"  {t}: R2 base {r['r2_base']:.3f} full {r['r2_full']:.3f} dR2 {r['delta_r2']:+.3f} CI [{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]", flush=True)
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p_boot"].to_numpy())
    res.to_csv(OUT / "wp4_targets.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "wp4_predictions.csv", index=False)

    # Dilution-level secondary: 1/1000 intensity
    dil = pd.read_csv(DATA / "external" / "dream" / "dream_dil_1_1000.csv")
    smiles = pd.read_csv(DATA / "external" / "dream" / "dream_smiles.csv")
    dil = dil.merge(smiles[["CID", "inchikey"]], on="CID", how="left")
    dd = dil.merge(obs, on="inchikey", how="inner", suffixes=("", "_obs"))
    dil_res = None
    if len(dd) >= 50 and "INTENSITY/STRENGTH" in dd.columns:
        folds_d = list(KFold(n_splits=5, shuffle=True, random_state=20261025).split(dd))
        dil_res = run_target(dd, "INTENSITY/STRENGTH", folds_d, n_boot=n_boot)
        dil_res = {k: v for k, v in dil_res.items() if k not in ("pred_base", "pred_full", "y")}
        dil_res["n_dilution"] = int(len(dd))

    summary = {
        "n_labelled": int(len(dm)),
        "n_with_3d": int(len(df)),
        "n_targets": int(len(targets)),
        "targets": rows,
        "dilution_1_1000_intensity": dil_res,
        "model": "StandardScaler + RidgeCV(alphas=1e-3..1e3), 5-fold CV",
        "base_features": BASE,
        "ensemble_features": ENS,
        "note": "delta R2 is full minus base on the same 5 folds; CI is molecule-level bootstrap of out-of-fold predictions.",
    }
    (OUT / "wp4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "targets"}, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    main(n_boot=args.n_boot)
