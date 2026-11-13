"""WP1: inverted-U test on semantic complexity (L_sem).

Primary model:
  L_sem ~ S_hat + S_hat^2 + M_w + cLogP + N_HA + TPSA
          + capped20 + has_censored + run_pass_1b + n_unspecified_stereo
  negative-binomial (NB2), log link, molecule-level bootstrap.

Outputs under paper3/results/wp1_semantic/.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import CACHE, RESULTS  # noqa: E402
from p3.stats import (  # noqa: E402
    bootstrap_p_value,
    fit_nb,
    nb_bootstrap,
    percentile_ci,
)

OUT = RESULTS / "wp1_semantic"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_COVS = ["mw", "clogp", "n_ha", "tpsa"]
TECH_COVS = ["capped20", "has_censored", "run_pass_1b", "n_unspec_z"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CACHE / "ensemble_observables.csv")
    df["run_pass_1b"] = (df["run_pass"].astype(str) == "1b").astype(float)
    for c in ["mw", "clogp", "n_ha", "tpsa", "n_unspecified_stereo"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
    df["n_unspec_z"] = df["n_unspecified_stereo_z"]
    return df


def design(df: pd.DataFrame, obs: str = "S_conf", with_rotors: bool = False, keep_cap: bool = True):
    s = df[obs].astype(float)
    z = (s - s.mean()) / s.std(ddof=0)
    cols = [np.ones(len(df)), z.to_numpy(), (z * z).to_numpy()]
    names = ["intercept", "S_hat", "S_hat2"]
    for c in PRIMARY_COVS:
        cols.append(df[c + "_z"].to_numpy())
        names.append(c + "_z")
    if with_rotors:
        cols.append(((df["n_rotors"] - df["n_rotors"].mean()) / df["n_rotors"].std(ddof=0)).to_numpy())
        names.append("n_rotors_z")
    if keep_cap:
        cols.append(df["capped20"].to_numpy())
        names.append("capped20")
    cols.append(df["has_censored"].to_numpy())
    names.append("has_censored")
    cols.append(df["run_pass_1b"].to_numpy())
    names.append("run_pass_1b")
    cols.append(df["n_unspec_z"].to_numpy())
    names.append("n_unspec_z")
    return np.column_stack(cols), names, float(s.mean()), float(s.std(ddof=0))


def summarize(fit, X, y, names, n_boot=1000, seed=20261025, label="primary", progress=True):
    t0 = time.time()
    boot = nb_bootstrap(X, y, n_boot=n_boot, seed=seed, init=np.r_[fit.beta, np.log(fit.alpha)],
                        progress=(lambda b, n: print(f"  {label} bootstrap {b}/{n} ({time.time()-t0:.0f}s)", flush=True)) if progress else None)
    rows = []
    for j, nm in enumerate(names):
        b = fit.beta[j]
        bs = boot[:, j]
        lo, hi = percentile_ci(bs)
        rows.append({
            "term": nm,
            "beta": float(b),
            "se_boot": float(np.nanstd(bs, ddof=1)),
            "ci_lo": lo,
            "ci_hi": hi,
            "irr": float(math.exp(b)) if abs(b) < 50 else float("nan"),
            "p_boot": bootstrap_p_value(bs, 0.0),
        })
    coef = pd.DataFrame(rows)
    alpha_ci = percentile_ci(boot[:, -1])
    return coef, boot, alpha_ci


def peak_from_fit(fit, s_mean, s_std):
    b1, b2 = float(fit.beta[1]), float(fit.beta[2])
    if b2 >= 0:
        return None
    s_hat_peak = -b1 / (2.0 * b2)
    s_peak = s_mean + s_std * s_hat_peak
    return {
        "S_hat_peak": float(s_hat_peak),
        "S_conf_peak": float(s_peak),
        "N_eff_peak": float(math.exp(s_peak)),
    }


def peak_bootstrap(boot, s_mean, s_std, s_min=None, s_max=None):
    b1 = boot[:, 1]
    b2 = boot[:, 2]
    ok = np.isfinite(b1) & np.isfinite(b2) & (b2 < 0)
    s_hat = -b1[ok] / (2.0 * b2[ok])
    s_peak = s_mean + s_std * s_hat
    ne = np.exp(np.clip(s_peak, -20, 20))
    inside = np.ones(len(ne), dtype=bool)
    if s_min is not None and s_max is not None:
        inside = (s_peak >= s_min) & (s_peak <= s_max)
    out = {
        "n_valid": int(ok.sum()),
        "n_inside_observed_range": int(inside.sum()),
        "N_eff_peak_ci_lo": float(np.percentile(ne, 2.5)) if len(ne) else None,
        "N_eff_peak_ci_hi": float(np.percentile(ne, 97.5)) if len(ne) else None,
        "N_eff_peak_median": float(np.median(ne)) if len(ne) else None,
    }
    if inside.any():
        out["N_eff_peak_inside_ci_lo"] = float(np.percentile(ne[inside], 2.5))
        out["N_eff_peak_inside_ci_hi"] = float(np.percentile(ne[inside], 97.5))
    return out


def curve(fit, s_mean, s_std, names):
    grid = np.linspace(0.0, 3.0, 121)
    X = np.zeros((len(grid), len(names)))
    X[:, 0] = 1.0
    X[:, 1] = (grid - s_mean) / s_std
    X[:, 2] = X[:, 1] ** 2
    mu = fit.predict(X)
    return pd.DataFrame({"S_conf": grid, "N_eff": np.exp(grid), "pred_L_sem": mu})


def binned(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.qcut(d["N_eff"], q=n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True)["L_sem"]
    out = g.agg(["count", "mean", "std"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["bin_left"] = out["bin"].apply(lambda x: x.left)
    out["bin_right"] = out["bin"].apply(lambda x: x.right)
    return out.drop(columns=["bin"])


def fit_simple(df, obs, label, with_rotors=False, keep_cap=True, n_boot=0):
    X, names, sm, ss = design(df, obs=obs, with_rotors=with_rotors, keep_cap=keep_cap)
    y = df["L_sem"].to_numpy(dtype=float)
    fit = fit_nb(X, y)
    if n_boot and n_boot > 0:
        coef, boot, alpha_ci = summarize(fit, X, y, names, n_boot=n_boot, label=label)
    else:
        coef = pd.DataFrame([{
            "term": nm, "beta": float(fit.beta[j]), "se_boot": np.nan,
            "ci_lo": np.nan, "ci_hi": np.nan,
            "irr": float(math.exp(fit.beta[j])) if abs(fit.beta[j]) < 50 else np.nan,
            "p_boot": np.nan,
        } for j, nm in enumerate(names)])
        boot = None
        alpha_ci = (np.nan, np.nan)
    peak = peak_from_fit(fit, sm, ss)
    return fit, coef, boot, names, sm, ss, peak, alpha_ci


def main(n_boot: int = 1000) -> None:
    df = load()
    y = df["L_sem"].to_numpy(dtype=float)

    # Primary
    X, names, sm, ss = design(df, "S_conf")
    fit = fit_nb(X, y)
    print("primary fit:", fit.converged, "alpha", fit.alpha, "loglik", fit.loglik, flush=True)
    coef, boot, alpha_ci = summarize(fit, X, y, names, n_boot=n_boot, label="primary")
    coef.to_csv(OUT / "wp1_coefficients.csv", index=False)
    peak = peak_from_fit(fit, sm, ss)
    if peak is None:
        peak_ci = {"n_valid": int(np.sum(boot[:, 2] < 0)), "note": "point beta2 >= 0; no interior peak"}
    else:
        peak_ci = peak_bootstrap(boot, sm, ss, float(df["S_conf"].min()), float(df["S_conf"].max()))
    curve(fit, sm, ss, names).to_csv(OUT / "wp1_curve.csv", index=False)
    binned(df).to_csv(OUT / "wp1_binned.csv", index=False)

    b2 = coef[coef.term == "S_hat2"].iloc[0].to_dict()
    summary = {
        "n": int(len(df)),
        "model": "NB2 log link",
        "primary": "L_sem ~ S_hat + S_hat^2 + M_w + cLogP + N_HA + TPSA + capped20 + has_censored + run_pass_1b + n_unspec_z",
        "fit_converged": bool(fit.converged),
        "alpha": float(fit.alpha),
        "alpha_ci": [float(alpha_ci[0]), float(alpha_ci[1])],
        "loglik": float(fit.loglik),
        "aic": float(2 * len(fit.beta) + 2 - 2 * fit.loglik),
        "beta2": b2,
        "peak": peak,
        "peak_bootstrap": peak_ci,
        "n_boot": int(n_boot),
        "s_mean": sm,
        "s_std": ss,
        "decision_rule": "beta2 < 0 and upper 95% bootstrap CI < 0",
        "decision": bool(b2["ci_hi"] < 0) if np.isfinite(b2["ci_hi"]) else None,
    }
    (OUT / "wp1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Secondary: add n_rotors (E2)
    fit2, coef2, boot2, names2, sm2, ss2, peak2, alpha2 = fit_simple(df, "S_conf", "with_rotors", with_rotors=True, n_boot=0)
    coef2.to_csv(OUT / "wp1_coefficients_with_rotors.csv", index=False)
    # Secondary: other ensemble observables
    uni = []
    for obs in ["rmsd_ab", "tr_sigma_npr", "sigma_sasa_polar", "sigma2_rg"]:
        try:
            f, c, _, _, _, _, pk, _ = fit_simple(df, obs, obs, n_boot=0)
            row = c[c.term == "S_hat2"].iloc[0].to_dict()
            row.update({"observable": obs, "peak": pk, "converged": bool(f.converged)})
            uni.append(row)
        except Exception as e:
            uni.append({"observable": obs, "error": f"{type(e).__name__}: {e}"})
    pd.DataFrame(uni).to_csv(OUT / "wp1_other_observables.csv", index=False)

    # Sensitivity analyses
    sens = []
    def add_sens(label, d, obs="S_conf", keep_cap=True):
        try:
            f, c, _, _, _, _, pk, _ = fit_simple(d, obs, label, n_boot=0, keep_cap=keep_cap)
            row = c[c.term == "S_hat2"].iloc[0].to_dict()
            row.update({"analysis": label, "n": int(len(d)), "peak": pk, "converged": bool(f.converged)})
            sens.append(row)
        except Exception as e:
            sens.append({"analysis": label, "n": int(len(d)), "error": f"{type(e).__name__}: {e}"})

    add_sens("non_capped", df[df.capped20 == 0].copy(), keep_cap=False)
    add_sens("run_pass_1b_only", df[df.run_pass.astype(str) == "1b"].copy())
    add_sens("n_unspec_lt2", df[df.n_unspecified_stereo < 2].copy())
    # C4 proxy: normalized entropy S/ln(20)
    dnorm = df.copy()
    dnorm["S_norm"] = dnorm["S_conf"] / math.log(20.0)
    add_sens("S_norm_ln20", dnorm, obs="S_norm", keep_cap=False)
    pd.DataFrame(sens).to_csv(OUT / "wp1_sensitivity.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    main(n_boot=args.n_boot)
