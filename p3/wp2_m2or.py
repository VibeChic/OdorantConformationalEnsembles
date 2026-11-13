"""WP2: M2OR receptor breadth as external directional validation.

Primary (limited) analysis: beta-binomial GLM of k_active / n_tested on S_conf
for the 294 M2OR ligands that also have 3D observables. The 61 wide-panel
ligands are reported separately (only 42 of them have 3D observables).

Outputs under paper3/results/wp2_m2or/.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import CACHE, DATA, RESULTS  # noqa: E402
from p3.stats import (  # noqa: E402
    beta_binomial_bootstrap,
    bootstrap_p_value,
    fit_beta_binomial,
    percentile_ci,
)

OUT = RESULTS / "wp2_m2or"
OUT.mkdir(parents=True, exist_ok=True)


def ensure_inchikey(df: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE / "inchikey_map.csv"
    if cache.exists():
        mp = pd.read_csv(cache)
        return df.merge(mp, on="name", how="left")
    r1 = pd.read_csv(DATA / "3d" / "run1" / "run1_molecules.csv", low_memory=False).set_index("name")
    rows = []
    for name in df["name"]:
        smi = r1.loc[name, "canonical_smiles"] if name in r1.index else None
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        rows.append({"name": name, "inchikey": Chem.MolToInchiKey(mol) if mol is not None else None})
    mp = pd.DataFrame(rows)
    mp.to_csv(cache, index=False)
    return df.merge(mp, on="name", how="left")


def build_matrix() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    m = pd.read_csv(DATA / "external" / "m2or" / "m2or_receptor_ligand.csv", index_col=0)
    tested = m.notna().sum(axis=0)
    active = (m == 1).sum(axis=0)
    return m, tested, active


def design(df: pd.DataFrame, keep_cap: bool = True) -> tuple[np.ndarray, list[str], float, float]:
    s = df["S_conf"].astype(float)
    z = (s - s.mean()) / s.std(ddof=0)
    cols = [np.ones(len(df)), z.to_numpy(), (z * z).to_numpy()]
    names = ["intercept", "S_hat", "S_hat2"]
    for c in ["mw", "clogp", "n_ha", "tpsa"]:
        vals = df[c].astype(float)
        cols.append(((vals - vals.mean()) / vals.std(ddof=0)).to_numpy())
        names.append(c + "_z")
    if keep_cap:
        cols.append(df["capped20"].to_numpy())
        names.append("capped20")
    cols.append(df["run_pass_1b"].to_numpy())
    names.append("run_pass_1b")
    return np.column_stack(cols), names, float(s.mean()), float(s.std(ddof=0))


def fit_and_summary(df, label, n_boot=500, keep_cap=True, cluster_resample=True):
    X, names, sm, ss = design(df, keep_cap=keep_cap)
    k = df["k_active"].to_numpy(dtype=float)
    n = df["n_tested"].to_numpy(dtype=float)
    fit = fit_beta_binomial(X, k, n)
    # receptor-cluster bootstrap: resample receptor rows and recompute k/n
    if cluster_resample:
        m, tested, active = build_matrix()
        rng = np.random.default_rng(20261025)
        boot = []
        receptors = np.asarray(m.index)
        for _ in range(n_boot):
            rs = rng.choice(receptors, size=len(receptors), replace=True)
            sub = m.loc[rs]
            kk = (sub == 1).sum(axis=0)
            nn = sub.notna().sum(axis=0)
            d = df.copy()
            d["k_active"] = d["inchikey"].map(kk).astype(float)
            d["n_tested"] = d["inchikey"].map(nn).astype(float)
            d = d[(d.n_tested > 0) & d.k_active.notna()]
            if len(d) < 20:
                boot.append(np.full(len(names) + 1, np.nan))
                continue
            Xb, _, _, _ = design(d, keep_cap=keep_cap)
            try:
                f = fit_beta_binomial(Xb, d["k_active"].to_numpy(float), d["n_tested"].to_numpy(float))
                boot.append(np.r_[f.beta, f.phi])
            except Exception:
                boot.append(np.full(len(names) + 1, np.nan))
        boot = np.asarray(boot)
    else:
        boot = beta_binomial_bootstrap(X, k, n, n_boot=n_boot, seed=20261025)
    rows = []
    for j, nm in enumerate(names):
        bs = boot[:, j]
        lo, hi = percentile_ci(bs)
        rows.append({
            "term": nm,
            "beta": float(fit.beta[j]),
            "se_boot": float(np.nanstd(bs, ddof=1)),
            "ci_lo": lo,
            "ci_hi": hi,
            "odds_ratio": float(math.exp(fit.beta[j])) if abs(fit.beta[j]) < 50 else float("nan"),
            "p_boot": bootstrap_p_value(bs, 0.0),
        })
    coef = pd.DataFrame(rows)
    b2 = coef[coef.term == "S_hat2"].iloc[0].to_dict()
    peak = None
    if fit.beta[2] < 0:
        sh = -fit.beta[1] / (2 * fit.beta[2])
        sp = sm + ss * sh
        peak = {"S_hat_peak": float(sh), "S_conf_peak": float(sp), "N_eff_peak": float(math.exp(sp))}
    return {
        "label": label,
        "n": int(len(df)),
        "n_wide50": int((df["n_tested"] >= 50).sum()) if "n_tested" in df else None,
        "fit": fit,
        "coef": coef,
        "boot": boot,
        "beta2": b2,
        "peak": peak,
        "names": names,
        "s_mean": sm,
        "s_std": ss,
    }


def main(n_boot: int = 500) -> None:
    obs = pd.read_csv(CACHE / "ensemble_observables.csv")
    obs["run_pass_1b"] = (obs["run_pass"].astype(str) == "1b").astype(float)
    obs = ensure_inchikey(obs)
    m, tested, active = build_matrix()
    obs["n_tested"] = obs["inchikey"].map(tested)
    obs["k_active"] = obs["inchikey"].map(active)
    d = obs[obs["n_tested"].notna() & (obs["n_tested"] > 0)].copy()
    d["b_tilde"] = d["k_active"] / d["n_tested"]
    print(f"M2OR overlap with 3D: {len(d)}; wide>=50: {(d.n_tested>=50).sum()}", flush=True)

    primary = fit_and_summary(d, "all_3d_overlap", n_boot=n_boot, keep_cap=True, cluster_resample=True)
    primary["coef"].to_csv(OUT / "wp2_coefficients_all.csv", index=False)
    wide = d[d.n_tested >= 50].copy()
    wide_res = fit_and_summary(wide, "wide50_3d_overlap", n_boot=n_boot, keep_cap=False, cluster_resample=True)
    wide_res["coef"].to_csv(OUT / "wp2_coefficients_wide50.csv", index=False)

    # human-only sensitivity: receptors annotated as homo sapiens in the raw M2OR table
    raw_path = Path(os.environ.get("M2OR_RAW_CSV", DATA / "M2OR_20230428.csv"))
    human_res = None
    if raw_path.exists():
        raw = pd.read_csv(raw_path, sep=";", dtype=str, low_memory=False)
        human = raw.loc[raw["species"].str.contains("homo sapiens", case=False, na=False), "Uniprot ID"].dropna().unique()
        human = [u for u in human if u in m.index]
        mh = m.loc[human]
        th = mh.notna().sum(axis=0)
        kh = (mh == 1).sum(axis=0)
        dh = d.copy()
        dh["n_tested"] = dh["inchikey"].map(th)
        dh["k_active"] = dh["inchikey"].map(kh)
        dh = dh[(dh.n_tested > 0) & dh.k_active.notna()]
        if len(dh) >= 50:
            human_res = fit_and_summary(dh, "human_only", n_boot=max(100, n_boot // 2), keep_cap=True, cluster_resample=False)
    else:
        print(f"M2OR raw table not found at {raw_path}; skipping human-only sensitivity", flush=True)
    if human_res is not None:
        human_res["coef"].to_csv(OUT / "wp2_coefficients_human.csv", index=False)

    def slim(res):
        if res is None:
            return None
        return {
            "label": res["label"], "n": res["n"], "n_wide50": res["n_wide50"],
            "beta2": res["beta2"], "peak": res["peak"],
            "alpha_or_phi": float(res["fit"].phi), "converged": bool(res["fit"].converged),
        }

    summary = {
        "n_overlap": int(len(d)),
        "n_wide50_overlap": int((d.n_tested >= 50).sum()),
        "n_tested_median": float(d.n_tested.median()),
        "b_or_mean": float(d.k_active.mean()),
        "b_tilde_mean": float(d.b_tilde.mean()),
        "primary": slim(primary),
        "wide50": slim(wide_res),
        "human_only": slim(human_res),
        "n_human_receptors": int(len(human)),
        "bootstrap": "receptor-cluster bootstrap for primary/wide; molecule bootstrap for human",
        "note": "Only 42 of the 61 wide-panel M2OR ligands have 3D observables; the other 19 are M2OR-novel and cannot enter WP2 without new 3D generation.",
    }
    (OUT / "wp2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    d[["name", "inchikey", "n_tested", "k_active", "b_tilde", "S_conf", "N_eff", "capped20", "run_pass"]].to_csv(OUT / "wp2_ligand_table.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    main(n_boot=args.n_boot)
