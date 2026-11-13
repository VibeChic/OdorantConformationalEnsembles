"""WP0: per-molecule ensemble observables.

Computes, for each of the 4,809 molecules in the Paper 2 ensemble cache:
  S_conf, N_eff, <RMSD_ab>, sigma^2(Rg), Tr(Sigma_NPR), sigma(SASA_polar)

Outputs:
  paper3/cache/ensemble_observables.csv
  paper3/cache/observables_manifest.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import (  # noqa: E402
    CACHE,
    SDF_DIR,
    ensure_dirs,
    heavy_atom_coords,
    kabsch_rmsd,
    load_cache,
    load_run1_table,
    load_sdf,
    npr_mass_weighted,
    polar_sasa,
    rdkit_descriptors,
    rg_mass_weighted,
    weighted_mean,
    weighted_var,
)

PT = Chem.GetPeriodicTable()


def match_conformers(cache_pos: np.ndarray, sdf_mols: list[Chem.Mol]) -> tuple[list[int], float]:
    """Match each cache conformer to a unique SDF conformer via heavy-atom RMSD."""
    sdf_heavy = [heavy_atom_coords(m, 0) for m in sdf_mols]
    n_cache = cache_pos.shape[0]
    n_sdf = len(sdf_heavy)
    cost = np.full((n_cache, n_sdf), 1e6, dtype=float)
    for i in range(n_cache):
        for j in range(n_sdf):
            if cache_pos[i].shape == sdf_heavy[j].shape:
                cost[i, j] = kabsch_rmsd(cache_pos[i], sdf_heavy[j], allow_reflection=False)
    rows, cols = linear_sum_assignment(cost)
    mapping = [-1] * n_cache
    worst = 0.0
    for i, j in zip(rows, cols):
        mapping[i] = int(j)
        worst = max(worst, float(cost[i, j]))
    if any(x < 0 for x in mapping):
        raise ValueError("could not match all cache conformers to SDF conformers")
    return mapping, worst


def compute_one(name: str, rec: dict, row: pd.Series, split: str) -> dict:
    pos = np.asarray(rec["pos"].numpy(), dtype=float)
    pi = np.asarray(rec["pi"].numpy(), dtype=float)
    pi = pi / pi.sum()
    n_conf = int(pos.shape[0])

    # 1. S_conf / N_eff
    nz = pi > 0
    s_conf = float(-np.sum(pi[nz] * np.log(pi[nz])))
    n_eff = float(math.exp(s_conf))

    # 2/3. pairwise RMSD, no reflection
    if n_conf > 1:
        rmsd_ab = 0.0
        for a in range(n_conf):
            for b in range(a + 1, n_conf):
                rmsd_ab += 2.0 * pi[a] * pi[b] * kabsch_rmsd(pos[a], pos[b], allow_reflection=False)
    else:
        rmsd_ab = 0.0

    # masses from atomic numbers in x[:, 0]
    z = np.asarray(rec["x"].numpy()[:, 0], dtype=int)
    masses = np.asarray([PT.GetAtomicWeight(int(zz)) for zz in z], dtype=float)

    # 4. Rg ensemble variance
    rg = np.asarray([rg_mass_weighted(pos[c], masses) for c in range(n_conf)])
    rg_mean = weighted_mean(rg, pi)
    sigma2_rg = weighted_var(rg, pi)

    # 5. NPR covariance trace
    npr = np.asarray([npr_mass_weighted(pos[c], masses)[:2] for c in range(n_conf)])
    npr1 = npr[:, 0]
    npr2 = npr[:, 1]
    npr_cov = np.cov(np.vstack([npr1, npr2]), aweights=pi, bias=True)
    tr_sigma_npr = float(np.trace(npr_cov))
    npr_degenerate = bool(np.nanmin(npr1) < 1e-3)

    # 6. polar SASA (SDF with explicit H)
    sasa_vals = np.full(n_conf, np.nan, dtype=float)
    sdf_rmsd = float("nan")
    no_polar = True
    try:
        sdf_mols = load_sdf(name)
        mapping, sdf_rmsd = match_conformers(pos, sdf_mols)
        for c, s in enumerate(mapping):
            sasa_vals[c] = polar_sasa(sdf_mols[s], 0)
        no_polar = not any(a.GetAtomicNum() in (7, 8, 15, 16) for a in sdf_mols[0].GetAtoms())
    except Exception:
        mapping = []

    sasa_mean = float(weighted_mean(sasa_vals, pi)) if np.isfinite(sasa_vals).all() else float("nan")
    sigma_sasa = float(weighted_var(sasa_vals, pi)) if np.isfinite(sasa_vals).all() else float("nan")
    sigma_sasa = float(math.sqrt(max(sigma_sasa, 0.0))) if np.isfinite(sigma_sasa) else float("nan")

    # descriptors
    smiles = str(row.get("canonical_smiles", "") or "")
    desc = rdkit_descriptors(smiles) if smiles else {"mw": float("nan"), "clogp": float("nan"), "n_ha": float("nan"), "tpsa": float("nan")}

    return {
        "name": name,
        "split": split,
        "n_conf": n_conf,
        "S_conf": s_conf,
        "N_eff": n_eff,
        "n_eff_cache": float(rec["n_eff"]),
        "rmsd_ab": float(rmsd_ab),
        "rg_mean": float(rg_mean),
        "sigma2_rg": float(sigma2_rg),
        "npr1_mean": float(weighted_mean(npr1, pi)),
        "npr2_mean": float(weighted_mean(npr2, pi)),
        "tr_sigma_npr": tr_sigma_npr,
        "sasa_polar_mean": sasa_mean,
        "sigma_sasa_polar": sigma_sasa,
        "npr_degenerate": int(npr_degenerate),
        "no_polar_atoms": int(no_polar),
        "sdf_match_rmsd": float(sdf_rmsd),
        "has_censored": int(rec["has_censored"]),
        "run_pass": str(rec["run_pass"]),
        "capped20": int(n_conf >= 20),
        "L_sem": int(rec["y"].sum()),
        "n_rotors": float(row.get("n_rotors", float("nan"))),
        "n_unspecified_stereo": float(row.get("n_unspecified_stereo", float("nan"))),
        **desc,
    }


def kabsch_validation(records: list[dict], run1: pd.DataFrame, n: int = 20) -> dict:
    """Reflect the lowest-energy conformer of chiral molecules and test alignment."""
    candidates = []
    for r in records:
        name = str(r["name"])
        if name not in run1.index:
            continue
        if float(r.get("n_tetra_tag", 0)) > 0 and int(r["pos"].shape[0]) >= 2:
            candidates.append(r)
        if len(candidates) >= n:
            break
    rows = []
    for r in candidates:
        pos = np.asarray(r["pos"].numpy(), dtype=float)
        P = pos[0]
        Q = P.copy()
        Q[:, 2] *= -1.0  # mirror image
        rows.append({
            "name": str(r["name"]),
            "rmsd_no_reflection": kabsch_rmsd(P, Q, allow_reflection=False),
            "rmsd_allow_reflection": kabsch_rmsd(P, Q, allow_reflection=True),
        })
    df = pd.DataFrame(rows)
    return {
        "n": int(len(df)),
        "n_no_refl_gt_0p3": int((df["rmsd_no_reflection"] > 0.3).sum()) if len(df) else 0,
        "n_allow_refl_lt_0p1": int((df["rmsd_allow_reflection"] < 0.1).sum()) if len(df) else 0,
        "max_no_reflection": float(df["rmsd_no_reflection"].max()) if len(df) else None,
        "min_no_reflection": float(df["rmsd_no_reflection"].min()) if len(df) else None,
        "max_allow_reflection": float(df["rmsd_allow_reflection"].max()) if len(df) else None,
        "rows": df.to_dict(orient="records"),
    }


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    records, split_of = load_cache()
    run1 = load_run1_table()
    print(f"loaded {len(records)} molecules, {len(run1)} run1 rows", flush=True)

    rows = []
    missing = []
    for i, rec in enumerate(records):
        name = str(rec["name"])
        if name not in run1.index:
            missing.append({"name": name, "reason": "not in run1_molecules"})
            continue
        try:
            out = compute_one(name, rec, run1.loc[name], split_of[name])
            rows.append(out)
        except Exception as e:  # noqa: BLE001
            missing.append({"name": name, "reason": f"{type(e).__name__}: {e}"})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(records)}  elapsed {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out_csv = CACHE / "ensemble_observables.csv"
    df.to_csv(out_csv, index=False)

    ne = df["N_eff"].to_numpy()
    nec = df["n_eff_cache"].to_numpy()
    rel = np.abs(ne - nec) / np.maximum(np.abs(nec), 1e-12)
    val = kabsch_validation(records, run1, n=20)
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_molecules": int(len(df)),
        "n_conformers": int(df["n_conf"].sum()),
        "n_missing": int(len(missing)),
        "missing": missing,
        "missing_rate": float(len(missing) / max(len(records), 1)),
        "max_n_eff_rel_err": float(rel.max()) if len(rel) else None,
        "n_eff_rel_err_gt_1e-4": int((rel > 1e-4).sum()) if len(rel) else 0,
        "n_eff_rel_err_gt_1e-5": int((rel > 1e-5).sum()) if len(rel) else 0,
        "kabsch_validation": {k: v for k, v in val.items() if k != "rows"},
        "columns": list(df.columns),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (CACHE / "observables_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("missing", "columns")}, indent=2, ensure_ascii=False))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
