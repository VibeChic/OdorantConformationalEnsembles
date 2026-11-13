"""WP3: MMP cliffs and ensemble shifts.

For every MMP pair with both molecules in the 3D observables:
  |Delta S_conf|, N_eff ratio, W2(NPR_a, NPR_b)
Primary inference is test-only (116 cliffs / 976 controls); all-split results
are descriptive.

Outputs under paper3/results/wp3_cliffs/.
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
from scipy.optimize import linprog
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import CACHE, DATA, RESULTS, load_cache  # noqa: E402
from p3.observables import npr_mass_weighted  # noqa: E402
from p3.common import PT  # noqa: E402

OUT = RESULTS / "wp3_cliffs"
OUT.mkdir(parents=True, exist_ok=True)


def canon(s: str) -> str:
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else ""


def build_name_map() -> dict[str, str]:
    r1 = pd.read_csv(DATA / "3d" / "run1" / "run1_molecules.csv", low_memory=False)
    r1 = r1[r1.status == "ok"]
    mp = {}
    for name, smi in zip(r1["name"].astype(str), r1["canonical_smiles"].astype(str)):
        k = canon(smi)
        if k and k not in mp:
            mp[k] = name
    return mp


def npr_points(pos: np.ndarray, z: np.ndarray, pi: np.ndarray) -> np.ndarray:
    masses = np.asarray([PT.GetAtomicWeight(int(zz)) for zz in z], dtype=float)
    return np.asarray([npr_mass_weighted(pos[c], masses)[:2] for c in range(pos.shape[0])])


def sinkhorn_w2(x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 0.01, n_iter: int = 200) -> float:
    C = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    K = np.exp(-C / eps)
    u = np.ones(len(a)) / len(a)
    v = np.ones(len(b)) / len(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    P = u[:, None] * K * v[None, :]
    return float(math.sqrt(max(np.sum(P * C), 0.0)))


def exact_w2(x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    C = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    n, m = C.shape
    c = C.reshape(-1)
    A = []
    rhs = []
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1
        A.append(row); rhs.append(a[i])
    for j in range(m):
        row = np.zeros(n * m)
        row[j::m] = 1
        A.append(row); rhs.append(b[j])
    res = linprog(c, A_eq=np.asarray(A), b_eq=np.asarray(rhs), bounds=(0, None), method="highs")
    if not res.success:
        return float("nan")
    return float(math.sqrt(max(res.fun, 0.0)))


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 1000, seed: int = 20261025):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, score)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        ii = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(roc_auc_score(y[ii], score[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(base), float(lo), float(hi), len(vals)


def main(n_boot: int = 1000) -> None:
    obs = pd.read_csv(CACHE / "ensemble_observables.csv").set_index("name")
    pairs = pd.read_csv(DATA / "mmp" / "mmp_pairs.csv")
    mp = build_name_map()
    pairs["name_a"] = pairs["smiles_a"].map(lambda s: mp.get(canon(s)))
    pairs["name_b"] = pairs["smiles_b"].map(lambda s: mp.get(canon(s)))
    ok = pairs.dropna(subset=["name_a", "name_b"]).copy()
    print(f"pairs {len(pairs)}, mapped {len(ok)} ({len(ok)/len(pairs):.1%})", flush=True)

    records, _ = load_cache()
    rec = {str(r["name"]): r for r in records}
    npr_cache = {}

    def get_npr(name):
        if name not in npr_cache:
            r = rec[name]
            npr_cache[name] = (npr_points(np.asarray(r["pos"].numpy(), float), np.asarray(r["x"].numpy()[:, 0], int), np.asarray(r["pi"].numpy(), float)),
                               np.asarray(r["pi"].numpy(), float))
        return npr_cache[name]

    rows = []
    t0 = time.time()
    for i, r in enumerate(ok.itertuples(index=False)):
        a, b = r.name_a, r.name_b
        if a not in obs.index or b not in obs.index:
            continue
        sa, sb = float(obs.loc[a, "S_conf"]), float(obs.loc[b, "S_conf"])
        na, nb = float(obs.loc[a, "N_eff"]), float(obs.loc[b, "N_eff"])
        xa, pa = get_npr(a)
        xb, pb = get_npr(b)
        w2 = sinkhorn_w2(xa, xb, pa, pb)
        rows.append({
            "split": r.split,
            "is_cliff": int(r.is_cliff),
            "tier": r.tier,
            "name_a": a, "name_b": b,
            "tanimoto": float(r.struct_tanimoto),
            "hac_delta": abs(float(r.hac_a) - float(r.hac_b)),
            "dS": abs(sa - sb),
            "dN_eff": abs(na - nb),
            "N_eff_ratio": max(na, nb) / max(min(na, nb), 1e-9),
            "w2_npr": w2,
        })
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(ok)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wp3_pair_metrics.csv", index=False)

    test = df[df.split == "test"].copy()
    y = test["is_cliff"].to_numpy(int)
    auc_rows = []
    for metric in ["dS", "w2_npr", "dN_eff", "N_eff_ratio"]:
        score = test[metric].to_numpy(float)
        # orient so higher = more cliff-like
        if metric in ("dN_eff", "N_eff_ratio"):
            score = -score
        auc, lo, hi, nb = auc_ci(y, score, n_boot=n_boot)
        auc_rows.append({"metric": metric, "auc": auc, "ci_lo": lo, "ci_hi": hi, "n_boot_valid": nb})
    pd.DataFrame(auc_rows).to_csv(OUT / "wp3_auc.csv", index=False)

    g = test.groupby("is_cliff")[["dS", "w2_npr", "dN_eff", "N_eff_ratio", "tanimoto"]].agg(["count", "mean", "median", "std"])
    g.to_csv(OUT / "wp3_group_stats.csv")

    # exact-LP cross-check on 100 random pairs
    rng = np.random.default_rng(20261025)
    idx = rng.choice(len(df), size=min(100, len(df)), replace=False)
    diffs = []
    for i in idx:
        r = df.iloc[i]
        xa, pa = get_npr(r.name_a); xb, pb = get_npr(r.name_b)
        e = exact_w2(xa, xb, pa, pb)
        s = float(r.w2_npr)
        diffs.append(abs(e - s) / max(e, 1e-9))
    diffs = np.asarray(diffs)
    summary = {
        "n_pairs_total": int(len(pairs)),
        "n_pairs_mapped": int(len(df)),
        "mapping_rate": float(len(df) / len(pairs)),
        "n_test": int(len(test)),
        "n_test_cliffs": int(y.sum()),
        "auc_test": auc_rows,
        "exact_lp_check": {
            "n": int(len(diffs)),
            "max_rel_diff": float(np.nanmax(diffs)) if len(diffs) else None,
            "median_rel_diff": float(np.nanmedian(diffs)) if len(diffs) else None,
        },
        "note": "AUC oriented so higher score = more cliff-like (dN_eff and N_eff_ratio are negated).",
    }
    (OUT / "wp3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    main(n_boot=args.n_boot)
