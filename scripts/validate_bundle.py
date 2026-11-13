#!/usr/bin/env python3
"""Pre-publish gate: fail on any column with nulls, out-of-domain values or
count drift. This is the check that would have caught the `maxcap` defect
(100% null) before the first data release.

    python scripts/validate_bundle.py --bundle dist/zenodo_data_QH-Odor-1821
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DOMAIN = {
    "split": {"train", "val", "test"},
    "maxcap": {20, 200},
    "has_censored": {0, 1},
    "capped20": {0, 1},
}
REQUIRED = ["molecule_id", "smiles", "split", "L_sem", "mw", "clogp", "hba", "tpsa",
            "has_censored", "maxcap", "n_conf", "S_conf", "N_eff", "sigma_sasa_polar"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()
    b = Path(args.bundle)
    fails = []

    cohort = pd.read_csv(b / "cohorts/gs_lf_4809_master_observables.csv")
    print(f"cohort: {len(cohort)} x {len(cohort.columns)}")

    # 1. no nulls in required columns
    for c in REQUIRED:
        if c not in cohort:
            fails.append(f"missing column {c}")
            continue
        n = int(cohort[c].isna().sum())
        if n:
            fails.append(f"{c}: {n} nulls ({n/len(cohort):.1%})")

    # 2. domains
    for c, allowed in DOMAIN.items():
        if c in cohort:
            bad = set(cohort[c].dropna().unique()) - allowed
            if bad:
                fails.append(f"{c}: unexpected values {sorted(bad)[:5]}")

    # 3. counts
    checks = {
        "rows==4809": len(cohort) == 4809,
        "sum(n_conf)==55895": int(cohort.n_conf.sum()) == 55895,
        "split 3363/716/730": cohort.split.value_counts().to_dict() == {"train": 3363, "test": 730, "val": 716},
        "censored==601": int(cohort.has_censored.sum()) == 601,
        "maxcap 2987/1822": cohort.maxcap.value_counts().to_dict() == {20: 2987, 200: 1822},
        "complete 1821/106526": (int(cohort.n_conf_complete.notna().sum()) == 1821
                                 and int(cohort.n_conf_complete.sum()) == 106526),
        "L_sem 1..21": bool(cohort.L_sem.between(1, 21).all()),
    }
    for k, v in checks.items():
        if not v:
            fails.append(k)

    # 4. companion tables
    c95 = pd.read_csv(b / "probes/saturation_c95_1821.csv")
    if len(c95) != 1821 or float(c95.C95.median()) != 28.0:
        fails.append("saturation_c95: expected 1821 rows and median 28")
    m = json.loads((b / "metrics/paper2_21arms_10seeds_metrics.json").read_text())["arms"]
    if len(m) != 21 or any(len(v) != 10 for v in m.values()):
        fails.append("metrics: expected 21 arms x 10 seeds")

    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: no nulls in required columns, domains valid, counts match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
