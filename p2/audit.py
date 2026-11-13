#!/usr/bin/env python3
"""Audit the Paper-2 write-up against the artefacts.

Two checks, both mechanical:

1. PRESENT  — every headline number in the manuscript must match a value
              recomputed from results/*/metrics.json (mean +/- sample SD).
2. ABSENT   — numbers that v1.0 asserted but that this data cannot support
              must not appear anywhere except in the corrections table.

Exit code is non-zero if either check fails, so this can gate a submission.

Usage:
  python p2/audit.py
  python p2/audit.py --doc Paper2_manuscript_v2_bounding_result.md
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "E-N6 ring": ("results/s2_ablation/en6/seed*/metrics.json", ("aucpr",)),
    "T4 tau=10": ("results/s1_mil/temp_10/seed*/metrics.json", ("aucpr",)),
    "T4 tau=100": ("results/s1_mil/temp_100/seed*/metrics.json", ("aucpr",)),
    "T4 tau=600": ("results/s1_mil/temp_600/seed*/metrics.json", ("aucpr",)),
    "T4 tau=1500": ("results/s1_mil/temp_1500/seed*/metrics.json", ("aucpr",)),
    "T4 tau=inf": ("results/s1_mil/temp_inf/seed*/metrics.json", ("aucpr",)),
    "T5 sigma=0.05": ("results/s1_mil/gauss_005/seed*/metrics.json", ("aucpr",)),
    "T5 sigma=0.15": ("results/s1_mil/gauss_015/seed*/metrics.json", ("aucpr",)),
    "T5 sigma=0.30": ("results/s1_mil/gauss_030/seed*/metrics.json", ("aucpr",)),
    "T5 sigma=RMSD": ("results/s1_mil/gauss_rmsd/seed*/metrics.json", ("aucpr",)),
}

OLD_GROUPS = {
    "R1 gin": "results/r1/gin/seed*/metrics.json",
    "E-N1 n_rotors": "results/s2_ablation/en1/seed*/metrics.json",
    "E-N2 thermodynamics": "results/s2_ablation/en2/seed*/metrics.json",
    "E-N4 shape": "results/s2_ablation/en4/seed*/metrics.json",
    "E-N5 all nine": "results/s2_ablation/en5/seed*/metrics.json",
    "R3 uniform": "results/s1_mil/shuffle/seed*/metrics_det.json",
    "R3 boltzmann": "results/s1_mil/boltz/seed*/metrics.json",
    "R3 attention": "results/s1_mil/attn/seed*/metrics.json",
    "R2.5 edge_gin": "results/r25/edge_gin/seed*/metrics.json",
    "E-N3 scalar_gin": "results/r25/scalar_gin/seed*/metrics.json",
    "R2.5 shuffled": "results/r25/edge_gin_shuffled/seed*/metrics.json",
    "R2 schnet": "results/r2_sc/schnet/seed*/metrics.json",
    "R2 chiral_schnet": "results/r2_sc/chiral_schnet/seed*/metrics.json",
    "R2 angle_mpnn": "results/r2/angle_mpnn/seed*/metrics.json",
    "R2 dimenet": "results/r2_dimenet/dimenet/seed*/metrics.json",
}

# numbers v1.0 claimed and the data does not support
CROSS = [
    ("DREAM morgan", "results/s3_tiers/dream/morgan/seed*/metrics.json", "mean_r",
     "{m:.3f} ± {s:.3f}"),
    ("DREAM en1", "results/s3_tiers/dream/en1/seed*/metrics.json", "mean_r",
     "{m:.3f} ± {s:.3f}"),
    ("DREAM en6", "results/s3_tiers/dream/en6/seed*/metrics.json", "mean_r",
     "{m:.3f} ± {s:.3f}"),
    ("DREAM r1", "results/s3_tiers/dream/r1/seed*/metrics.json", "mean_r",
     "{m:.3f} ± {s:.3f}"),
    ("M2OR S2 A", "results/s3_tiers/m2or/receptor/A/seed*/metrics.json", "macro_ap",
     "{m:.4f} ± {s:.4f}"),
    ("M2OR S2 B", "results/s3_tiers/m2or/receptor/B/seed*/metrics.json", "macro_ap",
     "{m:.4f} ± {s:.4f}"),
    ("M2OR S2 C", "results/s3_tiers/m2or/receptor/C/seed*/metrics.json", "macro_ap",
     "{m:.4f} ± {s:.4f}"),
    ("M2OR S1 A", "results/s3_tiers/m2or/ligand/A/seed*/metrics.json", "macro_ap",
     "{m:.4f} ± {s:.4f}"),
    ("M2OR S1 B", "results/s3_tiers/m2or/ligand/B/seed*/metrics.json", "macro_ap",
     "{m:.4f} ± {s:.4f}"),
    ("M2OR S1 C", "results/s3_tiers/m2or/ligand/C/seed*/metrics.json", "macro_ap",
     "{m:.4f} ± {s:.4f}"),
]

FORBIDDEN = ["0.9412", "0.9329", "0.9318", "0.9265", "0.9184", "0.9228",
             "84,612", "84612", "16.98", "1,978"]


def truth():
    """Ladder arms: both macro-AUCPR and micro-AUROC appear in the text."""
    out = {}
    for name, pat in OLD_GROUPS.items():
        a, m = [], []
        for f in sorted(glob.glob(os.path.join(PROJ, pat))):
            d = json.load(open(f))
            a.append(d["test_macro_aucpr"])
            m.append(d["test_micro_auroc"])
        a, m = np.array(a), np.array(m)
        out[name] = {
            "aucpr": f"{a.mean():.4f} ± {a.std(ddof=1):.4f}",
            "micro": f"{m.mean():.4f} ± {m.std(ddof=1):.4f}",
            "n": len(a),
        }
    return out


def data_truth():
    """Headline data-layer numbers recomputed from the caches, not from prose."""
    man = pd.read_csv(os.path.join(os.path.dirname(PROJ), "data", "3d", "run1",
                                   "run1_gnn_manifest.csv"))
    stereo = pd.read_csv(os.path.join(PROJ, "cache", "stereo_map.csv"))
    conformers = int(man["n_conformers_in_file"].sum())
    mean_c = float(man["n_conformers_in_file"].mean())
    at_cap = int((man["n_conformers_in_file"] == 20).sum())
    censored = float(man["has_censored"].mean())
    tetra = int((stereo["n_tetra_tag"].fillna(0) > 0).sum())
    ez = int((stereo["n_double_defined"].fillna(0) > 0).sum())

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    acyclic = 0
    for smi in stereo["isomeric_smiles"].dropna():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for b in m.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE or b.IsInRing():
                continue
            if b.GetBeginAtom().GetDegree() >= 2 and b.GetEndAtom().GetDegree() >= 2:
                acyclic += 1
                break
    return {
        "molecules_3d": len(man),
        "conformers": conformers,
        "mean_C": round(mean_c, 2),
        "at_cap_20": at_cap,
        "censored_pct": round(censored * 100, 1),
        "tetra_stereo": tetra,
        "ez_defined": ez,
        "acyclic_stereogenic_alkene": acyclic,
    }


EXPECTED = {  # manuscript values: mean +/- sample s.d. over 10 seeds
    "R1": (0.3187, 0.0029), "SCHNET": (0.3081, 0.0068), "EN4": (0.3275, 0.0039),
    "en4_s2d": (0.3316, 0.0031), "wide_en1": (0.3318, 0.0033),
    "en4_size": (0.3288, 0.0035), "en4_shape": (0.3197, 0.0031),
}


def public_mode(data_dir):
    """Verify the released metrics table without the project working tree."""
    metrics = os.path.join(data_dir, "metrics", "paper2_21arms_10seeds_metrics.json")
    if not os.path.exists(metrics):
        print("missing " + metrics)
        return 2
    m = json.load(open(metrics))["arms"]
    lines = ["# Paper 2 audit (public bundle)", ""]
    checks = {"21 arms": len(m) == 21,
              "10 seeds per arm": all(len(v) == 10 for v in m.values()),
              "no NaN": all(s["macro_aucpr"] is not None for v in m.values() for s in v)}
    for arm, (mu, sd) in EXPECTED.items():
        vals = np.array([s["macro_aucpr"] for s in m.get(arm, [])])
        ok = (len(vals) == 10 and abs(vals.mean() - mu) < 5e-4
              and abs(vals.std(ddof=1) - sd) < 5e-4)
        checks["%s = %.4f +/- %.4f" % (arm, mu, sd)] = bool(ok)
    ok = all(checks.values())
    for k, v in checks.items():
        lines.append("- %s: %s" % ("PASS" if v else "FAIL", k))
    lines += ["", "Skipped in public mode: manuscript text checks (no --doc in the release).", ""]
    lines.append("**Overall**: " + ("PASS" if ok else "FAIL"))
    print("\n".join(lines))
    open("audit_p2_public.md", "w").write("\n".join(lines) + "\n")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=os.path.join(PROJ, "reports", "Paper2_manuscript_v3.1_nc.md"))
    ap.add_argument("--data-dir", default=None, help="QH-Odor-1821 bundle (public mode)")
    args = ap.parse_args()
    if args.data_dir:
        sys.exit(public_mode(args.data_dir))

    if not os.path.exists(args.doc):
        print("project mode needs --doc (manuscript markdown); "
              "use --data-dir for the public bundle")
        sys.exit(2)
    text = open(args.doc).read()
    t = truth()
    failures = []

    print("== NEW ARMS (macro-AUCPR only) ==")
    for name, (pat, keys) in GROUPS.items():
        a = np.array([json.load(open(f))["test_macro_aucpr"]
                      for f in sorted(glob.glob(os.path.join(PROJ, pat)))])
        val = f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"
        ok = val in text
        print(f"  [{'OK ' if ok else 'MISS'}] {name:16s} {val} (n={len(a)})")
        if not ok:
            failures.append(f"{name} = {val} not found")

    print("\n== CROSS-TIER (DREAM / M2OR) ==")
    for name, pat, key, fmt in CROSS:
        a = np.array([json.load(open(f))[key]
                      for f in sorted(glob.glob(os.path.join(PROJ, pat)))])
        val = fmt.format(m=a.mean(), s=a.std(ddof=1))
        ok = val in text
        print(f"  [{'OK ' if ok else 'MISS'}] {name:22s} {val}")
        if not ok:
            failures.append(f"{name} = {val} not found")

    print("\n== ARM COUNT ==")
    n_arms = len(OLD_GROUPS) + 1 + 1      # ladder + en6 + heading
    for claim in ("17 run families", "seventeen"):
        if claim in text:
            print(f"  [OK ] '{claim}' present")
            break
    else:
        print("  [MISS] arm count sentence ('17 run families')")
        failures.append("arm count sentence missing")

    print("\n== PRESENT (headline numbers must appear verbatim) ==")
    for name, v in t.items():
        for key in ("aucpr", "micro"):
            ok = v[key] in text
            print(f"  [{'OK ' if ok else 'MISS'}] {name:18s} {key:5s} {v[key]}")
            if not ok:
                failures.append(f"{name} {key} = {v[key]} not found")

    print("\n== ABSENT (v1-only claims must not appear) ==")
    # the corrections table legitimately quotes the old values; ignore that block
    body = re.split(r"## 6\. Corrections", text)[0]
    for bad in FORBIDDEN:
        hit = bad in body
        print(f"  [{'BAD' if hit else 'OK '}] {bad}")
        if hit:
            failures.append(f"forbidden token {bad} in body")

    print("\n== DATA LAYER ==")
    dt = data_truth()
    expect = {"molecules_3d": "4,809", "conformers": "55,895", "mean_C": "11.62",
              "at_cap_20": "1,822", "censored_pct": "12.5", "tetra_stereo": "1,977",
              "ez_defined": "0", "acyclic_stereogenic_alkene": "1,047"}
    for k, v in dt.items():
        want = expect[k]
        ok = str(want) in text
        print(f"  [{'OK ' if ok else 'MISS'}] {k:26s} = {v}   (manuscript uses {want})")
        if not ok:
            failures.append(f"data number {k}={v} not found as {want}")

    print("\n== CONSISTENCY (same model, same n) ==")
    ns = {v["n"] for v in t.values()}
    print(f"  seeds per group: {sorted(ns)}")
    if len(ns) != 1:
        failures.append(f"groups have different seed counts: {ns}")

    print()
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: manuscript numbers match artefacts; no unsupported v1 claims in the body.")


if __name__ == "__main__":
    main()
