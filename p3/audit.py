"""Paper 3 audit: PRESENT / ABSENT / CONSISTENCY checks.

Exit code 0 only if all required files exist and the core consistency checks pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.common import CACHE, REPORTS, RESULTS  # noqa: E402

REQUIRED = [
    CACHE / "ensemble_observables.csv",
    CACHE / "observables_manifest.json",
    RESULTS / "wp1_semantic" / "wp1_summary.json",
    RESULTS / "wp2_m2or" / "wp2_summary.json",
    RESULTS / "wp3_cliffs" / "wp3_summary.json",
    RESULTS / "wp4_dream" / "wp4_summary.json",
    RESULTS / "wp5_bioactive" / "wp5_summary.json",
]


def public_mode(data_dir: Path) -> int:
    """Verify the released QH-Odor-1821 bundle without the project working tree."""
    import numpy as np

    cohort = data_dir / "cohorts" / "gs_lf_4809_master_observables.csv"
    c95 = data_dir / "probes" / "saturation_c95_1821.csv"
    lines = ["# Paper 3 audit (public bundle)", ""]
    if not cohort.exists():
        print(f"missing {cohort}")
        return 2

    d = pd.read_csv(cohort)
    checks = {
        "n_molecules == 4809": len(d) == 4809,
        "sum(Phase I conformers) == 55895": int(d.n_conf.sum()) == 55895,
        "split 3363/716/730": d.split.value_counts().to_dict() == {"train": 3363, "test": 730, "val": 716},
        "censored == 601": int(d.has_censored.sum()) == 601,
        "maxcap 2987 x 20 / 1822 x 200": d.maxcap.value_counts().to_dict() == {20: 2987, 200: 1822},
        "complete cohort 1821 molecules / 106526 conformers":
            (int(d.n_conf_complete.notna().sum()) == 1821 and int(d.n_conf_complete.sum()) == 106526),
        "N_eff == exp(S_conf)": float((d.N_eff - np.exp(d.S_conf)).abs().div(d.N_eff).max()) < 1e-9,
        "sigma(polar SASA) complete": bool(d.sigma_sasa_polar.notna().all()),
    }
    if c95.exists():
        c = pd.read_csv(c95)
        checks["median C95 == 28 (n=1821)"] = (len(c) == 1821 and float(c.C95.median()) == 28.0)
    ok = all(checks.values())
    for k, v in checks.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: {k}")
    lines += ["", "Skipped in public mode: project-tree presence checks "
                  "(cache/, results/wp*) and manuscript text comparison.", ""]
    lines.append(f"**Overall**: {'PASS' if ok else 'FAIL'}")
    print("\n".join(lines))
    (Path.cwd() / "audit_p3_public.md").write_text("\n".join(lines) + "\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="QH-Odor-1821 bundle (public mode)")
    args = ap.parse_args()
    if args.data_dir:
        return public_mode(Path(args.data_dir))

    lines = ["# Paper 3 audit", ""]
    ok = True
    lines.append("## PRESENT")
    for p in REQUIRED:
        exists = p.exists()
        lines.append(f"- {'OK' if exists else 'MISSING'}: `{p.relative_to(p.parents[2])}`")
        ok = ok and exists

    lines.append("")
    lines.append("## CONSISTENCY")
    if (CACHE / "ensemble_observables.csv").exists():
        df = pd.read_csv(CACHE / "ensemble_observables.csv")
        checks = {
            "n_molecules == 4809": len(df) == 4809,
            "n_conformers == 55895": int(df.n_conf.sum()) == 55895,
            "no missing S_conf": df.S_conf.notna().all(),
            "no missing SASA": df.sigma_sasa_polar.notna().all(),
            "max N_eff rel err < 1e-4": float((df.N_eff - df.n_eff_cache).abs().div(df.n_eff_cache).max()) < 1e-4,
        }
        for k, v in checks.items():
            lines.append(f"- {'PASS' if v else 'FAIL'}: {k}")
            ok = ok and bool(v)
    else:
        ok = False

    if (RESULTS / "wp1_semantic" / "wp1_summary.json").exists():
        s = json.loads((RESULTS / "wp1_semantic" / "wp1_summary.json").read_text())
        lines.append(f"- INFO: primary beta2={s['beta2']['beta']:.4f}, CI=[{s['beta2']['ci_lo']:.4f},{s['beta2']['ci_hi']:.4f}], decision={s['decision']}")
    if (RESULTS / "wp3_cliffs" / "wp3_summary.json").exists():
        s = json.loads((RESULTS / "wp3_cliffs" / "wp3_summary.json").read_text())
        lines.append(f"- INFO: WP3 test n={s['n_test']} (cliffs {s['n_test_cliffs']}); AUC dS={s['auc_test'][0]['auc']:.3f}")
    if (RESULTS / "wp4_dream" / "wp4_summary.json").exists():
        s = json.loads((RESULTS / "wp4_dream" / "wp4_summary.json").read_text())
        lines.append(f"- INFO: WP4 n_with_3d={s['n_with_3d']}; dilution dR2={s['dilution_1_1000_intensity']['delta_r2']:.3f}")

    lines.append("")
    lines.append(f"**Overall**: {'PASS' if ok else 'FAIL'}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
