#!/usr/bin/env python3
"""Write README_DATA.md, datapackage.json, SHA256SUMS.txt and manifest_sha256.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "dist" / "zenodo_data_QH-Odor-1821"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


README = """# QH-Odor-1821 — data release

Complete 300 K GFN2-xTB conformational ensembles and derived observables for
**“Odorant conformational ensembles encode topology, molecular size and
polar-surface coherence, not shape”**.

Licence: **CC BY 4.0** (see the manuscript and the Zenodo record). Code: MIT,
separate repository.

## Contents

| Path | Rows | What it is |
| :--- | ---: | :--- |
| `cohorts/gs_lf_4809_master_observables.csv` | 4,809 | One row per converged odorant: split, `L_sem`, covariates, rotor counts, observables at both ceilings, InChIKey and Pyrfume CID |
| `cohorts/phase1_cache_4809_observables.csv.gz` | 4,809 | Phase I (C_max = 20) scalar cache as used by the 2D/3D ladder |
| `cohorts/dream_338_psychophysics_observables.csv` | 338 | DREAM ratings, 21 attributes (304 of these have 3D ensembles) |
| `cohorts/m2or_ligand_panel_294.csv` | 294 | M2OR ligands with 3D observables and receptor-breadth counts |
| `ensembles/phase2_complete_1821_106526confs.sdf.gz` | 106,526 | All Phase II conformers (1,821 molecules, C_max = 200), with per-conformer properties |
| `ensembles/phase2_thermochemistry_106526.csv.gz` | 106,526 | One row per conformer: `g_abs_kcal`, `dg_kcal`, populations, `n_imag`, entropies |
| `probes/alpb_solvation_300mols_16836confs.csv.gz` | 16,836 | ALPB single points in water / 1-octanol / n-hexane + chameleonicity |
| `probes/usrcat_60d_pharmacophore_entropy_1821.csv.gz` | 1,821 | USRCAT pharmacophore entropies, five estimators |
| `probes/saturation_curves_C95_1821.csv.gz` | 97,144 | S(C)/S(C_total) trajectories per molecule and cap |
| `probes/saturation_c95_1821.csv` | 1,821 | Per-molecule C_95 and n_conf |
| `probes/perceptual_cliffs_pairs.csv` | 15,235 | All candidate MMP pairs with Tanimoto, ΔL_sem, `is_cliff`, `tier` |
| `metrics/paper2_21arms_10seeds_metrics.json` | 21 × 10 | Per-seed macro-AUCPR / AUROC, best epoch, parameter count |
| `Figures/figure_source_data.json`, `Figures/source_data_paper2_arms.json` | — | Every value plotted in the manuscript |

## Units and definitions

* `S_conf` — nats; `S = −Σ p_i ln p_i` over deduplicated minima.
* `N_eff` — effective state count, `exp(S_conf)`.
* `rmsd_ab` — Å; mean pairwise heavy-atom RMSD after Kabsch superposition.
* `sigma_sasa_polar` — Å²; Boltzmann-weighted s.d. of the polar SASA.
* `g_abs_kcal`, `dg_kcal`, `zpe_kcal`, `ts_kcal` — kcal mol⁻¹, 300 K, quasi-RRHO.
* `population` — Boltzmann weight at 300 K within the molecule's conformer set.
* `n_rotors_pipeline` — permissive count used for conformer generation (median 6).
* `n_rotors_rdkit` — strict RDKit `CalcNumRotatableBonds` (median 3). The 9.2-bond
  optimum refers to the permissive count; 6.1 bonds to the strict one.
* `L_sem` — integer row sum of the 138 binary catalogue descriptors (1–21).
  **The binary matrix itself is not redistributed.**

## What is deliberately absent

1. **The 138-descriptor binary matrix** and any commercial catalogue text. Only
   the row sum `L_sem` is released.
2. **Phase I geometries (55,895 conformers).** They were produced with a lighter
   embedding budget and were not retained as an SDF; all Phase I numbers are
   reproducible from `cohorts/phase1_cache_4809_observables.csv.gz` and the
   Phase II ensembles.
3. **The 488-ligand M2OR matrix.** The release carries the 294 ligands that also
   have 3D observables, which are the rows used in every receptor-tier model.
4. **The 527-pair cliff subset.** The release carries all 15,235 candidate pairs
   with `tier` and `is_cliff`; the analysis subset used in Fig. 5d is obtained by
   the filter in `p3/wp3_cliffs.py`.

## Coverage notes

* `pyrfume_id` resolves for 4,430 of 4,809 molecules (92.2%) by exact InChIKey
  match against the Pyrfume registry; the remaining 379 keep their InChIKey and
  can be joined by structure.
* `m2or_ligand_panel_294.csv` is the 3D-overlap subset used in every receptor
  model; the wider 488-ligand assay matrix is not redistributed.

## Reproducing the numbers

```bash
python scripts/download_zenodo_data.py --record <id> --out data
python p2/audit.py
python p3/audit.py
```

The audits recompute every table and figure source value from these files and
exit non-zero on any disagreement.

## Provenance

* 4,809 converged Phase I ensembles (55,895 conformers) and 1,821 complete
  Phase II ensembles (106,526 conformers) at 300 K, GFN2-xTB 6.7.1 `--ohess`.
* Frozen analysis plan: `SHA-256 667044a3e8b0b158ed247c1faa4c86babe99f45cac901c90252927e9737b0d4f`.
* Built from the project working trees on 2026-11-13; every file hashed in
  `SHA256SUMS.txt` and `manifest_sha256.json`.
"""


def main() -> int:
    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file() and p.name not in
                   {"README_DATA.md", "datapackage.json", "SHA256SUMS.txt", "manifest_sha256.json"})
    manifest = {}
    lines = []
    for p in files:
        h = sha(p)
        rel = str(p.relative_to(BUNDLE))
        manifest[rel] = {"sha256": h, "bytes": p.stat().st_size}
        lines.append(f"{h}  {rel}")
    (BUNDLE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    (BUNDLE / "manifest_sha256.json").write_text(json.dumps(manifest, indent=1))
    (BUNDLE / "README_DATA.md").write_text(README)

    resources = [{"name": Path(k).stem, "path": k, "bytes": v["bytes"],
                  "hash": f"sha256:{v['sha256']}",
                  "mediatype": "chemical/x-mdl-sdfile" if k.endswith(".sdf.gz") else
                               "text/csv" if k.endswith(".csv") else
                               "application/gzip" if k.endswith(".gz") else "application/json"}
                 for k, v in manifest.items()]
    dp = {"name": "qh-odor-1821", "title": "QH-Odor-1821", "profile": "data-package",
          "licenses": [{"name": "CC-BY-4.0", "path": "https://creativecommons.org/licenses/by/4.0/",
                        "title": "Creative Commons Attribution 4.0"}],
          "resources": resources}
    (BUNDLE / "datapackage.json").write_text(json.dumps(dp, indent=1))
    print(f"{len(files)} files, {sum(v['bytes'] for v in manifest.values())/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
