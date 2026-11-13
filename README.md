# OdorantConformationalEnsembles

Code for the manuscript **“Odorant conformational ensembles encode topology,
molecular size and polar-surface coherence, not shape”**.

The study has two computational halves that share one cohort and one locked
numerical register:

* a **21-arm neural representation ladder** (10 seeds per arm) asking whether
  explicit 300 K ensembles beat 2D topology on olfactory benchmarks;
* a **pre-registered negative-binomial analysis of odour breadth** on complete
  300 K GFN2-xTB ensembles (1,821 flexible odorants, 106,526 conformers),
  including the three-axis confound dissection.

## Licensing

* **Code** (everything in `p2/`, `p3/`, `en4_decomp/`, `scripts/`, `container/`,
  `make_figures.py`, `make_figures_p3.py`): **MIT** — see `LICENSE`.
* **Data** (the QH-Odor-1821 bundle and any derived tables / figure source
  JSON): **CC BY 4.0** — see `LICENSE-DATA`.

GitHub's licence detector reads only the top-level `LICENSE` (MIT). The data
licence is stated here and in `LICENSE-DATA`; please do not treat the code
licence as covering the data.

## Repository layout

```
p2/                 21-arm ladder, DREAM and M2OR models, audit.py
p3/                 conformer observables, NB2 curvature models, audit.py
en4_decomp/         size/shape decomposition and control arms
conformer_pipeline/ pointer to the generation entry points (see its README)
preregistration/    frozen analysis plan (SHA-256 667044a3…b0d4f) + deviation log
container/          Dockerfile / Apptainer definition for landscape-md:1.1
scripts/            download, leakage scan, Zenodo upload helpers
make_figures.py     Paper 2 panels (ladder, truncation, tiers)
make_figures_p3.py  Paper 3 panels (cohort, saturation, curvature, probes)
```

## Quick start

```bash
# 1. container (pinned environment; ~8 min)
docker build -t landscape-md:1.1 container/

# 2. data (QH-Odor-1821, CC BY 4.0)
export ZENODO_RECORD=<record-id>          # printed on the Zenodo record page
python scripts/download_zenodo_data.py --record "$ZENODO_RECORD" --out data

# 3. audits
python p2/audit.py
python p3/audit.py

# 4. figures (source data come from the data bundle, not from this repository)
python make_figures.py out_figures
python make_figures_p3.py out_figures
```

### Reproduction status

The code in this repository is the **exact version that produced the
manuscript**. The audits were executed against the project working layout
(`cache/`, `results/`, `v3_results/`), which is reconstructed from the Zenodo
bundle. Two adapters — `DATA_DIR` for `p3/` and `--metrics-root` for `p2/` —
are being finalised so that both audits can run from a clean checkout using
only the public bundle; until they land, `p2/audit.py` and `p3/audit.py`
require the working layout described in `preregistration/` and in the Zenodo
`README_DATA.md`.

## What is deliberately not here

* the proprietary 138-descriptor commercial catalogue matrix;
* raw commercial catalogue annotations;
* conformer geometries and thermochemistry (distributed via Zenodo, CC BY 4.0);
* figure images — only figure *source data* are distributed, via the data
  bundle.

## Pre-registration

`preregistration/PREREGISTERED_ANALYSIS_PLAN.md` is byte-identical to the
frozen plan; verify with

```bash
cd preregistration && sha256sum -c PREREGISTERED_ANALYSIS_PLAN.sha256
# 667044a3e8b0b158ed247c1faa4c86babe99f45cac901c90252927e9737b0d4f
```

The single logged deviation (Stage 2 completed all 1,821 capped molecules after
Stage 1 returned a 95% bootstrap half-width of 0.099 > 0.05) is in
`preregistration/DEVIATION_LOG.md`.

## Citation

See `CITATION.cff`. The archived release is deposited on Zenodo
(`10.5281/zenodo.[CODE_DOI]`, to be minted at publication) and links to the
data record (`10.5281/zenodo.[DATA_DOI]`).

## Contact

Corresponding author (see the manuscript). Please open an issue for
reproducibility problems rather than emailing, so that fixes are visible.
