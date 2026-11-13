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
export ZENODO_RECORD=24187214             # QH-Odor-1821 (concept record id)
python scripts/download_zenodo_data.py --record "$ZENODO_RECORD" --out data

# 3. audits (public mode: reads only the downloaded bundle)
python p2/audit.py --data-dir data
python p3/audit.py --data-dir data

# 4. figures (optional)
#    The figure scripts read the project working tree (results/, cache/,
#    v3_results/), which is not part of this repository. The values they plot
#    are shipped as Figures/*.json inside the data bundle, so every panel can
#    be checked numerically without re-running the models.
```

### Reproduction status

Both audits run from a clean checkout against the published bundle:

```bash
python p2/audit.py --data-dir data   # 21 arms x 10 seeds, headline means/s.d.
python p3/audit.py --data-dir data   # cohort invariants, C95, N_eff identity
```

Without `--data-dir` they run in project mode against the internal working
layout (`cache/`, `results/`, `v3_results/`) and additionally compare the
manuscript text (`--doc`). Public mode verifies the released data and the
headline statistics; it does not re-fit the NB2 models.

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

See `CITATION.cff`. Archive this code via the Zenodo concept DOI
`10.5281/zenodo.24189991`; the data are at `10.5281/zenodo.24187222`.

## Dependencies

The complete environment is the container (`container/Dockerfile`). For the
audits alone, `pip install -r requirements-audit.txt` is enough: public mode
needs numpy and pandas only. `requirements.txt` lists the full stack used for
training and conformer generation.

## Contact

Corresponding author (see the manuscript). Please open an issue for
reproducibility problems rather than emailing, so that fixes are visible.
