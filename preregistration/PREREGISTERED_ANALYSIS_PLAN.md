# Paper 3 v3 — Frozen Analysis Plan

**Version**: v3.0-frozen
**Frozen at**: 2026-10-28 (server time; hash recorded in `00_v3_analysis_plan.sha256` and in `v3_logs/s0_freeze.log`)
**Frozen before**: any run of the remaining 200 molecules (S1), any ALPB calculation (S3), any `S_pharm` computation (S5), any 4D MIL training (S6).
**Already seen before freezing (declared)**: the P4 100-molecule pilot results (`v3_results/pilot100_*`), including its `β2` at both caps. See §2 for how this is handled.

---

## 1. Scientific question

Does the breadth and semantic complexity of an odorant's odour image depend on the depth of its Boltzmann conformational ensemble, or is it invariant to "silent" torsional entropy that does not change the 3D pharmacophore?

Two outcomes are publishable and neither is preferred by this plan:

- **Scenario A**: after removing the `Cmax = 20` truncation and weighting back to the capped population, `S_conf` (or `S_pharm`) carries a curvature/association signal that the truncated data masked.
- **Scenario B**: no signal survives. Odour breadth is then invariant to deep ensemble size, and the paper reports a bounded null with a stated minimum detectable effect.

---

## 2. Sample structure (declared, not chosen after the fact)

| Sample | n | Status |
|---|---|---|
| P4 pilot | 100 | **Exploratory**. Already analysed (`β2` at Cmax=20 and 200). Cannot be confirmatory. |
| New remainder | 200 | **Prospective / confirmatory**. Not yet run at freeze time. |
| Pooled | 300 | **Descriptive**. Reported alongside, never used to rescue a failed 200-sample test. |
| Capped population | 1,822 | Inference target for the weighted estimator. |
| Full cohort | 4,809 | Descriptive/context; only used for the frozen standardisation constants. |

The 200 is the complement of the 100 inside `pilot_300.csv` (verified: `pilot100 ⊂ pilot_300`, 200 remaining, all capped).

---

## 3. Frozen constants

Standardisation uses the **full 4,809 cohort** so that the predictor scale in the 200 is independent of the 200's own data. Source: `v3_cache/frozen_constants.json`.

| Variable | Mean | SD (population, ddof=0) |
|---|---:|---:|
| `S_conf(20)` | 1.597200 | 0.918917 |
| `N_eff` | 7.030259 | 5.115756 |
| `MW` | 176.700702 | 56.077556 |
| `cLogP` | 2.617251 | 1.402220 |
| `N_HA` | 1.950302 | 1.118208 |
| `TPSA` | 26.208232 | 18.508894 |
| `n_unspecified_stereo` | 0.951341 | 1.267556 |

`S_conf` is dimensionless (`−Σ p ln p`, nats). `N_eff = exp(S_conf)`. Never divide by R.

---

## 4. Primary estimand and test

**Population**: the 1,822 capped molecules (`Cmax = 20` in the v2.0 cache).

**Confirmatory sample**: the new 200.

**Strata**: 40 cells = `N_eff` decile (10) × `MW` quartile (4), both cut on the 1,822 capped population. `run_pass` is **not** a stratum (it is a technical batch); it enters as a covariate. Verified at freeze: all 40 cells have `n_200 ≥ 1`; `w = N_h / n_h` ranges 4.0–21.0 (`v3_cache/strata40_design.csv`).

**Model**: Negative Binomial NB2, log link.

```
L_sem ~ 1 + S + S² + MW_z + cLogP_z + N_HA_z + TPSA_z + run_pass_1b + n_unspec_z
```

- `S` = `S_conf(200)` standardised by the frozen constants in §3.
- `n_unspec_z` = `n_unspecified_stereo` standardised by the frozen constants.
- All molecules in the 200 are capped at 20 in v2.0, so no `capped20` term is needed in this sample.
- Weights `w = N_h / n_h`; `svy`-style weighted NB2 (pseudo-maximum-likelihood), with `α` estimated.

**Inference**: 1,000 bootstrap replicates resampled **within stratum** (molecules are the units); percentile 95% CI; design df = number of strata − 1 = 39.

**Primary decision rule (α = 0.05, two-sided)**:

| Code | Condition | Claim permitted |
|---|---|---|
| **D1** | 95% CI for `β2` excludes 0 | "ensemble depth carries a curvature signal" (direction as estimated) |
| **D2** | additionally CI upper bound < 0 | "inverted-U" (the blueprint's hypothesis) |
| **D3** | CI includes 0 | report the estimate and CI; **no peak claim** in the abstract |

A positive, significant `β2` is **not** evidence for the blueprint's inverted U, and must be reported as monotone convex.

**Mandatory co-reports**: unweighted fit on the same 200; weighted fit on the pooled 300 (descriptive); P4 fit reproduced bit-for-bit from raw CSVs (S0.5).

---

## 5. Pre-declared escalation (power)

At n = 100 the observed `β2` CI half-width was 0.069 (SE ≈ 0.035). At n = 200 the expected half-width is ≈ 0.049 and the 80%-power MDES ≈ 0.07. A null at n = 200 therefore cannot distinguish "no curvature" from "curvature below 0.05".

**Rule, fixed now**:

1. If the 200-sample 95% CI half-width **> 0.05** → escalate to the **full 1,822 capped molecules** (~34 h) and re-fit the identical model, unweighted (the population is then fully observed, so weights are 1).
2. If the half-width **≤ 0.05** → stop at 300 and report the estimate as final for the cap question.
3. The escalation is triggered by CI width only, **never** by the sign or significance of the point estimate.

---

## 6. Secondary endpoints

| Exp | Endpoint | Analysis | Multiplicity |
|---|---|---|---|
| 1 | Saturation `S(C) = S∞(1 − e^{−C/τ})`, `C95` | per-molecule NLS on subsampled pools `C ∈ {1,2,5,10,15,20,30,50,75,100,150,200}`; monotone-spline sensitivity; `C95` by `N_rot`/MW | descriptive |
| 3 | `β2` under water and octanol reweighting; `I_cham` | `G_i(solv) = G_i(gas,ohess) + ΔG_solv,i(ALPB)`; refit §4 model | BH across 3 tests |
| 4 | `S_pharm` (USRCAT) | `S_pharm = ½ ln det(2πe Σ_LW)`; ΔAIC / ΔR² vs `S_conf` | BH across 4 tests |
| 5 | Cliff discrimination | AUROC of `abs(ΔS)`, `W2(NPR)`, `abs(ΔN_eff)` on 116 test cliffs; Tanimoto > 0.85 sensitivity | BH across 3 tests |
| 6 | 4D MIL | macro-AUPRC primary, macro-AUROC secondary; 3 seeds; random split + scaffold-disjoint | BH across tiers |

**Promotion rule (`S_pharm`)**: primary only if ΔAIC > 10 **or** ΔR² > 0.02 over the `S_conf` model. Otherwise secondary.

**Tier definition for Exp 6**:

| Tier | Representation | n |
|---|---|---|
| A | 2D graph | 4,809 |
| B | single lowest-free-energy conformer | 4,809 |
| C | Cmax=20 ensemble | 4,809 |
| D | Cmax=200 ensemble | **3,287** = 2,987 uncapped (`C_total < 20`, so C ≡ D) + 300 re-run |

The 1,522 capped molecules not re-run never enter a Tier D comparison.

---

## 7. Gates

| Gate | Condition to proceed |
|---|---|
| G0 | this file hashed and logged before S1 |
| G1 | ≥ 195/200 new molecules succeed; no NaN; `ΔS` distribution compatible with P4 (KS p > 0.01) |
| G3a | primary fit converges; weighted and unweighted `β2` share sign; no zero-`n_h` cell |
| G2 | gas single points reproduce cached `g_abs` within 1 kcal/mol (20 molecules); re-opt changes `S_conf` < 0.15 nats |
| G4 | `S_pharm` promotion rule (§6) |
| G5 | 4D MIL runs only if S2–S5 complete and Tier D vs C CI excludes 0, or the PI overrides |

---

## 8. What is forbidden after the hash

No change to: outcome, cap, covariate set, weights, strata, test direction, sample split, escalation rule, promotion rule, tier definition.

Any change → dated entry in `v3_reports/01_deviations.md`, and the affected analysis is demoted to exploratory in the manuscript.

Ambiguities that this plan does **not** resolve (and that will be reported as such): the choice of spline basis for Exp 1 sensitivity, the GBM hyper-parameters for imputation, the MIL seed set. These are fixed in code at first run and recorded in `v3_logs/`.

---

## 9. Code and data provenance

- Only `v3_*` paths are written. `v2.0`, `results/`, `reports/`, `cache/` are frozen (`v3_reports/00_freeze_manifest.sha256`, 46 files).
- `v2.0` sha256: `10f6c7741f2b0ef9716243bc863df5516b9d3f37fe279cec951608fc5df1b1e6`.
- Engine versions at freeze: xtb 6.7.1; Docker `landscape-md:1.1` (OpenFF 0.19.0, OpenMM 8.6.1, CUDA 12.9.1); R 4.6.1 with lme4/glmmTMB.

---

**End of frozen plan.**
