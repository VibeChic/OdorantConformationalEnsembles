# EN4 size-vs-shape decomposition — results

**Run window**: 2026-11-03 19:12–20:03 (server time) · **GPU**: RTX 2080 Ti, 3 runs in parallel
**Plan**: `PLAN.md`, written before the first run · **20/20 runs completed**, no failures
**Code**: `run_en4_decomp.py` (training, injects arms into `p2.train_s2.ARMS` at runtime; the
original script and all existing `results/` directories were left untouched),
`summarize_en4_decomp.py` (statistics)

---

## 1. What was run

`EN4`'s six scalars were partitioned exhaustively and disjointly:

| arm | scalars | count |
|---|---|---|
| `en4_size` | `Rg_mean`, `Rg_sd` | 2 |
| `en4_shape` | `NPR1_mean`, `NPR2_mean`, `NPR1_sd`, `NPR2_sd` | 4 |

Identical backbone, data, split, hyperparameters and seeds 0–9 as every other arm
(100 epochs max, patience 15, AdamW 3e-4, wd 1e-4, batch 32, dropout 0.15, training-set
z-scoring). Only the appended scalar block differs.

## 2. Results

| arm | scalars | macro-AUCPR | best epoch (median) | parameters |
|---|---|---|---|---|
| `R1` | — | 0.3187 ± 0.0029 | 84 | 433,549 |
| `EN1` | n_rotors (1) | 0.3240 ± 0.0038 | 91 | 544,119 |
| `en4_shape` | NPR1/2 mean+sd (4) | **0.3197 ± 0.0031** | 86 | 544,215 |
| `EN3` | N_eff, top_pop, n_rotors (3) | 0.3260 ± 0.0023 | 94 | 544,183 |
| `EN4` | all six | 0.3275 ± 0.0039 | 90 | 544,279 |
| **`en4_size`** | `Rg_mean`, `Rg_sd` (2) | **0.3288 ± 0.0035** | 96 | 544,151 |

Mean ± sample SD over 10 seeds, 730 held-out odorants.

| contrast | Δ | 95% CI | seeds | exact p | TOST p | verdict |
|---|---|---|---|---|---|---|
| `en4_size − R1` | **+0.0101** | +0.0069, +0.0133 | 10/10 | 0.0020 | 0.997 | superior |
| `en4_shape − R1` | +0.0010 | −0.0013, +0.0034 | 6/10 | 0.354 | **0.002** | **equivalent to 2D** |
| `en4_size − EN1` | +0.0048 | +0.0013, +0.0083 | 8/10 | 0.0195 | 0.444 | superior (marginal) |
| `en4_shape − EN1` | **−0.0043** | −0.0068, −0.0018 | 1/10 | 0.0039 | 0.271 | **inferior to a rotor count** |
| `en4_size − en4_shape` | **+0.0091** | +0.0066, +0.0116 | 10/10 | 0.0020 | 0.998 | superior |
| `EN4 − en4_size` | −0.0012 | −0.0038, +0.0014 | 4/10 | 0.354 | **0.004** | **equivalent** |
| `EN4 − en4_shape` | +0.0079 | +0.0046, +0.0111 | 10/10 | 0.0020 | 0.960 | superior |

**Additivity**: (size − R1) + (shape − R1) = +0.0111 versus EN4 − R1 = +0.0089; the excess is
+0.0022 (−0.0017, +0.0062; p = 0.248). The two components are roughly additive with a small
redundancy.

**Near-duplicate sensitivity** (test molecules with max Tanimoto < 0.90; n = 635): `en4_size`
0.3235 (−0.0053), `en4_shape` 0.3149 (−0.0048), against `R1` 0.3131. The pattern is unchanged
(`en4_size − R1` = +0.0104; `en4_shape − R1` = +0.0018).

**Secondary contrasts for `en4_size`**: versus `EN3` +0.0027 (−0.0000, +0.0055; 7/10;
p = 0.0449, boundary); versus `EN5` +0.0018 (p = 0.072); versus `R3unif` +0.0034 (p = 0.063).
The top of the ladder is therefore a plateau of three to four arms, not a single winner.

## 3. Verdict against the pre-specified rules

`PLAN.md` fixed four possible outcomes. **Rule 1 fired: the gain is molecular size.**

1. Two radius-of-gyration moments reproduce the entire six-scalar arm (`EN4 − en4_size` is
   TOST-equivalent at a ±0.005 margin, p = 0.004) and are numerically the best arm in the study.
2. The four NPR shape moments are statistically **equivalent to the bare 2D graph**
   (+0.0010, TOST p = 0.002) and **worse than a single rotor count** (−0.0043, 1/10 seeds).
3. Size beats shape by +0.0091 (10/10 seeds), the largest effect among the scalar contrasts.

So `EN4` is not a "3D shape" arm. It is a **molecular size and size-fluctuation** arm: `Rg_mean`
captures overall extent, `Rg_sd` captures how much that extent varies across the ensemble.
Neither is rotameric shape, and `Rg_mean` is 92.3 % predictable from four raw 2D descriptors
(manuscript §2.6), so both quantities sit on the 2D side of the information boundary.

## 4. Changes applied to the manuscript (2026-11-03, after the user accepted both points)

| item | was | now |
|---|---|---|
| Arm description | "`EN4`, six shape moments" | "six ensemble moments—two of size, four of shape" (§2.2, Table 1, Fig. 1) |
| `EN4 − EN1` | indeterminate (p = 0.098, TOST 0.234) | kept as indeterminate in the direct test, then resolved by the decomposition paragraph in §2.3 |
| Title / abstract | "add coarse flexibility, not thermodynamics" | "add coarse flexibility **and molecular size**, not thermodynamics" |
| Limitations | "a decomposition … was not run" | replaced with the result, plus the count-mismatch and NPR-only caveats |
| Tables | 17-arm ladder only | new **Table 4** reports the two component arms and six contrasts; the 17-arm ladder is unchanged |

The finding strengthens the paper's thesis rather than weakening it: the best 3D-derived
quantity is a size proxy that 2D topology already supplies at R² = 0.92, and genuine 3D shape
descriptors add nothing over a rotatable-bond count.

## 5. Caveats

- **Multiplicity.** `en4_size − EN1` (p = 0.0195) does not survive Bonferroni correction for the
  three pre-specified endpoints (threshold 0.0167). The decisive evidence is instead the pair
  "`EN4` ≡ size-only" (TOST) and "size ≫ shape" (p = 0.0020, 10/10), neither of which depends on
  the marginal comparison.
- **Feature count.** The components differ in scalar count (2 vs 4); no count-matched control was
  run (declared in advance). The direction of the result (fewer scalars doing better) makes a
  count artefact unlikely but does not exclude it.
- **What "shape" means here.** NPR1/NPR2 are normalised principal-moment ratios and carry no size
  information by construction; other shape descriptors (e.g. surface anisotropy, USRCAT moments)
  were not tested.
- **One schedule.** All arms share one hyperparameter set; `en4_size` stops at epoch 96 versus 84
  for `R1`, i.e. it is not an early-stopping artefact.

## 6. Files

| file | content |
|---|---|
| `PLAN.md` | pre-specified analysis plan |
| `run_en4_decomp.py` | training driver (resumable; injects arms at runtime) |
| `results/{en4_size,en4_shape}/seed0-9/` | per-run `metrics.json`, `config_used.yaml`, `scalar_transform.json`, `test_preds.npz` |
| `logs/` | per-run training logs |
| `runs.json` | flat table of the 20 runs |
| `summarize_en4_decomp.py`, `en4_decomp_summary.json` | statistics and machine-readable summary |
