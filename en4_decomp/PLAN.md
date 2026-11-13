# EN4 size-vs-shape decomposition — pre-specified analysis plan

**Created**: 2026-11-03 19:10 (server time), before any run.
**Question**: `EN4` (six ensemble scalars) is the best arm (0.3275 ± 0.0039) and is statistically
indeterminate against `EN1` (`n_rotors`, 0.3240; Δ = +0.0036, p = 0.098, TOST p = 0.234).
The 2D GIN uses global mean pooling, which divides out heavy-atom count; `EN4` re-injects
molecular size through `Rg_mean`/`Rg_sd`. Does `EN4`'s edge come from **size** or from
**3D shape**?

**Partition** (exhaustive and disjoint; the six columns are exactly EN4's):
- `en4_size`  = [`Rg_mean`, `Rg_sd`]                                   (2 scalars)
- `en4_shape` = [`NPR1_mean`, `NPR2_mean`, `NPR1_sd`, `NPR2_sd`]       (4 scalars)

**Design**: identical backbone, data, split, hyperparameters and seeds as every other arm in
`p2/train_s2.py` (100 epochs max, patience 15, AdamW 3e-4, wd 1e-4, batch 32, dropout 0.15,
scalars z-scored on the training split). 10 seeds (0–9) per arm. Nothing else is changed.

**Primary endpoints** (fixed now):
1. `en4_size − EN1` and `en4_shape − EN1`: which component recovers `EN4`'s +0.0036?
2. `en4_size − en4_shape`: direct comparison of the two components.
3. Additivity check: `EN4 − R1` versus `en4_size − R1 + en4_shape − R1`.

**Statistics**: exact paired sign-flip permutation over 2^10 allocations (same as the
manuscript); 95 % paired t intervals for superiority; TOST against δ = 0.005 for parity.
Reference values are the locked runs already on disk (`results/r1/gin`, `results/s2_ablation/en1`,
`results/s2_ablation/en4`), not re-trained here.

**Interpretation rules fixed in advance**:
- If `en4_size ≈ EN4` and `en4_shape ≈ EN1` → the gain is molecular size, and the title's
  "coarse flexibility" must be extended to "coarse flexibility and molecular size".
- If `en4_shape ≈ EN4` and `en4_size ≈ EN1` → the gain is genuinely 3D shape, and the
  "size proxy" objection is answered.
- If neither reaches `EN4` → the two components act jointly and the six-scalar arm should be
  reported as a composite.
- If both are indeterminate → report as inconclusive and keep the composite framing.

**Known limitation (not fixed by this run)**: the two components differ in scalar count
(2 vs 4); a count-matched control would be a separate experiment and is not run here.
