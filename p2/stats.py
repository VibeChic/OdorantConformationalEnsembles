"""Small statistics helpers shared by the paper-2 experiment reports.

Two procedures, deliberately kept separate:

  sign_flip_test   exact paired permutation over seeds (2^n assignments, n <= 12).
                   This answers "does A beat B across training runs", which is
                   what a claim about a *model* requires.

  stratified_bootstrap
                   resamples molecules within label-prevalence strata and
                   returns a CI for a metric of the seed-averaged predictions.
                   This answers "would this hold on new molecules", a different
                   question. Never average these endpoints across seeds.
"""
import itertools

import numpy as np


def sign_flip_test(diffs, n_perm=None):
    """Exact two-sided sign-flip permutation test on paired differences."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "p": float("nan")}
    obs = abs(d.mean())
    if n <= 12:
        signs = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
        null = np.abs((signs * d).mean(axis=1))
        p = float((null >= obs - 1e-12).mean())
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([1.0, -1.0], size=(10000, n))
        null = np.abs((signs * d).mean(axis=1))
        p = float(((null >= obs - 1e-12).sum() + 1) / (len(null) + 1))
    return {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else 0.0,
            "p": p, "diffs": d.tolist()}


def macro_aucpr(Y, P, min_pos=1):
    from sklearn.metrics import average_precision_score
    vals = []
    for j in range(Y.shape[1]):
        if Y[:, j].sum() >= min_pos:
            vals.append(average_precision_score(Y[:, j], P[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def _grouped_macro(Y, P, groups):
    from sklearn.metrics import average_precision_score
    vals = []
    for g in groups:
        sub = [j for j in g if Y[:, j].sum() >= 1]
        if not sub:
            continue
        vals.append(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in sub]))
    return float(np.mean(vals)) if vals else float("nan")


def stratified_bootstrap(Y, P, n_boot=2000, seed=0):
    """CI for a macro-AUCPR that weights prevalence strata equally.

    Molecules are resampled with replacement; within each resample the macro is
    computed separately for well-powered (>=30% positives), mid (5-30%) and rare
    (<5%) labels, then averaged over the strata that still contain a positive.
    This stops 1-positive labels from dominating the macro average. It is a
    different estimator from the plain macro-AUCPR and is reported as such.
    """
    prev = Y.mean(0)
    groups = [np.where(prev >= 0.30)[0],
              np.where((prev >= 0.05) & (prev < 0.30))[0],
              np.where((prev < 0.05) & (Y.sum(0) >= 1))[0]]
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(_grouped_macro(Y[idx], P[idx], groups))
    stats = np.array([x for x in stats if np.isfinite(x)])
    return {"point": _grouped_macro(Y, P, groups),
            "lo": float(np.percentile(stats, 2.5)),
            "hi": float(np.percentile(stats, 97.5)),
            "n_boot": int(len(stats))}
