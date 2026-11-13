"""Statistical helpers for Paper 3.

Implements negative-binomial (NB2) and beta-binomial GLMs with scipy, plus
molecule-level bootstrap. statsmodels is not installed in this environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -20.0, 20.0))


@dataclass
class NBFit:
    beta: np.ndarray
    alpha: float
    loglik: float
    converged: bool
    n_iter: int
    hess_inv: np.ndarray | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _safe_exp(X @ self.beta)


def nb_loglik(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    beta = theta[:-1]
    log_alpha = float(theta[-1])
    alpha = float(np.exp(np.clip(log_alpha, -12.0, 5.0)))
    mu = _safe_exp(X @ beta)
    # NB2 log-likelihood; alpha -> 0 handled by clipping
    ll = (
        gammaln(y + 1.0 / alpha)
        - gammaln(y + 1.0)
        - gammaln(1.0 / alpha)
        + y * np.log(alpha * mu)
        - (y + 1.0 / alpha) * np.log1p(alpha * mu)
    )
    return float(np.sum(ll))


def fit_nb(X: np.ndarray, y: np.ndarray, init: np.ndarray | None = None, maxiter: int = 3000) -> NBFit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if init is None:
        beta0 = np.zeros(p)
        beta0[0] = np.log(max(y.mean(), 1e-6))
        init = np.r_[beta0, np.log(1.0)]
    bounds = [(None, None)] * p + [(-12.0, 5.0)]
    res = minimize(
        lambda th: -nb_loglik(th, X, y),
        init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7},
    )
    return NBFit(
        beta=res.x[:-1],
        alpha=float(np.exp(np.clip(res.x[-1], -12.0, 5.0))),
        loglik=float(-res.fun),
        converged=bool(res.success),
        n_iter=int(res.nit),
        hess_inv=getattr(res, "hess_inv", None),
    )


def nb_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261025,
    init: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Molecule-level nonparametric bootstrap; returns (n_boot, p+1) [beta, alpha]."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    p = X.shape[1]
    out = np.full((n_boot, p + 1), np.nan, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            fit = fit_nb(X[idx], y[idx], init=init)
            out[b, :p] = fit.beta
            out[b, p] = fit.alpha
        except Exception:
            pass
        if progress is not None and (b + 1) % 100 == 0:
            progress(b + 1, n_boot)
    return out


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    lo = (1.0 - level) / 2.0 * 100.0
    hi = (1.0 - lo / 100.0) * 100.0
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def bootstrap_p_value(samples: np.ndarray, null: float = 0.0) -> float:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    # two-sided percentile p-value with +1 correction
    le = np.sum(x <= null)
    ge = np.sum(x >= null)
    return float(2.0 * min(le, ge) / len(x))


# ---------------------------------------------------------------------------
# Beta-binomial GLM (logit link), used by WP2.
# ---------------------------------------------------------------------------

def _bb_loglik(theta: np.ndarray, X: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    beta = theta[:-1]
    log_phi = float(theta[-1])
    phi = float(np.exp(np.clip(log_phi, -8.0, 8.0)))
    eta = np.clip(X @ beta, -20.0, 20.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    a = phi * p
    b = phi * (1.0 - p)
    # log C(n,k) + log B(k+a, n-k+b) - log B(a,b)
    ll = (
        gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
        + gammaln(k + a) + gammaln(n - k + b) - gammaln(n + a + b)
        - gammaln(a) - gammaln(b) + gammaln(a + b)
    )
    return float(np.sum(ll))


@dataclass
class BBFit:
    beta: np.ndarray
    phi: float
    loglik: float
    converged: bool
    n_iter: int

    def predict_p(self, X: np.ndarray) -> np.ndarray:
        eta = np.clip(X @ self.beta, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-eta))


def fit_beta_binomial(
    X: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    init: np.ndarray | None = None,
    maxiter: int = 3000,
) -> BBFit:
    X = np.asarray(X, dtype=float)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = X.shape[1]
    if init is None:
        # crude logit init, clipped
        prop = np.clip((k + 0.5) / (n + 1.0), 1e-4, 1 - 1e-4)
        beta0 = np.zeros(p)
        beta0[0] = np.log(np.mean(prop) / (1 - np.mean(prop)))
        init = np.r_[beta0, np.log(5.0)]
    bounds = [(None, None)] * p + [(-8.0, 8.0)]
    res = minimize(
        lambda th: -_bb_loglik(th, X, k, n),
        init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-11},
    )
    return BBFit(
        beta=res.x[:-1],
        phi=float(np.exp(np.clip(res.x[-1], -8.0, 8.0))),
        loglik=float(-res.fun),
        converged=bool(res.success),
        n_iter=int(res.nit),
    )


def beta_binomial_bootstrap(
    X: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    cluster: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 20261025,
) -> np.ndarray:
    """Bootstrap resampling molecules, or clusters (e.g. receptors) if given."""
    rng = np.random.default_rng(seed)
    if cluster is None:
        units = np.arange(X.shape[0])
        cluster_of = np.arange(X.shape[0])
    else:
        units = np.unique(cluster)
        cluster_of = np.asarray(cluster)
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([np.where(cluster_of == u)[0] for u in chosen]) if cluster is not None else chosen
        try:
            fit = fit_beta_binomial(X[idx], k[idx], n[idx])
            out.append(np.r_[fit.beta, fit.phi])
        except Exception:
            out.append(np.full(X.shape[1] + 1, np.nan))
    return np.asarray(out, dtype=float)
