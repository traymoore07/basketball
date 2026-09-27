"""Proper scoring rules and forecast summaries.

Conventions (pre-registered, see docs/prereg/08_metric_spec.md):

* Every score is a LOSS: lower is better.
* Marginal forecasts of integer targets are probability mass functions (pmfs)
  over the support {0, 1, ..., K}; ``pmf`` arrays have shape (n, K + 1).
  Outcomes above K are clipped into the top bin, so each target's K must be
  chosen large enough that this essentially never binds (see targets.py).
* Log scores use base 2 (bits) and are computed on a pmf that has been mixed
  with a uniform distribution using ``LOG_SCORE_EPS``. The same transform is
  applied to every model, so no model is advantaged, and a single impossible
  outcome cannot produce an infinite score.
* Joint forecasts are Monte Carlo samples with shape (S, d).
"""

from __future__ import annotations

import numpy as np

LOG_SCORE_EPS = 1e-4  # pre-registered uniform mixture weight for log scores


# --------------------------------------------------------------------------
# pmf utilities
# --------------------------------------------------------------------------

def normalize_pmf(pmf: np.ndarray) -> np.ndarray:
    pmf = np.clip(np.asarray(pmf, dtype=float), 0.0, None)
    s = pmf.sum(axis=-1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("pmf row with zero total mass")
    return pmf / s


def smooth_pmf(pmf: np.ndarray, eps: float = LOG_SCORE_EPS) -> np.ndarray:
    pmf = normalize_pmf(pmf)
    k = pmf.shape[-1]
    return (1.0 - eps) * pmf + eps / k


def clip_outcome(y: np.ndarray, k_max: int) -> np.ndarray:
    y = np.asarray(y)
    if np.any(y < 0):
        raise ValueError("negative outcome for a count target")
    return np.minimum(y.astype(int), k_max)


def pmf_from_samples(samples: np.ndarray, k_max: int) -> np.ndarray:
    """Empirical pmf(s) from integer samples.

    samples: shape (S,) or (n, S). Returns (K+1,) or (n, K+1).
    """
    s = np.asarray(samples)
    squeeze = s.ndim == 1
    if squeeze:
        s = s[None, :]
    s = np.clip(np.rint(s).astype(int), 0, k_max)
    n, S = s.shape
    out = np.zeros((n, k_max + 1))
    rows = np.repeat(np.arange(n), S)
    np.add.at(out, (rows, s.ravel()), 1.0)
    out /= S
    return out[0] if squeeze else out


def smoothed_pmf_from_samples(samples: np.ndarray, k_max: int, bw_scale: float = 1.0) -> np.ndarray:
    """Kernel-smoothed pmf from integer Monte Carlo samples (reflected at 0).

    Discrete Gaussian kernel with a Silverman-type bandwidth (>= 0.5). Simulation-based
    models use this so that log scores measure the model rather than Monte Carlo noise.
    """
    x = np.clip(np.rint(np.asarray(samples, float)), 0, k_max).astype(int)
    n = len(x)
    if n == 0:
        out = np.zeros(k_max + 1)
        out[0] = 1.0
        return out
    hist = np.bincount(x, minlength=k_max + 1).astype(float)
    sd = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = max(0.5, bw_scale * 0.9 * spread * n ** (-0.2))
    half = int(np.ceil(4 * h))
    offs = np.arange(-half, half + 1)
    ker = np.exp(-0.5 * (offs / h) ** 2)
    ker /= ker.sum()
    full = np.convolve(hist, ker)              # index i corresponds to value i - half
    vals = np.arange(len(full)) - half
    out = np.zeros(k_max + 1)
    pos = (vals >= 0) & (vals <= k_max)
    out[vals[pos]] += full[pos]
    neg = vals < 0
    out[np.minimum(-vals[neg] - 1, k_max)] += full[neg]   # reflect below zero
    out[k_max] += full[vals > k_max].sum()
    return out / out.sum()


def cdf(pmf: np.ndarray) -> np.ndarray:
    return np.cumsum(normalize_pmf(pmf), axis=-1)


def pmf_mean(pmf: np.ndarray) -> np.ndarray:
    pmf = normalize_pmf(pmf)
    return pmf @ np.arange(pmf.shape[-1])


def pmf_quantile(pmf: np.ndarray, q: float) -> np.ndarray:
    """Smallest k with F(k) >= q (per row)."""
    F = cdf(pmf)
    return (F < q - 1e-12).sum(axis=-1)


# --------------------------------------------------------------------------
# Scores for integer (count) targets given pmfs
# --------------------------------------------------------------------------

def log_score(pmf: np.ndarray, y: np.ndarray, eps: float = LOG_SCORE_EPS) -> np.ndarray:
    """Negative log2 probability of the observed outcome (bits)."""
    p = smooth_pmf(pmf, eps)
    y = clip_outcome(y, p.shape[-1] - 1)
    return -np.log2(p[np.arange(len(y)), y])


def crps_pmf(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact CRPS for an integer-valued forecast and outcome.

    CRPS = sum_k (F(k) - 1{k >= y})^2 over integers k; terms with k >= K are 0.
    For a point-mass forecast this reduces to |x - y| (absolute error).
    """
    F = cdf(pmf)
    K = F.shape[-1]
    y = clip_outcome(y, K - 1)
    ks = np.arange(K)
    ind = (ks[None, :] >= y[:, None]).astype(float)
    return ((F - ind) ** 2)[:, : K - 1].sum(axis=-1)


def tw_crps_upper(pmf: np.ndarray, y: np.ndarray, threshold: int) -> np.ndarray:
    """Threshold-weighted CRPS with weight 1{k >= threshold} (upper-tail emphasis)."""
    F = cdf(pmf)
    K = F.shape[-1]
    y = clip_outcome(y, K - 1)
    ks = np.arange(K)
    ind = (ks[None, :] >= y[:, None]).astype(float)
    w = (ks >= threshold).astype(float)
    return (w[None, : K - 1] * (F - ind)[:, : K - 1] ** 2).sum(axis=-1)


def bin_pmf(pmf: np.ndarray, edges: list[int]) -> np.ndarray:
    """Aggregate integer pmf into bins. edges=[20,25,30,35] -> <20,20-24,25-29,30-34,35+."""
    pmf = normalize_pmf(pmf)
    K = pmf.shape[-1]
    bounds = [0] + list(edges) + [K]
    return np.stack([pmf[:, a:b].sum(axis=-1) for a, b in zip(bounds[:-1], bounds[1:])], axis=-1)


def bin_outcome(y: np.ndarray, edges: list[int]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), np.asarray(y), side="right")


def rps(prob_bins: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """Ranked probability score for ordered categories, normalised by (m - 1)."""
    P = np.cumsum(prob_bins, axis=-1)
    m = P.shape[-1]
    O = (np.arange(m)[None, :] >= np.asarray(y_bin)[:, None]).astype(float)
    return ((P - O) ** 2)[:, : m - 1].sum(axis=-1) / (m - 1)


# --------------------------------------------------------------------------
# Binary events
# --------------------------------------------------------------------------

def brier(p: np.ndarray, y01: np.ndarray) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y01, float)) ** 2


def binary_log_score(p: np.ndarray, y01: np.ndarray, eps: float = LOG_SCORE_EPS) -> np.ndarray:
    p = (1 - eps) * np.asarray(p, float) + eps / 2
    y = np.asarray(y01, float)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


# --------------------------------------------------------------------------
# Point-forecast scores (diagnostic only; each is consistent for one functional)
# --------------------------------------------------------------------------

def abs_error_of_median(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pmf_quantile(pmf, 0.5) - np.asarray(y))


def sq_error_of_mean(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (pmf_mean(pmf) - np.asarray(y)) ** 2


# --------------------------------------------------------------------------
# PIT and coverage
# --------------------------------------------------------------------------

def randomized_pit(pmf: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised PIT for discrete forecasts: U = F(y-1) + V * p(y), V ~ U(0,1).

    Uniform on (0,1) iff the forecast is calibrated (Czado, Gneiting & Held 2009).
    """
    pmf = normalize_pmf(pmf)
    F = np.cumsum(pmf, axis=-1)
    y = clip_outcome(y, pmf.shape[-1] - 1)
    idx = np.arange(len(y))
    upper = F[idx, y]
    lower = upper - pmf[idx, y]
    return lower + rng.random(len(y)) * (upper - lower)


def pit_coverage(pit: np.ndarray, level: float) -> float:
    """Fraction of PIT values inside the central `level` interval.

    Equal to `level` in expectation for a calibrated forecast, which avoids the
    over-coverage that discrete quantile intervals have by construction.
    """
    lo, hi = (1 - level) / 2, (1 + level) / 2
    pit = np.asarray(pit)
    return float(np.mean((pit > lo) & (pit < hi)))


def pit_histogram(pit: np.ndarray, bins: int = 10) -> np.ndarray:
    h, _ = np.histogram(pit, bins=bins, range=(0, 1))
    return h / max(len(pit), 1)


def pit_ks(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0,1)."""
    u = np.sort(np.asarray(pit))
    n = len(u)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def pit_dispersion_ratio(pit: np.ndarray) -> float:
    """Var(PIT) / (1/12). > 1 means forecasts are too narrow; < 1 too wide."""
    return float(np.var(pit) * 12.0)


# --------------------------------------------------------------------------
# Sample-based scores (continuous or joint)
# --------------------------------------------------------------------------

def crps_samples(samples: np.ndarray, y: float) -> float:
    """CRPS from an ensemble: E|X - y| - 0.5 E|X - X'| (O(S log S))."""
    x = np.sort(np.asarray(samples, float))
    S = len(x)
    term1 = np.mean(np.abs(x - y))
    i = np.arange(1, S + 1)
    term2 = np.sum((2 * i - S - 1) * x) / (S * S)
    return float(term1 - term2)


def energy_score(samples: np.ndarray, y: np.ndarray, scale: np.ndarray | None = None,
                 max_pairs: int = 4000, rng: np.random.Generator | None = None) -> float:
    """Energy score ES = E||X - y|| - 0.5 E||X - X'|| (Gneiting et al. 2008).

    `scale` divides each component (pre-registered per-target scales, fitted on
    training data only) so that high-variance stats do not dominate.
    """
    X = np.asarray(samples, float)
    y = np.asarray(y, float)
    if scale is not None:
        X = X / scale
        y = y / scale
    t1 = np.mean(np.linalg.norm(X - y[None, :], axis=1))
    S = X.shape[0]
    rng = rng or np.random.default_rng(0)
    m = min(max_pairs, S * (S - 1) // 2)
    a = rng.integers(0, S, m)
    b = rng.integers(0, S, m)
    keep = a != b
    t2 = np.mean(np.linalg.norm(X[a[keep]] - X[b[keep]], axis=1))
    return float(t1 - 0.5 * t2)


def variogram_score(samples: np.ndarray, y: np.ndarray, p: float = 0.5,
                    weights: np.ndarray | None = None, scale: np.ndarray | None = None) -> float:
    """Variogram score of order p (Scheuerer & Hamill 2015).

    Sensitive to the dependence structure between components, unlike sums of
    marginal scores. Lower is better.
    """
    X = np.asarray(samples, float)
    y = np.asarray(y, float)
    if scale is not None:
        X = X / scale
        y = y / scale
    d = X.shape[1]
    if weights is None:
        weights = np.ones((d, d))
    obs = np.abs(y[:, None] - y[None, :]) ** p
    # E|X_i - X_j|^p via samples: mean over S of |X_i - X_j|^p
    ev = np.mean(np.abs(X[:, :, None] - X[:, None, :]) ** p, axis=0)
    iu = np.triu_indices(d, 1)
    return float(np.sum(weights[iu] * (obs[iu] - ev[iu]) ** 2))
