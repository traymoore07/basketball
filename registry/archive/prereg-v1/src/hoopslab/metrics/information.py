"""Information-theoretic instruments: usable information, plug-in entropy (for contrast),
placebo controls, and an arithmetic coder demonstrating code length = log loss.

Usable information (Xu et al. 2020, "V-information") of context C about Y given base
features B, for a model family V, is estimated OUT OF SAMPLE:

    I_V(Y; C | B) ~= mean_test[-log2 q_B(y)] - mean_test[-log2 q_{B,C}(y)]

where q_B and q_{B,C} are fitted on training data only. Unlike the plug-in
H(Y|B) - H(Y|B,C) on the same data, this cannot be inflated by sparse cells: a
context with no information gives <= 0 in expectation, since overfitting only
hurts held-out log loss. Uncertainty comes from a cluster (game) bootstrap of
per-event differences.
"""

from __future__ import annotations

import numpy as np

from .compare import cluster_bootstrap


def plugin_entropy(y: np.ndarray, k: int | None = None) -> float:
    c = np.bincount(y, minlength=k or 0).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def plugin_conditional_entropy(y: np.ndarray, ctx: np.ndarray) -> float:
    """In-sample plug-in H(Y | C). Biased DOWNWARD with sparse contexts (for contrast only)."""
    y = np.asarray(y)
    ctx = np.asarray(ctx)
    _, inv = np.unique(ctx, return_inverse=True)
    h = 0.0
    n = len(y)
    for c in np.unique(inv):
        m = inv == c
        h += m.sum() / n * plugin_entropy(y[m])
    return h


def bits_per_event(P: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = P[np.arange(len(y)), y]
    return -np.log2(np.clip(p, eps, 1.0))


def usable_information(bits_base: np.ndarray, bits_ctx: np.ndarray, cluster: np.ndarray,
                       B: int = 1000, block: int = 1, rng=None) -> dict:
    """Mean held-out bits saved by the context (positive = informative) with a cluster bootstrap CI."""
    vals = {"a": bits_base, "b": bits_ctx, "n": np.ones_like(bits_base)}
    stat = lambda s: (s["a"].sum(-1) - s["b"].sum(-1)) / s["n"].sum(-1)  # noqa: E731
    est, draws = cluster_bootstrap(vals, cluster, stat, B=B, block=block, rng=rng)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"bits": float(est), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float((1 + np.sum(draws <= 0)) / (len(draws) + 1)),
            "base_bits": float(np.mean(bits_base)), "n": int(len(bits_base))}


# --------------------------------------------------------------------------
# Arithmetic coding (Witten, Neal & Cleary 1987), 32-bit integer implementation
# --------------------------------------------------------------------------

_PREC = 32
_FULL = (1 << _PREC) - 1
_HALF = 1 << (_PREC - 1)
_QTR = 1 << (_PREC - 2)
FREQ_TOTAL = 1 << 16


def quantize(probs: np.ndarray, total: int = FREQ_TOTAL) -> np.ndarray:
    """Integer frequency table (each >= 1) summing to `total`, approximating probs."""
    f = np.maximum(1, np.floor(probs * (total - len(probs))).astype(np.int64))
    f[np.argmax(f)] += total - f.sum()
    return f


def encode(symbols: np.ndarray, prob_rows: np.ndarray) -> tuple[list[int], float]:
    """Encode symbols given per-step predictive distributions.

    Returns (bits, ideal_bits) where ideal_bits = sum -log2 q(symbol) under the
    quantised model. len(bits) <= ideal_bits + 2 (plus tiny finite-precision overhead).
    """
    low, high, pending = 0, _FULL, 0
    out: list[int] = []
    ideal = 0.0

    def emit(bit):
        nonlocal pending
        out.append(bit)
        out.extend([1 - bit] * pending)
        pending = 0

    for s, p in zip(symbols, prob_rows):
        f = quantize(p)
        cum = np.concatenate([[0], np.cumsum(f)])
        ideal += -np.log2(f[s] / FREQ_TOTAL)
        rng_ = high - low + 1
        high = low + (rng_ * int(cum[s + 1])) // FREQ_TOTAL - 1
        low = low + (rng_ * int(cum[s])) // FREQ_TOTAL
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QTR and high < 3 * _QTR:
                pending += 1
                low -= _QTR
                high -= _QTR
            else:
                break
            low = 2 * low
            high = 2 * high + 1
    pending += 1
    emit(0 if low < _QTR else 1)
    return out, ideal


def decode(bits: list[int], prob_rows: np.ndarray) -> np.ndarray:
    """Inverse of `encode` given the same predictive distributions (verifies losslessness)."""
    low, high = 0, _FULL
    value = 0
    it = iter(bits + [0] * _PREC)
    for _ in range(_PREC):
        value = (value << 1) | next(it)
    out = []
    for p in prob_rows:
        f = quantize(p)
        cum = np.concatenate([[0], np.cumsum(f)])
        rng_ = high - low + 1
        target = ((value - low + 1) * FREQ_TOTAL - 1) // rng_
        s = int(np.searchsorted(cum, target, side="right") - 1)
        out.append(s)
        high = low + (rng_ * int(cum[s + 1])) // FREQ_TOTAL - 1
        low = low + (rng_ * int(cum[s])) // FREQ_TOTAL
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QTR and high < 3 * _QTR:
                low -= _QTR
                high -= _QTR
                value -= _QTR
            else:
                break
            low = 2 * low
            high = 2 * high + 1
            value = (value << 1) | next(it)
    return np.array(out)
