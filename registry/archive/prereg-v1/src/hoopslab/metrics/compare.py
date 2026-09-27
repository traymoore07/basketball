"""Paired model comparison, uncertainty estimates, and the pre-registered decision rule.

Units (player-games, events, games) are grouped into *clusters* (by default the
game-day). Scores of different units within a cluster are dependent: teammates
share a game, and all forecasts issued on the same day share model state. Clusters
are resampled with a moving-block bootstrap over time so that serial dependence
between nearby days is preserved as well.

Decision rule (docs/prereg/08_metric_spec.md, "Scorecard rule"). Given an effect
estimate E (oriented so that positive = candidate better), its bootstrap CI
[lo, hi], a minimum practical effect MPE, and a multiplicity-adjusted p-value:

    SUPPORTED     p_adj < alpha  and  lo > 0  and  E >= MPE
    HARMFUL       hi < 0
    FALSIFIED     hi < MPE        (confidently smaller than practically relevant)
    PROMISING     otherwise, with E >= MPE (supports further research, proves nothing)
    INCONCLUSIVE  otherwise
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

DEFAULT_B = 2000
DEFAULT_BLOCK = 7
ALPHA = 0.05


def cluster_sums(values: dict[str, np.ndarray], cluster: np.ndarray):
    """Sum each unit-level array within clusters, clusters sorted by key."""
    keys, inv = np.unique(np.asarray(cluster), return_inverse=True)
    out = {}
    for name, v in values.items():
        v = np.asarray(v, float)
        s = np.zeros(len(keys))
        np.add.at(s, inv, v)
        out[name] = s
    return keys, out


def moving_block_indices(n_clusters: int, B: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n_clusters))
    n_blocks = int(np.ceil(n_clusters / block))
    starts = rng.integers(0, n_clusters - block + 1, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)
    return idx[:, :n_clusters]


def cluster_bootstrap(values: dict[str, np.ndarray], cluster: np.ndarray,
                      stat: Callable[[dict[str, np.ndarray]], np.ndarray],
                      B: int = DEFAULT_B, block: int = DEFAULT_BLOCK,
                      rng: np.random.Generator | None = None):
    """Bootstrap a statistic of cluster-summed quantities.

    `stat` receives a dict of arrays with shape (..., C) of per-cluster sums, and
    must reduce over the last axis. It is called once on the observed data (arrays
    of shape (C,)) and once, vectorised, on bootstrap draws (arrays of shape (B, C)).
    Returns (estimate, draws).
    """
    rng = rng or np.random.default_rng(12345)
    _, sums = cluster_sums(values, cluster)
    est = float(stat(sums))
    C = len(next(iter(sums.values())))
    idx = moving_block_indices(C, B, block, rng)
    boot = {k: v[idx] for k, v in sums.items()}
    draws = np.asarray(stat(boot), float)
    return est, draws


def _ratio_skill(s):
    return 1.0 - s["cand"].sum(-1) / s["ref"].sum(-1)


@dataclass
class Comparison:
    metric: str
    reference: str
    candidate: str
    n_units: int
    n_clusters: int
    mean_ref: float
    mean_cand: float
    effect: float           # relative skill 1 - cand/ref (positive = candidate better)
    ci_low: float
    ci_high: float
    p_value: float          # one-sided, H0: effect <= 0
    p_adj: float | None = None
    mpe: float | None = None
    decision: str | None = None

    def to_dict(self):
        return asdict(self)


def compare_losses(ref_loss: np.ndarray, cand_loss: np.ndarray, cluster: np.ndarray,
                   metric: str = "", reference: str = "ref", candidate: str = "cand",
                   B: int = DEFAULT_B, block: int = DEFAULT_BLOCK, level: float = 0.95,
                   rng: np.random.Generator | None = None, absolute: bool = False) -> Comparison:
    """Paired comparison of two models' losses on the same units.

    Effect = relative skill 1 - sum(cand)/sum(ref), or the absolute mean difference
    (ref - cand) when `absolute=True` (used for log scores in bits).
    """
    ref_loss = np.asarray(ref_loss, float)
    cand_loss = np.asarray(cand_loss, float)
    assert ref_loss.shape == cand_loss.shape
    vals = {"ref": ref_loss, "cand": cand_loss, "n": np.ones_like(ref_loss)}
    if absolute:
        stat = lambda s: (s["ref"].sum(-1) - s["cand"].sum(-1)) / s["n"].sum(-1)  # noqa: E731
    else:
        stat = _ratio_skill
    est, draws = cluster_bootstrap(vals, cluster, stat, B=B, block=block, rng=rng)
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    p = (1 + np.sum(draws <= 0)) / (len(draws) + 1)
    return Comparison(metric, reference, candidate, len(ref_loss), len(np.unique(cluster)),
                      float(ref_loss.mean()), float(cand_loss.mean()), est, float(lo), float(hi), float(p))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def decide(effect: float, ci_low: float, ci_high: float, mpe: float,
           p_adj: float | None = None, alpha: float = ALPHA) -> str:
    """Pre-registered three-way (plus harm/promising) decision rule."""
    p_ok = True if p_adj is None else (p_adj < alpha)
    if p_ok and ci_low > 0 and effect >= mpe:
        return "SUPPORTED"
    if ci_high < 0:
        return "HARMFUL"
    if ci_high < mpe:
        return "FALSIFIED"
    if effect >= mpe:
        return "PROMISING"
    return "INCONCLUSIVE"


def apply_decisions(comparisons: list[Comparison], mpes: list[float], alpha: float = ALPHA) -> list[Comparison]:
    """Holm-adjust a *pre-registered family* of comparisons, then apply `decide`."""
    adj = holm([c.p_value for c in comparisons])
    for c, pa, m in zip(comparisons, adj, mpes):
        c.p_adj = pa
        c.mpe = m
        c.decision = decide(c.effect, c.ci_low, c.ci_high, m, pa, alpha)
    return comparisons


def stratum_interaction(ref_loss, cand_loss, cluster, in_stratum,
                        B: int = DEFAULT_B, block: int = DEFAULT_BLOCK,
                        rng: np.random.Generator | None = None, level: float = 0.95):
    """Difference in relative skill (candidate vs reference) inside vs outside a stratum.

    Positive = the candidate's advantage is larger inside the stratum. This is the
    H6 test: "does the structural model gain specifically where the world changed?"
    """
    m = np.asarray(in_stratum, bool)
    r = np.asarray(ref_loss, float)
    c = np.asarray(cand_loss, float)
    vals = {"ri": r * m, "ci": c * m, "ro": r * ~m, "co": c * ~m}

    def stat(s):
        si = 1 - s["ci"].sum(-1) / s["ri"].sum(-1)
        so = 1 - s["co"].sum(-1) / s["ro"].sum(-1)
        return si - so

    est, draws = cluster_bootstrap(vals, cluster, stat, B=B, block=block, rng=rng)
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    p = (1 + np.sum(draws <= 0)) / (len(draws) + 1)
    return {"effect": est, "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
            "n_in": int(m.sum()), "n_out": int((~m).sum())}


def spearman_across_variants(event_bits: dict[str, np.ndarray], event_cluster: np.ndarray,
                             downstream_loss: dict[str, np.ndarray], downstream_cluster: np.ndarray,
                             B: int = 500, rng: np.random.Generator | None = None):
    """H5: rank correlation, across model variants, between event-level code length and
    downstream player-stat loss. Bootstrap resamples clusters (days) jointly when the
    two cluster vocabularies coincide; otherwise resamples each independently.
    """
    from scipy.stats import spearmanr

    rng = rng or np.random.default_rng(7)
    names = sorted(event_bits)
    _, es = cluster_sums({n: event_bits[n] for n in names}, event_cluster)
    _, ds = cluster_sums({n: downstream_loss[n] for n in names}, downstream_cluster)
    e_tot = np.array([es[n].sum() for n in names])
    d_tot = np.array([ds[n].sum() for n in names])
    rho = spearmanr(e_tot, d_tot).statistic
    Ce, Cd = len(es[names[0]]), len(ds[names[0]])
    draws = []
    for _ in range(B):
        ie = rng.integers(0, Ce, Ce)
        idd = ie if Ce == Cd else rng.integers(0, Cd, Cd)
        draws.append(spearmanr([es[n][ie].sum() for n in names], [ds[n][idd].sum() for n in names]).statistic)
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return {"rho": float(rho), "ci_low": float(lo), "ci_high": float(hi), "variants": names,
            "event_bits_total": e_tot.tolist(), "downstream_total": d_tot.tolist()}
