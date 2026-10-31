import numpy as np

from hoopslab.metrics import compare as cp

rng = np.random.default_rng(1)


def test_decide_rules():
    assert cp.decide(0.03, 0.01, 0.05, 0.01, p_adj=0.01) == "SUPPORTED"
    assert cp.decide(0.03, 0.01, 0.05, 0.01, p_adj=0.2) == "PROMISING"
    assert cp.decide(0.001, -0.002, 0.004, 0.01) == "FALSIFIED"
    assert cp.decide(-0.03, -0.05, -0.01, 0.01) == "HARMFUL"
    assert cp.decide(0.005, -0.01, 0.02, 0.01) == "INCONCLUSIVE"


def test_holm():
    adj = cp.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_compare_detects_real_improvement_and_not_noise():
    n_days, per_day = 150, 40
    cluster = np.repeat(np.arange(n_days), per_day)
    day_effect = np.repeat(rng.normal(0, 1, n_days), per_day)
    ref = 5 + day_effect + rng.gamma(2, 1, len(cluster))
    better = ref * 0.97
    noise = ref + rng.normal(0, 0.5, len(cluster))
    c1 = cp.compare_losses(ref, better, cluster)
    c2 = cp.compare_losses(ref, noise, cluster)
    assert c1.ci_low > 0.02
    assert c2.ci_low < 0 < c2.ci_high


def test_stratum_interaction():
    n = 6000
    cluster = np.repeat(np.arange(200), 30)
    strat = rng.random(n) < 0.2
    ref = rng.gamma(3, 2, n)
    cand = np.where(strat, ref * 0.9, ref * 1.0)
    r = cp.stratum_interaction(ref, cand, cluster, strat)
    assert r["ci_low"] > 0.05
