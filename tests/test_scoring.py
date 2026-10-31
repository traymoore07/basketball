"""Sanity and properness checks for the scoring instruments."""
import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from hoopslab.metrics import scoring as sc

rng = np.random.default_rng(0)
K = 80


def nb_pmf(mean, disp, k=K):
    p = disp / (disp + mean)
    pm = nbinom.pmf(np.arange(k + 1), disp, p)
    return pm / pm.sum()


def test_crps_point_mass_equals_abs_error():
    pmf = np.zeros((3, K + 1)); pmf[:, 20] = 1
    y = np.array([20, 25, 11])
    assert np.allclose(sc.crps_pmf(pmf, y), np.abs(20 - y))


def test_crps_pmf_matches_sample_crps():
    p = nb_pmf(22, 8)
    y = 30
    samples = rng.choice(K + 1, size=200000, p=p)
    assert abs(sc.crps_pmf(p[None], np.array([y]))[0] - sc.crps_samples(samples, y)) < 0.05


@pytest.mark.parametrize("score", ["log", "crps", "rps"])
def test_propriety_true_distribution_wins(score):
    true = nb_pmf(22, 6)
    wrong = [nb_pmf(22, 60), nb_pmf(25, 6), nb_pmf(22, 2), poisson.pmf(np.arange(K + 1), 22)]
    y = rng.choice(K + 1, size=40000, p=true)

    def s(pmf):
        P = np.repeat(pmf[None], len(y), 0)
        if score == "log":
            return sc.log_score(P, y).mean()
        if score == "crps":
            return sc.crps_pmf(P, y).mean()
        edges = [20, 25, 30, 35]
        return sc.rps(sc.bin_pmf(P, edges), sc.bin_outcome(y, edges)).mean()

    st = s(true)
    for w in wrong:
        assert st < s(w)


def test_randomized_pit_uniform_when_calibrated_and_ushaped_when_narrow():
    true = nb_pmf(15, 4)
    y = rng.choice(K + 1, size=30000, p=true)
    pit = sc.randomized_pit(np.repeat(true[None], len(y), 0), y, rng)
    assert sc.pit_ks(pit) < 0.015
    assert abs(sc.pit_coverage(pit, 0.8) - 0.8) < 0.01
    narrow = poisson.pmf(np.arange(K + 1), 15)
    pit2 = sc.randomized_pit(np.repeat(narrow[None], len(y), 0), y, rng)
    assert sc.pit_dispersion_ratio(pit2) > 1.1
    assert sc.pit_coverage(pit2, 0.8) < 0.75


def test_variogram_detects_dependence_but_marginal_crps_does_not():
    S, n = 400, 300
    cov = np.array([[1, 0.8], [0.8, 1]])
    L = np.linalg.cholesky(cov)
    vs_joint, vs_ind = [], []
    for _ in range(n):
        y = L @ rng.standard_normal(2)
        X = rng.standard_normal((S, 2)) @ L.T
        Xi = X.copy(); Xi[:, 1] = rng.permutation(Xi[:, 1])
        vs_joint.append(sc.variogram_score(X, y)); vs_ind.append(sc.variogram_score(Xi, y))
        assert np.isclose(sc.crps_samples(X[:, 1], y[1]), sc.crps_samples(Xi[:, 1], y[1]))
    assert np.mean(vs_joint) < np.mean(vs_ind)


def test_energy_score_prefers_truth():
    S = 300
    es_t, es_w = [], []
    for _ in range(300):
        y = rng.standard_normal(3)
        es_t.append(sc.energy_score(rng.standard_normal((S, 3)), y, rng=rng))
        es_w.append(sc.energy_score(2.0 * rng.standard_normal((S, 3)), y, rng=rng))
    assert np.mean(es_t) < np.mean(es_w)


def test_log_score_is_finite_for_impossible_outcomes():
    pmf = np.zeros((1, K + 1)); pmf[0, 10] = 1
    assert np.isfinite(sc.log_score(pmf, np.array([50]))).all()
