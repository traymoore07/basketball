"""Fast subset of the synthetic verification suite (full suite: python -m hoopslab.experiments.synthetic_suite)."""
import pytest

from hoopslab.experiments import synthetic_suite as S


@pytest.mark.slow
@pytest.mark.parametrize("name", ["usable_information", "compression", "role_change_D", "param_uncertainty_G"])
def test_quick_checks(name):
    r = S.CHECKS[name](0)
    assert r["passed"], r
