"""The closed-form spherical-cap support function must match a cvxpy SOCP oracle."""
import cvxpy as cp
import numpy as np
import pytest

from cio.support_function import support_value, support_maximizer


def _cvxpy_oracle(x, theta_bar, alpha):
    theta = cp.Variable(len(x))
    constraints = [cp.norm(theta, 2) <= 1, theta_bar @ theta >= np.cos(alpha)]
    prob = cp.Problem(cp.Maximize(theta @ x), constraints)
    prob.solve(solver=cp.CLARABEL)
    return prob.value


@pytest.mark.parametrize("seed", range(20))
def test_closed_form_matches_socp_oracle(seed):
    rng = np.random.default_rng(seed)
    d = rng.integers(2, 8)
    theta_bar = rng.normal(size=d)
    theta_bar /= np.linalg.norm(theta_bar)
    alpha = rng.uniform(0.05, np.pi / 2 - 0.05)
    x = rng.normal(size=d) * rng.uniform(0.1, 5)

    closed = support_value(x, theta_bar, alpha)
    oracle = _cvxpy_oracle(x, theta_bar, alpha)

    assert closed == pytest.approx(oracle, abs=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_maximizer_achieves_support_value(seed):
    rng = np.random.default_rng(seed)
    d = rng.integers(2, 6)
    theta_bar = rng.normal(size=d)
    theta_bar /= np.linalg.norm(theta_bar)
    alpha = rng.uniform(0.05, np.pi / 2 - 0.05)
    x = rng.normal(size=d)

    theta_opt = support_maximizer(x, theta_bar, alpha)
    assert np.linalg.norm(theta_opt) == pytest.approx(1.0, abs=1e-6)
    assert theta_bar @ theta_opt >= np.cos(alpha) - 1e-6
    assert theta_opt @ x == pytest.approx(support_value(x, theta_bar, alpha), abs=1e-6)


def test_support_value_is_at_least_projection_onto_theta_bar():
    """Sanity floor: h_C(x) >= x . theta_bar (theta_bar itself is always feasible)."""
    rng = np.random.default_rng(42)
    theta_bar = rng.normal(size=5)
    theta_bar /= np.linalg.norm(theta_bar)
    for _ in range(20):
        x = rng.normal(size=5)
        alpha = rng.uniform(0.05, np.pi / 2 - 0.05)
        assert support_value(x, theta_bar, alpha) >= x @ theta_bar - 1e-8
