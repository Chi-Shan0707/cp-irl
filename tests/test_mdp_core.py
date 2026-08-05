"""Phase 0 gate: value iteration and the occupancy-measure LP must agree.

Run: python -m pytest tests/test_mdp_core.py -v   (inside conda env rlenv)
"""
import numpy as np
import pytest

from envs.mdp import (
    TabularMDP,
    value_iteration,
    occupancy_lp,
    policy_value,
    occupancy_of_policy,
    random_mdp,
)


@pytest.mark.parametrize("seed", range(20))
def test_vi_matches_lp_value(seed):
    rng = np.random.default_rng(seed)
    S = rng.integers(3, 8)
    A = rng.integers(2, 4)
    d_feat = rng.integers(1, 5)
    gamma = 0.9
    mdp = random_mdp(S, A, d_feat, gamma, rng)
    theta = rng.normal(size=d_feat)

    V, Q, policy = value_iteration(mdp, theta, tol=1e-12)
    v_vi = float(mdp.mu0 @ V)

    d, v_lp = occupancy_lp(mdp, theta)

    assert v_vi == pytest.approx(v_lp, abs=1e-6), (
        f"seed={seed}: VI value {v_vi} != LP value {v_lp}"
    )


@pytest.mark.parametrize("seed", range(20, 40))
def test_vi_policy_value_matches_direct_evaluation(seed):
    rng = np.random.default_rng(seed)
    S = rng.integers(3, 8)
    A = rng.integers(2, 4)
    d_feat = rng.integers(1, 5)
    gamma = 0.9
    mdp = random_mdp(S, A, d_feat, gamma, rng)
    theta = rng.normal(size=d_feat)

    V, Q, policy = value_iteration(mdp, theta, tol=1e-12)
    v_vi = float(mdp.mu0 @ V)
    v_eval = policy_value(mdp, theta, policy)

    assert v_vi == pytest.approx(v_eval, abs=1e-8)


@pytest.mark.parametrize("seed", range(40, 60))
def test_occupancy_of_optimal_policy_matches_lp_value(seed):
    """The occupancy measure of the VI-optimal policy should attain the LP optimum."""
    rng = np.random.default_rng(seed)
    S = rng.integers(3, 6)
    A = rng.integers(2, 4)
    d_feat = rng.integers(1, 4)
    gamma = 0.9
    mdp = random_mdp(S, A, d_feat, gamma, rng)
    theta = rng.normal(size=d_feat)

    V, Q, policy = value_iteration(mdp, theta, tol=1e-12)
    d_from_policy = occupancy_of_policy(mdp, policy)
    R = mdp.reward(theta)
    v_from_occupancy = float(d_from_policy.flatten() @ R.flatten()) / (1 - gamma)

    _, v_lp = occupancy_lp(mdp, theta)

    assert v_from_occupancy == pytest.approx(v_lp, abs=1e-6)


def test_occupancy_sums_to_one():
    rng = np.random.default_rng(0)
    mdp = random_mdp(5, 3, 2, 0.9, rng)
    theta = rng.normal(size=2)
    d, _ = occupancy_lp(mdp, theta)
    assert d.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(d >= -1e-9)
