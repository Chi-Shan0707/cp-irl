"""Sanity checks for the robust MDP solve. Note the sign convention (opposite of
cio/tests/test_robust.py): this is a REWARD-maximization robust problem, so a bigger
uncertainty set (larger alpha) can only make the worst-case value SMALLER (more
conservative), and alpha->0 should recover the plain forward-optimal value.
"""
import numpy as np
import pytest

from envs.mdp import random_mdp, occupancy_lp
from robust.mdp_robust import solve_robust_mdp


@pytest.mark.parametrize("seed", range(8))
def test_robust_value_shrinks_as_alpha_grows(seed):
    rng = np.random.default_rng(seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)

    _, val_small = solve_robust_mdp(mdp, theta_bar, alpha=0.1)
    _, val_big = solve_robust_mdp(mdp, theta_bar, alpha=1.0)

    assert val_big <= val_small + 1e-4


def test_robust_value_reduces_to_plain_fo_as_alpha_to_zero():
    rng = np.random.default_rng(3)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)

    _, val_plain = occupancy_lp(mdp, theta_bar)
    _, val_robust = solve_robust_mdp(mdp, theta_bar, alpha=1e-4)

    assert val_robust == pytest.approx(val_plain, abs=1e-2)


@pytest.mark.parametrize("seed", range(8))
def test_robust_value_never_exceeds_plain_fo_value(seed):
    """Hedging against worst-case reward can only ever be <= the value under the
    single point estimate theta_bar itself (theta_bar is always in the cap)."""
    rng = np.random.default_rng(seed + 20)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)
    alpha = rng.uniform(0.1, np.pi / 2 - 0.1)

    _, val_plain = occupancy_lp(mdp, theta_bar)
    _, val_robust = solve_robust_mdp(mdp, theta_bar, alpha)

    assert val_robust <= val_plain + 1e-4


def test_robust_occupancy_is_feasible():
    rng = np.random.default_rng(5)
    mdp = random_mdp(6, 3, 2, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)
    d_occ, _ = solve_robust_mdp(mdp, theta_bar, alpha=0.5)
    assert d_occ.sum() == pytest.approx(1.0, abs=1e-4)
    assert np.all(d_occ >= -1e-8)
