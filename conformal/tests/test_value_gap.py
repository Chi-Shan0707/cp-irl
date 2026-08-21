"""Tests for conformal/value_gap.py: Theorem V1 (coverage) and Theorem V2
(shaping invariance), notes/2026-08-09_value_gap_redesign.md.
"""
import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration
from conformal.value_gap import (
    value_gap_score, calibrate_value_gap, lipschitz_bound, ball_radius,
)


def _gen_population(mdp, theta_star, N, rng, noise_std=0.4):
    policies = []
    thetas = []
    for _ in range(N):
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = theta_star + eps
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
        thetas.append(theta_hat)
    return policies, thetas


def test_score_nonnegative_and_zero_for_theta_bar_optimal_policy():
    rng = np.random.default_rng(0)
    mdp = random_mdp(6, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    _, _, pi_star = value_iteration(mdp, theta_bar)

    gap_star = value_gap_score(mdp, pi_star, theta_bar)
    assert gap_star == pytest.approx(0.0, abs=1e-6)

    rng2 = np.random.default_rng(1)
    for _ in range(10):
        pi_other = rng2.integers(0, mdp.A, size=mdp.S)
        gap = value_gap_score(mdp, pi_other, theta_bar)
        assert gap >= -1e-9


@pytest.mark.parametrize("seed", range(10))
def test_coverage_theorem_v1(seed):
    """Empirical check of Theorem V1: P[c_new <= q_gamma] >= gamma (marginal, so we
    check across many independent (train/val/test) draws rather than one trial)."""
    rng = np.random.default_rng(seed)
    mdp = random_mdp(6, 3, 3, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=mdp.d)

    gamma_target = 0.8
    n_trials = 300
    hits = 0
    for _ in range(n_trials):
        theta_bar = theta_star + rng.normal(0, 0.3, size=mdp.d)  # fixed "estimator"
        pols_val, _ = _gen_population(mdp, theta_star, 40, rng)
        q_gamma, _ = calibrate_value_gap(mdp, pols_val, theta_bar, gamma_target)

        pols_new, _ = _gen_population(mdp, theta_star, 1, rng)
        c_new = value_gap_score(mdp, pols_new[0], theta_bar)
        if c_new <= q_gamma + 1e-9:
            hits += 1

    coverage = hits / n_trials
    # split-conformal marginal guarantee: >= gamma, allow finite-sample slack
    assert coverage >= gamma_target - 0.08, f"coverage {coverage} too far below {gamma_target}"


def test_shaping_invariance_theorem_v2():
    """Theorem V2: c_k(theta_bar') == c_k(theta_bar) when theta_bar' differs from
    theta_bar only by a potential-based-shaping direction that preserves the optimal
    policy. Construct such a pair by picking a random potential Phi(s) and checking
    the SAME policy remains theta_bar-optimal under theta_bar + shaping.
    """
    rng = np.random.default_rng(7)
    S, A, d = 5, 3, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=d)

    # Build a shaping feature direction: augment phi with an extra dimension whose
    # value is exactly a potential-based shaping term for a random potential Phi.
    Phi_pot = rng.normal(size=S)
    shaping_feat = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            shaping_feat[s, a] = mdp.gamma * (mdp.P[s, a, :] @ Phi_pot) - Phi_pot[s]

    phi_aug = np.concatenate([mdp.phi, shaping_feat[:, :, None]], axis=2)
    from envs.mdp import TabularMDP
    mdp_aug = TabularMDP(mdp.P, phi_aug, mdp.gamma, mdp.mu0)

    theta_bar_aug = np.concatenate([theta_bar, [0.0]])
    c_scale = 2.5
    theta_bar_aug_shaped = np.concatenate([theta_bar, [c_scale]])

    _, _, pi_base = value_iteration(mdp_aug, theta_bar_aug)
    _, _, pi_shaped = value_iteration(mdp_aug, theta_bar_aug_shaped)
    # shaping must preserve the optimal policy for V2's hypothesis to apply
    assert np.array_equal(pi_base, pi_shaped)

    rng2 = np.random.default_rng(11)
    for _ in range(8):
        pi_k = rng2.integers(0, A, size=S)
        c_base = value_gap_score(mdp_aug, pi_k, theta_bar_aug)
        c_shaped = value_gap_score(mdp_aug, pi_k, theta_bar_aug_shaped)
        assert c_shaped == pytest.approx(c_base, abs=1e-6)


def test_lipschitz_bound_and_ball_radius():
    rng = np.random.default_rng(3)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    nu_bar = lipschitz_bound(mdp)
    assert nu_bar > 0

    rho = ball_radius(2.0, mdp)
    assert rho == pytest.approx(2.0 / nu_bar)
    assert ball_radius(float("inf"), mdp) == float("inf")


def test_sentinel_when_tau_equals_n_plus_1():
    rng = np.random.default_rng(5)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    pols = [rng.integers(0, mdp.A, size=mdp.S) for _ in range(3)]
    # gamma close to 1 with tiny N forces tau = N+1
    q_gamma, _ = calibrate_value_gap(mdp, pols, theta_bar, gamma=0.999)
    assert q_gamma == float("inf")
