"""T6 (efficiency / non-vacuousness, plan.md Sec 5 stretch goal): the calibrated
uncertainty set must shrink toward a point as demonstrator-population noise
vanishes -- otherwise the whole approach would be trivially, uselessly conservative
regardless of how informative the data actually is.

A first attempt at this test used envs.mdp.random_mdp's default population-noise
knobs (a multiplicative p ~ U(0.5, 2.0) PER COORDINATE, held fixed, plus a separate
additive eps ~ N(0, noise_std)) and swept noise_std alone. That showed alpha
PLATEAUING at a nonzero floor (~7 degrees) instead of vanishing -- not a bug, but a
modeling artifact: p's spread is itself a substantial, ALWAYS-PRESENT source of
per-coordinate directional noise (matching the CIO paper's own data-generation
convention, theta_hat_i = (theta*_i p_i + eps_i)_+ + eps_0, p_i ~ U[1/2,2]) that
does not vanish just because the additive eps term shrinks. To test the actual
"does the set become non-vacuous as population noise -> 0" claim, ALL noise sources
must shrink together (a single `spread` parameter below), so every demonstrator's
theta_hat converges to theta_star exactly as spread -> 0.
"""
import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration
from irl.point_estimate import classic_irl
from conformal.calibrate import conformal_calibrate


def _gen_population(mdp, theta_star, N, rng, spread):
    policies = []
    for _ in range(N):
        p = 1.0 + rng.uniform(-spread, spread, size=theta_star.shape)
        eps = rng.normal(0, spread, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + 0.01 * spread
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
    return policies


def test_alpha_shrinks_to_near_zero_as_noise_vanishes():
    S, A, d = 6, 3, 3
    spreads = [1.0, 0.5, 0.2, 0.1, 0.05]
    n_seeds = 5

    mean_alphas = []
    for spread in spreads:
        alphas = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
            theta_star = rng.uniform(0.5, 2.0, size=d)
            pols_train = _gen_population(mdp, theta_star, 80, rng, spread)
            pols_val = _gen_population(mdp, theta_star, 80, rng, spread)
            theta_bar = classic_irl(mdp, pols_train)
            alpha, _ = conformal_calibrate(mdp, pols_val, theta_bar, gamma=0.8, n_jobs=4)
            alphas.append(alpha)
        mean_alphas.append(np.mean(alphas))

    # monotonically non-increasing (up to small numerical/sampling noise)
    for i in range(len(mean_alphas) - 1):
        assert mean_alphas[i + 1] <= mean_alphas[i] + 0.05, (
            f"alpha not shrinking: spreads={spreads}, mean_alphas={mean_alphas}"
        )
    # the smallest-noise setting should be near-degenerate (well under 5 degrees)
    assert mean_alphas[-1] < np.radians(5), (
        f"alpha did not shrink to near-zero at the smallest noise level: "
        f"{np.degrees(mean_alphas[-1]):.2f} degrees"
    )
