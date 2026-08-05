import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration
from irl.point_estimate import classic_irl


def _population(mdp, theta_star, N, rng, noise_std, eps0=0.0):
    policies = []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, policy = value_iteration(mdp, theta_hat)
        policies.append(policy)
    return policies


def test_classic_irl_recovers_direction_with_low_noise():
    rng = np.random.default_rng(0)
    S, A, d = 6, 3, 4
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=d)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    policies = _population(mdp, theta_star, N=40, rng=rng, noise_std=0.02)
    theta_bar = classic_irl(mdp, policies)

    cos_sim = theta_bar @ theta_star_unit
    assert cos_sim > 0.85, f"cos similarity {cos_sim} too low for near-noiseless data"


def test_classic_irl_does_not_collapse_to_zero():
    rng = np.random.default_rng(1)
    S, A, d = 5, 3, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=d)
    policies = _population(mdp, theta_star, N=20, rng=rng, noise_std=1.0)
    theta_bar = classic_irl(mdp, policies)
    assert np.linalg.norm(theta_bar) == pytest.approx(1.0, abs=1e-4)
    assert theta_bar.std() > 1e-4  # not degenerately uniform either


@pytest.mark.parametrize("seed", range(5))
def test_classic_irl_gives_low_average_suboptimality(seed):
    """theta_bar should make the observed policies close to optimal under itself --
    i.e. actually minimizes the loss it was built to minimize (sanity, not a strong
    correctness proof, but catches gross implementation errors)."""
    rng = np.random.default_rng(seed + 10)
    S, A, d = 5, 3, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=d)
    policies = _population(mdp, theta_star, N=25, rng=rng, noise_std=0.3)
    theta_bar = classic_irl(mdp, policies)

    from envs.mdp import occupancy_of_policy, occupancy_lp
    R = mdp.reward(theta_bar)
    _, opt_val = occupancy_lp(mdp, theta_bar)
    losses = []
    for pi in policies:
        d_pi = occupancy_of_policy(mdp, pi)
        val_pi = float(d_pi.flatten() @ R.flatten()) / (1 - mdp.gamma)
        losses.append(opt_val - val_pi)
    # Not every seed will have all-near-zero losses (a noisy demonstrator can be a
    # legitimate outlier); check the median is small rather than a tight mean bound.
    assert np.median(losses) < 0.1, f"median sub-optimality loss too high: {np.median(losses)}"
    assert all(l >= -1e-6 for l in losses), "sub-optimality loss must be nonnegative"
