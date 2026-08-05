import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration
from irl.bayesian_irl import log_likelihood, bayesian_irl


def test_log_likelihood_favors_generating_theta():
    rng = np.random.default_rng(0)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_true = rng.normal(size=3)
    theta_true /= np.linalg.norm(theta_true)
    _, _, pi = value_iteration(mdp, theta_true)
    policies = [pi.copy() for _ in range(10)]

    theta_wrong = rng.normal(size=3)
    theta_wrong /= np.linalg.norm(theta_wrong)

    ll_true = log_likelihood(mdp, theta_true, policies, temp=5.0)
    ll_wrong = log_likelihood(mdp, theta_wrong, policies, temp=5.0)
    assert ll_true >= ll_wrong


@pytest.mark.parametrize("seed", range(5))
def test_bayesian_irl_recovers_direction_with_low_noise(seed):
    rng = np.random.default_rng(seed)
    S, A, d = 5, 3, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=d)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    policies = []
    for _ in range(20):
        th = theta_star + rng.normal(0, 0.05, size=d)
        _, _, pi = value_iteration(mdp, th)
        policies.append(pi)

    theta_bar = bayesian_irl(mdp, policies, temp=8.0, n_samples=600, burn_in=200,
                              rng=np.random.default_rng(seed + 100))
    cos_sim = theta_bar @ theta_star_unit
    assert cos_sim > 0.7, f"seed={seed}: cos similarity {cos_sim} too low"


def test_bayesian_irl_output_is_unit_norm():
    rng = np.random.default_rng(1)
    mdp = random_mdp(4, 2, 2, gamma=0.9, rng=rng)
    theta_star = rng.normal(size=2)
    _, _, pi = value_iteration(mdp, theta_star)
    policies = [pi.copy() for _ in range(5)]
    theta_bar = bayesian_irl(mdp, policies, n_samples=300, burn_in=100,
                              rng=np.random.default_rng(2))
    assert np.linalg.norm(theta_bar) == pytest.approx(1.0, abs=1e-6)
