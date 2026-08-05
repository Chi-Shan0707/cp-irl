import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration, occupancy_of_policy
from irl.maxent import soft_value_iteration, occupancy_of_stochastic_policy, maxent_irl


def test_soft_policy_is_valid_distribution():
    rng = np.random.default_rng(0)
    mdp = random_mdp(6, 3, 3, gamma=0.9, rng=rng)
    theta = rng.normal(size=3)
    _, _, soft_policy = soft_value_iteration(mdp, theta)
    assert np.allclose(soft_policy.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(soft_policy >= 0)


def test_occupancy_of_stochastic_policy_matches_deterministic_case():
    """A one-hot 'stochastic' policy should exactly match occupancy_of_policy."""
    rng = np.random.default_rng(1)
    mdp = random_mdp(5, 3, 2, gamma=0.9, rng=rng)
    theta = rng.normal(size=2)
    _, _, pi = value_iteration(mdp, theta)

    soft_policy = np.zeros((mdp.S, mdp.A))
    soft_policy[np.arange(mdp.S), pi] = 1.0

    d_soft = occupancy_of_stochastic_policy(mdp, soft_policy)
    d_det = occupancy_of_policy(mdp, pi)
    assert np.allclose(d_soft, d_det, atol=1e-8)


def test_occupancy_of_stochastic_policy_sums_to_one():
    rng = np.random.default_rng(2)
    mdp = random_mdp(5, 3, 2, gamma=0.9, rng=rng)
    theta = rng.normal(size=2)
    _, _, soft_policy = soft_value_iteration(mdp, theta)
    d = occupancy_of_stochastic_policy(mdp, soft_policy)
    assert d.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_maxent_irl_matches_feature_expectations_at_convergence(seed):
    """The defining property of MaxEnt IRL: at the fitted theta, the model's
    expected feature counts should match the empirical target -- this is the
    gradient=0 condition, a strong sanity check independent of theta_star recovery."""
    rng = np.random.default_rng(seed)
    S, A, d = 5, 3, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_gen = rng.normal(size=d)
    policies = []
    for _ in range(15):
        th = theta_gen + rng.normal(0, 0.1, size=d)
        _, _, pi = value_iteration(mdp, th)
        policies.append(pi)

    theta_fit = maxent_irl(mdp, policies, lr=0.5, n_iters=300, normalize_l2=False)

    n_sa = S * A
    Phi_flat = mdp.phi.reshape(n_sa, d)
    target_feat = np.mean(
        [Phi_flat.T @ occupancy_of_policy(mdp, pi).flatten() for pi in policies], axis=0)
    _, _, soft_policy = soft_value_iteration(mdp, theta_fit)
    model_feat = Phi_flat.T @ occupancy_of_stochastic_policy(mdp, soft_policy).flatten()

    assert np.allclose(model_feat, target_feat, atol=0.05), (
        f"seed={seed}: feature mismatch at convergence: "
        f"model={model_feat}, target={target_feat}"
    )
