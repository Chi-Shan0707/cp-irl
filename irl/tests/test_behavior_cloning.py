import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration, occupancy_of_policy
from irl.behavior_cloning import behavior_cloning_policy


def test_bc_recovers_unanimous_policy():
    rng = np.random.default_rng(0)
    mdp = random_mdp(6, 3, 3, gamma=0.9, rng=rng)
    theta = rng.normal(size=3)
    _, _, pi = value_iteration(mdp, theta)
    # every demonstrator plays the SAME policy -> BC should recover it exactly
    policies = [pi.copy() for _ in range(10)]
    bc_pi = behavior_cloning_policy(mdp, policies)
    assert np.array_equal(bc_pi, pi)


def test_bc_is_majority_vote():
    rng = np.random.default_rng(1)
    mdp = random_mdp(4, 2, 2, gamma=0.9, rng=rng)
    # construct a population where action 0 wins at every state by majority
    policies = [np.zeros(4, dtype=int) for _ in range(6)] + [np.ones(4, dtype=int) for _ in range(4)]
    bc_pi = behavior_cloning_policy(mdp, policies)
    assert np.all(bc_pi == 0)


def test_bc_output_is_valid_policy_shape():
    rng = np.random.default_rng(2)
    mdp = random_mdp(5, 4, 3, gamma=0.9, rng=rng)
    policies = [rng.integers(0, 4, size=5) for _ in range(20)]
    bc_pi = behavior_cloning_policy(mdp, policies)
    assert bc_pi.shape == (5,)
    assert np.all((bc_pi >= 0) & (bc_pi < 4))
    # occupancy_of_policy should not error on the BC output
    d = occupancy_of_policy(mdp, bc_pi)
    assert d.sum() == pytest.approx(1.0, abs=1e-6)
