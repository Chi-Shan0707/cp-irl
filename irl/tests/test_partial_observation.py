import numpy as np
import pytest

from envs.mdp import random_mdp, value_iteration
from irl.feasible_set import c_k_reference, c_k_partial


@pytest.mark.parametrize("seed", range(10))
def test_c_k_partial_matches_reference_when_fully_visited(seed):
    rng = np.random.default_rng(seed)
    S = rng.integers(3, 7)
    A = rng.integers(2, 4)
    d = rng.integers(2, 5)
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=d)
    theta_bar /= np.linalg.norm(theta_bar)
    theta0 = rng.normal(size=d)
    _, _, policy = value_iteration(mdp, theta0)

    full_visited = {s: policy[s] for s in range(S)}

    c_full = c_k_reference(mdp, policy, theta_bar)
    c_partial = c_k_partial(mdp, full_visited, theta_bar)

    assert c_full == pytest.approx(c_partial, abs=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_c_k_partial_is_upper_bound_under_fewer_visited_states(seed):
    """Fewer constraints (partial visitation) -> larger or equal feasible set ->
    c_k can only increase or stay the same, never decrease."""
    rng = np.random.default_rng(seed + 100)
    S = rng.integers(4, 8)
    A = rng.integers(2, 4)
    d = rng.integers(2, 5)
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=d)
    theta_bar /= np.linalg.norm(theta_bar)
    theta0 = rng.normal(size=d)
    _, _, policy = value_iteration(mdp, theta0)

    full_visited = {s: policy[s] for s in range(S)}
    n_partial = max(1, S - 2)
    visited_subset = rng.choice(S, size=n_partial, replace=False)
    partial_visited = {int(s): int(policy[s]) for s in visited_subset}

    c_full = c_k_partial(mdp, full_visited, theta_bar)
    c_partial = c_k_partial(mdp, partial_visited, theta_bar)

    assert c_partial >= c_full - 1e-6
