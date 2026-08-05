"""Correctness of the Woodbury-based resolvent update against direct recomputation."""
import numpy as np
import pytest

from envs.mdp import random_mdp
from irl.feasible_set import resolvent, resolvent_update


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("k_diff", [1, 2, 3])
def test_resolvent_update_matches_direct_recompute(seed, k_diff):
    rng = np.random.default_rng(seed * 100 + k_diff)
    S, A, d = 12, 4, 3
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)

    base_policy = rng.integers(0, A, size=S)
    new_policy = base_policy.copy()
    diff_idx = rng.choice(S, size=k_diff, replace=False)
    for i in diff_idx:
        # force an actual change
        other_actions = [a for a in range(A) if a != base_policy[i]]
        new_policy[i] = rng.choice(other_actions)

    base_W = resolvent(mdp, base_policy)
    W_updated = resolvent_update(mdp, base_policy, base_W, new_policy)
    W_direct = resolvent(mdp, new_policy)

    assert np.allclose(W_updated, W_direct, atol=1e-8), (
        f"seed={seed}, k_diff={k_diff}: Woodbury update disagrees with direct recompute, "
        f"max abs diff = {np.max(np.abs(W_updated - W_direct))}"
    )


def test_resolvent_update_no_diff_returns_same():
    rng = np.random.default_rng(0)
    mdp = random_mdp(8, 3, 2, 0.9, rng)
    policy = rng.integers(0, 3, size=8)
    W = resolvent(mdp, policy)
    W_updated = resolvent_update(mdp, policy, W, policy.copy())
    assert np.allclose(W, W_updated)


def test_resolvent_update_all_states_differ_matches_direct():
    """Stress the k=S extreme (no savings expected, but must still be correct)."""
    rng = np.random.default_rng(7)
    S, A = 8, 3
    mdp = random_mdp(S, A, 2, 0.9, rng)
    base_policy = np.zeros(S, dtype=int)
    new_policy = np.ones(S, dtype=int)
    base_W = resolvent(mdp, base_policy)
    W_updated = resolvent_update(mdp, base_policy, base_W, new_policy)
    W_direct = resolvent(mdp, new_policy)
    assert np.allclose(W_updated, W_direct, atol=1e-6)
