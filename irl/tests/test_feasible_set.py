"""Correctness of the resolvent-accelerated c_k against the slow joint-(theta,V)
reference, and against ground-truth: theta's that satisfy the linear constraints
must make `policy` the actual value-iteration-optimal policy under r_theta.
"""
import numpy as np
import pytest

from envs.mdp import TabularMDP, random_mdp, value_iteration
from irl.feasible_set import c_k, c_k_reference, q_linear_operator, resolvent


def _random_policy(mdp, rng):
    return rng.integers(0, mdp.A, size=mdp.S)


@pytest.mark.parametrize("seed", range(15))
def test_fast_c_k_matches_reference(seed):
    rng = np.random.default_rng(seed)
    S = rng.integers(3, 7)
    A = rng.integers(2, 4)
    d = rng.integers(2, 5)
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=d)
    theta_bar /= np.linalg.norm(theta_bar)

    # use an actually-optimal policy for some random theta, so the feasible set is
    # nonempty (both formulations should agree on any policy, but a non-degenerate
    # feasible set is a more informative test)
    theta0 = rng.normal(size=d)
    _, _, policy = value_iteration(mdp, theta0)

    fast = c_k(mdp, policy, theta_bar)
    ref = c_k_reference(mdp, policy, theta_bar)

    assert fast == pytest.approx(ref, abs=1e-4), f"seed={seed}: fast={fast} ref={ref}"


@pytest.mark.parametrize("seed", range(10))
def test_maximizer_theta_actually_makes_policy_optimal(seed):
    """The strongest correctness check: solve for the ARGMAX theta (not just the
    value), then run independent value iteration under that theta and confirm it
    recovers exactly `policy`."""
    rng = np.random.default_rng(seed)
    S = rng.integers(3, 6)
    A = rng.integers(2, 4)
    d = rng.integers(2, 4)
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    theta0 = rng.normal(size=d)
    _, _, policy = value_iteration(mdp, theta0)

    import cvxpy as cp
    M_pi = q_linear_operator(mdp, policy)
    theta_var = cp.Variable(d)
    theta_bar = rng.normal(size=d)
    theta_bar /= np.linalg.norm(theta_bar)
    constraints = [cp.norm(theta_var, 2) <= 1]
    for s in range(S):
        a_star = policy[s]
        for a in range(A):
            if a == a_star:
                continue
            constraints.append(M_pi[s, a] @ theta_var <= M_pi[s, a_star] @ theta_var)
    prob = cp.Problem(cp.Maximize(theta_var @ theta_bar), constraints)
    prob.solve(solver=cp.CLARABEL)
    theta_opt = theta_var.value
    assert theta_opt is not None

    # Tie-tolerant check: the argmax-SOCP can land exactly on a Bellman-optimality
    # tie between two actions at a state (both equally optimal under theta_opt), in
    # which case value_iteration's fixed tie-break (smallest index) may pick a
    # DIFFERENT but equally-valid action than `policy`. So the correct check is not
    # "policy_recovered == policy" but "policy(s) attains the max Q(s, .) under
    # theta_opt at every state" -- i.e. policy is *a* valid optimal policy, not
    # necessarily *the* canonical VI one. (Confirmed by direct inspection this is a
    # real tie, not a bug: at the differing state(s), Q(s, policy[s]) exactly equals
    # Q(s, policy_recovered[s]) to solver precision.)
    _, Q_opt, policy_recovered = value_iteration(mdp, theta_opt)
    q_of_policy = Q_opt[np.arange(S), policy]
    q_max = Q_opt.max(axis=1)
    assert np.allclose(q_of_policy, q_max, atol=1e-4), (
        f"seed={seed}: policy is NOT optimal under theta_opt -- "
        f"Q(s,policy[s])={q_of_policy} vs max Q(s,.)={q_max}"
    )


def test_resolvent_matches_direct_linear_solve():
    rng = np.random.default_rng(0)
    mdp = random_mdp(5, 3, 2, 0.9, rng)
    policy = _random_policy(mdp, rng)
    W = resolvent(mdp, policy)
    P_pi = mdp.P[np.arange(mdp.S), policy, :]
    # (I - gamma P_pi) @ W should be I
    lhs = (np.eye(mdp.S) - mdp.gamma * P_pi) @ W
    assert np.allclose(lhs, np.eye(mdp.S), atol=1e-8)


@pytest.mark.parametrize("seed", range(10))
def test_q_linear_operator_matches_direct_policy_evaluation(seed):
    """M_pi[s,a] . theta should equal the Q-value obtained by directly evaluating
    the policy's induced value function (independent of the resolvent machinery)."""
    rng = np.random.default_rng(seed + 50)
    S, A, d = 5, 3, 3
    mdp = random_mdp(S, A, d, 0.9, rng)
    policy = _random_policy(mdp, rng)
    theta = rng.normal(size=d)

    M_pi = q_linear_operator(mdp, policy)
    Q_from_operator = M_pi @ theta  # (S, A)

    # direct: V^pi_theta via linear solve, then Q(s,a) = R(s,a) + gamma P(s,a,:).V
    R = mdp.reward(theta)
    P_pi = mdp.P[np.arange(S), policy, :]
    R_pi = R[np.arange(S), policy]
    V_pi = np.linalg.solve(np.eye(S) - mdp.gamma * P_pi, R_pi)
    Q_direct = R + mdp.gamma * (mdp.P @ V_pi)

    assert np.allclose(Q_from_operator, Q_direct, atol=1e-8)
