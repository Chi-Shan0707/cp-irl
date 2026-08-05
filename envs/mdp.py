"""Tabular finite MDP core: value iteration and the occupancy-measure LP.

Reward is linear in a feature vector: R_theta(s,a) = theta . phi(s,a). This is the
structure needed throughout the project (feasible reward sets, robust MDP in
occupancy-measure space); see plan.md Sec 4.

Conventions
-----------
- P has shape (S, A, S): P[s, a, s'] = Pr(s' | s, a).
- phi has shape (S, A, d): feature vector per (s, a).
- theta has shape (d,): reward weights, R(s, a) = phi[s, a] @ theta.
- mu0 has shape (S,): initial state distribution.
- gamma in (0, 1): discount factor.
- Occupancy measure d(s, a) is normalized to sum to 1 (it is (1-gamma) times the
  discounted state-action visitation frequency). True discounted value is then
  V = (mu0-weighted return) = d . R / (1 - gamma).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


class TabularMDP:
    def __init__(self, P: np.ndarray, phi: np.ndarray, gamma: float, mu0: np.ndarray):
        S, A, S2 = P.shape
        assert S == S2, "P must be (S, A, S)"
        assert phi.shape[:2] == (S, A), "phi must be (S, A, d)"
        assert mu0.shape == (S,)
        assert np.allclose(P.sum(axis=2), 1.0), "P must be row-stochastic over s'"
        assert np.isclose(mu0.sum(), 1.0)
        assert 0.0 < gamma < 1.0

        self.P = P
        self.phi = phi
        self.gamma = gamma
        self.mu0 = mu0
        self.S, self.A, self.d = S, A, phi.shape[2]

    def reward(self, theta: np.ndarray) -> np.ndarray:
        """Return R(s, a) = phi[s, a] . theta, shape (S, A)."""
        return self.phi @ theta


def value_iteration(mdp: TabularMDP, theta: np.ndarray, tol: float = 1e-10,
                     max_iter: int = 100_000):
    """Standard value iteration on R_theta. Returns (V, Q, policy).

    policy is deterministic, shape (S,), argmax_a Q(s, a) (ties -> smallest index).
    """
    R = mdp.reward(theta)  # (S, A)
    V = np.zeros(mdp.S)
    for _ in range(max_iter):
        # Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') V(s')
        Q = R + mdp.gamma * (mdp.P @ V)  # (S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol * (1 - mdp.gamma):
            V = V_new
            break
        V = V_new
    Q = R + mdp.gamma * (mdp.P @ V)
    policy = np.argmax(Q, axis=1)
    return V, Q, policy


def occupancy_lp(mdp: TabularMDP, theta: np.ndarray):
    """Solve the occupancy-measure LP: max_d  d . R  s.t. flow constraints, d >= 0.

    Returns (d, value) where d has shape (S, A) sums to 1, and
    value = d.flatten() @ R.flatten() / (1 - gamma) is the true discounted return
    from mu0.
    """
    S, A, gamma, P, mu0 = mdp.S, mdp.A, mdp.gamma, mdp.P, mdp.mu0
    R = mdp.reward(theta)  # (S, A)

    n = S * A

    def idx(s, a):
        return s * A + a

    # Flow constraints: for each s', sum_a d(s',a) - gamma * sum_{s,a} P(s,a,s') d(s,a)
    #                    = (1-gamma) mu0(s')
    A_eq = np.zeros((S, n))
    b_eq = (1 - gamma) * mu0
    for sp in range(S):
        for a in range(A):
            A_eq[sp, idx(sp, a)] += 1.0
        for s in range(S):
            for a in range(A):
                A_eq[sp, idx(s, a)] -= gamma * P[s, a, sp]

    c = -R.flatten()  # linprog minimizes; we want to maximize d.R
    bounds = [(0, None)] * n

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"occupancy LP failed: {res.message}")

    d = res.x.reshape(S, A)
    value = float(d.flatten() @ R.flatten()) / (1 - gamma)
    return d, value


def policy_value(mdp: TabularMDP, theta: np.ndarray, policy: np.ndarray) -> float:
    """Evaluate a deterministic policy (shape (S,)) under reward theta via linear solve."""
    S = mdp.S
    R = mdp.reward(theta)
    R_pi = R[np.arange(S), policy]  # (S,)
    P_pi = mdp.P[np.arange(S), policy, :]  # (S, S)
    V_pi = np.linalg.solve(np.eye(S) - mdp.gamma * P_pi, R_pi)
    return float(mdp.mu0 @ V_pi)


def occupancy_of_policy(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    """Occupancy measure d(s,a) (sums to 1) induced by a deterministic policy."""
    S, A = mdp.S, mdp.A
    P_pi = mdp.P[np.arange(S), policy, :]  # (S, S)
    # d_s solves: d_s = (1-gamma) mu0 + gamma * P_pi^T d_s   (state occupancy, sums to 1)
    d_s = (1 - mdp.gamma) * np.linalg.solve(
        np.eye(S) - mdp.gamma * P_pi.T, mdp.mu0
    )
    d = np.zeros((S, A))
    d[np.arange(S), policy] = d_s
    return d


def random_mdp(S: int, A: int, d_feat: int, gamma: float, rng: np.random.Generator,
                sparsity: int | None = None) -> TabularMDP:
    """Generate a random tabular MDP with random features, for testing."""
    if sparsity is None:
        P = rng.dirichlet(np.ones(S), size=(S, A))
    else:
        P = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                support = rng.choice(S, size=min(sparsity, S), replace=False)
                probs = rng.dirichlet(np.ones(len(support)))
                P[s, a, support] = probs
    phi = rng.normal(size=(S, A, d_feat))
    mu0 = rng.dirichlet(np.ones(S))
    return TabularMDP(P, phi, gamma, mu0)
