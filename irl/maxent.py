"""Maximum-entropy IRL (Ziebart et al., 2008) point estimator, for the
population-of-demonstrators setting (plan.md P1 unit A) -- one of the baselines
plan.md's Phase 5 list calls for alongside the sub-optimality-loss LP already
implemented in irl/point_estimate.py.

Standard soft-value-iteration MaxEnt IRL, adapted to fit a single theta against the
AVERAGE occupancy measure across the training population (rather than a single
demonstrator's trajectory, matching this project's per-demonstrator-policy data
model established in Phase 2).
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP, occupancy_of_policy


def soft_value_iteration(mdp: TabularMDP, theta: np.ndarray, tol: float = 1e-10,
                          max_iter: int = 10_000):
    """Soft (log-sum-exp) Bellman backup. Returns (V, Q, soft_policy) where
    soft_policy[s, a] = exp(Q(s,a) - V(s)), a valid distribution over actions.
    """
    R = mdp.reward(theta)  # (S, A)
    V = np.zeros(mdp.S)
    for _ in range(max_iter):
        Q = R + mdp.gamma * (mdp.P @ V)  # (S, A)
        # log-sum-exp with a max-subtraction for numerical stability
        m = Q.max(axis=1, keepdims=True)
        V_new = (m[:, 0] + np.log(np.exp(Q - m).sum(axis=1)))
        if np.max(np.abs(V_new - V)) < tol * (1 - mdp.gamma):
            V = V_new
            break
        V = V_new
    Q = R + mdp.gamma * (mdp.P @ V)
    m = Q.max(axis=1, keepdims=True)
    soft_policy = np.exp(Q - m - np.log(np.exp(Q - m).sum(axis=1, keepdims=True)))
    return V, Q, soft_policy


def occupancy_of_stochastic_policy(mdp: TabularMDP, soft_policy: np.ndarray) -> np.ndarray:
    """Occupancy measure d(s,a) (sums to 1) induced by a STOCHASTIC policy
    soft_policy[s,a] = Pr(a|s). Generalizes envs.mdp.occupancy_of_policy (which only
    accepts deterministic policies) to soft/Boltzmann policies from MaxEnt IRL.
    """
    S, A = mdp.S, mdp.A
    # P_pi[s, s'] = sum_a soft_policy[s,a] * P[s,a,s']
    P_pi = np.einsum("sa,sat->st", soft_policy, mdp.P)  # (S, S)
    d_s = (1 - mdp.gamma) * np.linalg.solve(np.eye(S) - mdp.gamma * P_pi.T, mdp.mu0)
    return soft_policy * d_s[:, None]  # (S, A)


def maxent_irl(mdp: TabularMDP, policies: list[np.ndarray], lr: float = 0.5,
                n_iters: int = 300, normalize_l2: bool = True) -> np.ndarray:
    """MaxEnt IRL point estimate via gradient ascent on the log-likelihood, using
    soft value iteration for the model's expected feature counts. Target is the
    average occupancy-weighted feature vector across the training population.
    """
    d = mdp.d
    n_sa = mdp.S * mdp.A
    Phi_flat = mdp.phi.reshape(n_sa, d)

    target_feat = np.mean(
        [Phi_flat.T @ occupancy_of_policy(mdp, pi).flatten() for pi in policies],
        axis=0,
    )

    theta = np.ones(d) / np.sqrt(d)  # neutral init, avoids the theta=0 pathology
    for _ in range(n_iters):
        _, _, soft_policy = soft_value_iteration(mdp, theta)
        d_soft = occupancy_of_stochastic_policy(mdp, soft_policy)
        model_feat = Phi_flat.T @ d_soft.flatten()
        grad = target_feat - model_feat
        theta = theta + lr * grad

    if normalize_l2:
        norm = np.linalg.norm(theta)
        if norm > 1e-12:
            theta = theta / norm
    return theta
