"""Robust MDP solve against a EUCLIDEAN-BALL reward uncertainty set, the value-gap
calibrated alternative to the spherical-cap solver in robust/mdp_robust.py. See
notes/2026-08-09_value_gap_redesign.md Sec 3.

    C_ball(theta_bar, rho) := {theta : ||theta - theta_bar||_2 <= rho}

Unlike the spherical cap, this needs no unit-norm constraint (so no non-convex-arc
pitfall) and its support function is closed-form with no auxiliary dual variable:

    min_{theta in C_ball} theta^T x = theta_bar^T x - rho * ||x||_2

so the whole robust MDP

    max_{mu in M} min_{theta in C_ball} theta^T (Phi^T mu) / (1 - beta)
  = max_{mu in M} [theta_bar^T (Phi^T mu) - rho * ||Phi^T mu||_2] / (1 - beta)

is a single concave maximization (linear term minus a convex norm) over the
occupancy-measure polytope M -- solved directly via cvxpy, no case-splitting.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from envs.mdp import TabularMDP


def solve_robust_mdp_ball(mdp: TabularMDP, theta_bar: np.ndarray, rho: float):
    """Returns (d, value): the occupancy measure (S, A) maximizing the worst-case
    value over the Euclidean ball C_ball(theta_bar, rho), and that worst-case value
    (same normalization as envs.mdp.occupancy_lp: value = objective / (1 - gamma)).
    """
    theta_bar = np.asarray(theta_bar, dtype=float)
    if theta_bar.shape != (mdp.d,):
        raise ValueError(f"theta_bar must have shape ({mdp.d},), got {theta_bar.shape}")
    if rho < 0:
        raise ValueError(f"rho must be >= 0, got {rho}")

    S, A, gamma, P, mu0 = mdp.S, mdp.A, mdp.gamma, mdp.P, mdp.mu0
    n = S * A

    def idx(s, a):
        return s * A + a

    A_flow = np.zeros((S, n))
    for sp in range(S):
        for a in range(A):
            A_flow[sp, idx(sp, a)] += 1.0
        for s in range(S):
            for a in range(A):
                A_flow[sp, idx(s, a)] -= gamma * P[s, a, sp]
    b_flow = (1 - gamma) * mu0

    Phi_flat = mdp.phi.reshape(n, mdp.d)  # (S*A, d)

    d_var = cp.Variable(n, nonneg=True)
    x = Phi_flat.T @ d_var  # (d,) occupancy-weighted feature vector

    if not np.isfinite(rho):
        # Infinite radius: worst-case reward is unbounded below unless x = 0.
        # Match solve_robust_mdp's alpha=pi sentinel behavior by returning the
        # (finite) worst-case value achievable when the adversary can pick theta
        # freely on the whole ball -- only x=0 keeps the objective finite, which is
        # almost never useful; callers should treat this as the uninformative case.
        objective = cp.Maximize(theta_bar @ x - 1e6 * cp.norm(x, 2))
    else:
        objective = cp.Maximize(theta_bar @ x - rho * cp.norm(x, 2))
    constraints = [A_flow @ d_var == b_flow]

    prob = cp.Problem(objective, constraints)
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if d_var.value is not None:
            break
    if d_var.value is None:
        raise RuntimeError(f"robust MDP (ball) solve failed: status={prob.status}")

    d_raw = np.asarray(d_var.value).reshape(S, A)
    if np.min(d_raw) < -1e-7:
        raise RuntimeError("robust MDP (ball) solver returned a materially negative occupancy")
    d_occ = np.maximum(d_raw, 0.0)
    if np.max(np.abs(A_flow @ d_occ.ravel() - b_flow)) > 1e-5:
        raise RuntimeError("robust MDP (ball) solver returned an infeasible occupancy")

    x_val = Phi_flat.T @ d_occ.ravel()
    worst_case_obj = float(theta_bar @ x_val - (rho if np.isfinite(rho) else 0.0) * np.linalg.norm(x_val))
    value = worst_case_obj / (1 - gamma)
    return d_occ, value
