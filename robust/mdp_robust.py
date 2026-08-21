"""Robust MDP solve (plan.md Sec 4 P3): max_pi min_{theta in C(theta_bar,alpha)}
theta^T Phi^T mu(pi), in occupancy space -- the IRL analogue of CIO's RFO.

Sign convention flips relative to cio/robust.py: CIO's forward problem MINIMIZES cost,
so the robust decision hedges against the WORST-CASE (maximum) cost, h_C(x) =
max_{theta in C} theta^Tx. The MDP forward problem MAXIMIZES reward, so the robust
policy hedges against the WORST-CASE (minimum) reward:

    g_C(x) := min_{theta in C(theta_bar,alpha)} theta^T x

By Lagrangian duality on the inner min (Slater holds for alpha in (0, pi)):

    g_C(x) = max_{lambda >= 0} [lambda * cos(alpha) - ||x - lambda*theta_bar||_2]

so the whole robust MDP

    max_{d in M} min_{theta in C} theta^T (Phi^T d)  =  max_{d in M, lambda>=0} g_C(Phi^T d)

is a joint CONCAVE maximization over (d, lambda) -- M a polytope (the occupancy-measure
flow-conservation constraints, same as envs/mdp.py::occupancy_lp), lambda>=0 a scalar --
i.e. a single convex program (no case-splitting, no nested optimization), solved
directly via cvxpy/CLARABEL. The alpha=0 singleton is handled separately because
the displayed dual supremum need not be attained at finite lambda. This is the P3
tractability argument in plan.md:
reward-only, linearly-parameterized ambiguity that touches only the OBJECTIVE (never
the feasible occupancy set M itself) keeps the whole thing convex regardless of the
MDP's dynamics.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from envs.mdp import TabularMDP, occupancy_lp


def occupancy_to_policy(d_occ: np.ndarray) -> np.ndarray:
    """Convert a state-action occupancy to a stationary randomized policy.

    Unvisited states have no effect on the represented occupancy; we assign a
    uniform action distribution there to keep the returned policy well-defined.
    """
    d_occ = np.asarray(d_occ, dtype=float)
    if d_occ.ndim != 2 or d_occ.shape[1] == 0:
        raise ValueError("d_occ must be a nonempty two-dimensional array")
    if np.min(d_occ) < -1e-8:
        raise ValueError("d_occ contains materially negative entries")
    d_occ = np.maximum(d_occ, 0.0)
    state_mass = d_occ.sum(axis=1, keepdims=True)
    policy = np.full_like(d_occ, 1.0 / d_occ.shape[1])
    np.divide(d_occ, state_mass, out=policy, where=state_mass > 1e-12)
    return policy


def solve_robust_mdp(mdp: TabularMDP, theta_bar: np.ndarray, alpha: float):
    """Returns (d, value): the robust-optimal occupancy measure (S, A) and its
    worst-case discounted value (same normalization as envs.mdp.occupancy_lp:
    value = objective / (1 - gamma))."""
    theta_bar = np.asarray(theta_bar, dtype=float)
    if theta_bar.shape != (mdp.d,):
        raise ValueError(f"theta_bar must have shape ({mdp.d},), got {theta_bar.shape}")
    if not np.isclose(np.linalg.norm(theta_bar), 1.0, atol=1e-7):
        raise ValueError("theta_bar must have unit Euclidean norm")
    if not 0.0 <= alpha <= np.pi:
        raise ValueError(f"alpha must lie in [0, pi], got {alpha}")

    # At alpha=0 the cap is the singleton {theta_bar}.  The dual below has the
    # correct supremum but need not attain it at finite lambda, so solve the
    # point-reward occupancy LP directly.
    if alpha <= 1e-10:
        return occupancy_lp(mdp, theta_bar)

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
    lam = cp.Variable(nonneg=True)

    x = Phi_flat.T @ d_var  # (d,) occupancy-weighted feature vector
    objective = cp.Maximize(lam * np.cos(alpha) - cp.norm(x - lam * theta_bar, 2))
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
        raise RuntimeError(f"robust MDP solve failed: status={prob.status}")

    d_raw = np.asarray(d_var.value).reshape(S, A)
    if np.min(d_raw) < -1e-7:
        raise RuntimeError("robust MDP solver returned a materially negative occupancy")
    d_occ = np.maximum(d_raw, 0.0)
    if np.max(np.abs(A_flow @ d_occ.ravel() - b_flow)) > 1e-5:
        raise RuntimeError("robust MDP solver returned an infeasible occupancy")
    value = float(prob.value) / (1 - gamma)
    return d_occ, value
