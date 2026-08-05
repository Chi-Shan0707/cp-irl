"""Robust MDP solve (plan.md Sec 4 P3): max_pi min_{theta in C(theta_bar,alpha)}
theta^T mu(pi), in occupancy-measure space -- the IRL analogue of cio/robust.py's RFO.

Sign convention flips relative to cio/robust.py: CIO's forward problem MINIMIZES cost,
so the robust decision hedges against the WORST-CASE (maximum) cost, h_C(x) =
max_{theta in C} theta^Tx. The MDP forward problem MAXIMIZES reward, so the robust
policy hedges against the WORST-CASE (minimum) reward:

    g_C(x) := min_{theta in C(theta_bar,alpha)} theta^T x

By Lagrangian duality on the inner min (Slater holds for alpha in (0, pi/2); verified
numerically against a cvxpy oracle before relying on it here, see irl notes):

    g_C(x) = max_{lambda >= 0} [lambda * cos(alpha) - ||x - lambda*theta_bar||_2]

so the whole robust MDP

    max_{d in M} min_{theta in C} theta^T (Phi^T d)  =  max_{d in M, lambda>=0} g_C(Phi^T d)

is a joint CONCAVE maximization over (d, lambda) -- M a polytope (the occupancy-measure
flow-conservation constraints, same as envs/mdp.py::occupancy_lp), lambda>=0 a scalar --
i.e. a single convex program (no case-splitting, no nested optimization), solved
directly via cvxpy/CLARABEL. This is exactly the P3 tractability argument in plan.md:
reward-only, linearly-parameterized ambiguity that touches only the OBJECTIVE (never
the feasible occupancy set M itself) keeps the whole thing convex regardless of the
MDP's dynamics -- see notes/T7_sketch.md for the necessity direction of this claim.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from envs.mdp import TabularMDP


def solve_robust_mdp(mdp: TabularMDP, theta_bar: np.ndarray, alpha: float):
    """Returns (d, value): the robust-optimal occupancy measure (S, A) and its
    worst-case discounted value (same normalization as envs.mdp.occupancy_lp:
    value = objective / (1 - gamma))."""
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

    d_occ = np.maximum(d_var.value, 0.0).reshape(S, A)
    value = float(prob.value) / (1 - gamma)
    return d_occ, value
