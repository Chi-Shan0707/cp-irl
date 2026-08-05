"""Classic IRL point estimation (sub-optimality loss) for the population-of-
demonstrators setting (plan.md Sec 4 P1, unit A): all demonstrators share the same
tabular MDP (P, phi, mu0); demonstrator k is observed acting via policy `policy_k`
(near-)optimally under their own unobserved perceived reward theta_hat_k.

Same derivation pattern as cio/io_pipeline.py::classic_io, transplanted from the
shortest-path LP to the general discounted-MDP occupancy-measure LP:

  - CIO's forward problem MINIMIZES cost: FO(theta,u) = min_x theta^T x.
  - The MDP forward problem MAXIMIZES reward: FO(theta) = max_{d in M} theta^T Phi^T d
    (M = the occupancy-measure polytope, plan.md Sec 4 P3). Sign convention is
    therefore flipped throughout relative to cio/io_pipeline.py.

Sub-optimality loss for demonstrator k: l_k(theta) = FO(theta) - theta^T Phi^T d_k
(best achievable value under theta, minus the value theta assigns to what k actually
did) -- always >= 0, zero iff d_k is theta-optimal.

By LP strong duality (M's defining flow-conservation LP, primal is a MAXIMIZATION):
    max_{d in M} theta^T Phi^T d = min_y { b^T y : A^T y >= Phi theta }
(A, b the same flow-conservation matrix/RHS as envs/mdp.py::occupancy_lp; b = (1-gamma)*mu0,
 SHARED across demonstrators since they act in the same MDP with the same mu0 -- this
 is the P1 unit-A simplification; varying per-demonstrator contexts u_k, as in CIO's
 own setting, is a documented extension, not implemented here).

Substituting and embedding into the outer minimization over theta gives a JOINT LINEAR
PROGRAM over (theta, {y_k}) -- no SOCP needed here (Theta is the simplex, exactly as
in cio/io_pipeline.py's classic_io, for the same reason: the naive relaxed norm-ball
domain lets the loss collapse to the trivial theta=0 solution).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from envs.mdp import TabularMDP, occupancy_of_policy


def _flow_matrix(mdp: TabularMDP) -> np.ndarray:
    """Same (S, S*A) flow-conservation matrix as envs.mdp.occupancy_lp, extracted so
    it can be built once and reused across the joint LP construction."""
    S, A = mdp.S, mdp.A
    n = S * A

    def idx(s, a):
        return s * A + a

    A_eq = np.zeros((S, n))
    for sp in range(S):
        for a in range(A):
            A_eq[sp, idx(sp, a)] += 1.0
        for s in range(S):
            for a in range(A):
                A_eq[sp, idx(s, a)] -= mdp.gamma * mdp.P[s, a, sp]
    return A_eq


def classic_irl(mdp: TabularMDP, policies: list[np.ndarray],
                 normalize_l2: bool = True) -> np.ndarray:
    """Point estimate theta_bar minimizing average sub-optimality loss over a
    population of demonstrators, each observed via a full deterministic policy.
    Returns a d-vector on the simplex (or its L2-normalization if normalize_l2).
    """
    N = len(policies)
    S, A, d = mdp.S, mdp.A, mdp.d
    n_sa = S * A

    A_flow = _flow_matrix(mdp)  # (S, S*A)
    b_flow = (1 - mdp.gamma) * mdp.mu0  # (S,), shared across demonstrators
    Phi_flat = mdp.phi.reshape(n_sa, d)  # (S*A, d)

    d_ks = [occupancy_of_policy(mdp, pi).flatten() for pi in policies]  # each (S*A,)

    # Variables, in order: theta (d), y_1 (S), ..., y_N (S)
    n_vars = d + N * S

    def theta_slice():
        return slice(0, d)

    def y_slice(k):
        return slice(d + k * S, d + (k + 1) * S)

    # Objective: (1/N) sum_k [b^T y_k - theta^T (Phi_flat^T d_k)]
    c = np.zeros(n_vars)
    c[theta_slice()] = -(1.0 / N) * sum(Phi_flat.T @ dk for dk in d_ks)
    for k in range(N):
        c[y_slice(k)] = (1.0 / N) * b_flow

    # Constraints: for each k, Phi_flat @ theta - A_flow^T @ y_k <= 0
    A_ub = np.zeros((N * n_sa, n_vars))
    b_ub = np.zeros(N * n_sa)
    for k in range(N):
        rows = slice(k * n_sa, (k + 1) * n_sa)
        A_ub[rows, theta_slice()] = Phi_flat
        A_ub[rows, y_slice(k)] = -A_flow.T

    # theta on the simplex: theta >= 0, sum(theta) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, theta_slice()] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * d + [(None, None)] * (N * S)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                   method="highs")
    if not res.success:
        raise RuntimeError(f"classic_irl failed to solve: {res.message}")

    theta_hat = np.maximum(res.x[theta_slice()], 0.0)
    if normalize_l2:
        theta_hat = theta_hat / np.linalg.norm(theta_hat)
    return theta_hat
