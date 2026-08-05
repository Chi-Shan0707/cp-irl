"""Forward shortest-path LP, its dual, and the optimality-condition (ΘOPT) constraints
used both for classic IO (sub-optimality loss) and conformal calibration (Theorem 1
of Conformal Inverse Optimization, transplanted to this network).

Primal (forward problem, FO):  min_x theta^T x  s.t.  A x = b(s,t),  x >= 0
Dual:                           max_y b(s,t)^T y  s.t.  A^T y <= theta   (y free)

For an edge e = (u, v): (A^T y)[e] = y[v] - y[u]. Complementary slackness at an
optimal primal x* (a 0/1 path indicator, since the network is a DAG so the LP has an
integral optimum) says: for e on the path, y[v]-y[u] = theta[e] (tight); for e off the
path, y[v]-y[u] <= theta[e] (slack allowed).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .network import ShortestPathNetwork


def solve_forward(net: ShortestPathNetwork, theta: np.ndarray, s: int, t: int) -> np.ndarray:
    """min_x theta^T x s.t. A x = b(s,t), x >= 0. Returns edge-flow vector x (d,)."""
    b = net.rhs(s, t)
    res = linprog(theta, A_eq=net.A, b_eq=b, bounds=[(0, None)] * net.d, method="highs")
    if not res.success:
        raise RuntimeError(f"forward LP infeasible for (s={s}, t={t}): {res.message}")
    return res.x


def path_edges(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of edges on the path (x_e > tol)."""
    return x > tol


def dual_potentials(net: ShortestPathNetwork, theta: np.ndarray, s: int, t: int) -> np.ndarray:
    """Solve the dual LP directly (node potentials y), for cross-checking strong duality."""
    b = net.rhs(s, t)
    # max b^T y s.t. A^T y <= theta  <=>  min -b^T y s.t. A^T y <= theta, y free
    res = linprog(-b, A_ub=net.A.T, b_ub=theta, bounds=[(None, None)] * net.num_nodes,
                   method="highs")
    if not res.success:
        raise RuntimeError(f"dual LP infeasible: {res.message}")
    return res.x
