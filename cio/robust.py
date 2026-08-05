"""Robust forward problem: min_{x in X(u)} max_{theta in C(theta_bar, alpha)} theta^T x.

The inner max is the support function h_C(x) of the spherical cap C(theta_bar, alpha)
= {theta : ||theta||_2 <= 1, theta_bar^T theta >= cos(alpha)} (relaxed from equality
to <=1, harmless for a maximization). Its closed form
(cio/support_function.py) is piecewise and not directly SOC-representable as a single
expression, and plugging it into a black-box NLP solver (scipy SLSQP) proved
numerically unreliable at the kink between the two pieces (verified empirically:
SLSQP converged to points violating both the FO-comparison and alpha-to-0 sanity
checks in cio/tests/test_robust.py). Instead we use exact Lagrangian duality on the
inner max (a ball-intersect-halfspace linear program, Slater holds for alpha in
(0, pi/2)):

    h_C(x) = max_theta {theta^T x : ||theta||_2 <= 1, theta_bar^T theta >= cos(alpha)}
           = min_{lambda >= 0} ||x + lambda * theta_bar||_2 - lambda * cos(alpha)

(verified numerically against a cvxpy SOCP oracle for h_C itself before relying on
it here). Substituting this into RFO gives a single JOINT SOCP over (x, lambda):

    min_{x, lambda}  ||x + lambda * theta_bar||_2 - lambda * cos(alpha)
    s.t.             x in X(u),  lambda >= 0

solved directly and robustly by cvxpy/CLARABEL — no case-splitting, no NLP restarts.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .network import ShortestPathNetwork


def solve_rfo(net: ShortestPathNetwork, theta_bar: np.ndarray, alpha: float,
              s: int, t: int) -> np.ndarray:
    b = net.rhs(s, t)
    d = net.d

    x = cp.Variable(d)
    lam = cp.Variable(nonneg=True)

    constraints = [net.A @ x == b, x >= 0]
    objective = cp.Minimize(cp.norm(x + lam * theta_bar, 2) - lam * np.cos(alpha))

    prob = cp.Problem(objective, constraints)
    # CLARABEL can numerically fail as alpha -> 0 (the norm term ||x + lambda*theta_bar||
    # and the -lambda*cos(alpha) term nearly cancel, near-degenerate conic problem);
    # fall back to ECOS in that regime.
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if x.value is not None:
            break
    if x.value is None:
        raise RuntimeError(f"RFO solve failed for (s={s}, t={t}): status={prob.status}")
    return np.maximum(x.value, 0.0)
