"""Robust MDP solve against the DECISION-AWARE (occupancy-difference) uncertainty
set, the geometry-consistent replacement for the angular cap (robust/mdp_robust.py)
and the Euclidean ball (robust/value_ball_robust.py). See
CPIRL_FIRST_PRINCIPLES.md Sec 5.

    C_q(theta_bar) := {theta : ||theta - theta_bar||_D <= q}

is a ball in the span seminorm ||.||_D (conformal/decision_metric.py), which is
UNBOUNDED along the behaviorally-null subspace N (shaping and every other
annihilator direction). Consequently the raw max-min value criterion
max_mu min_{theta in C_q} theta^T Phi^T mu is -infinity: an adversary can always
add an unboundedly large multiple of a null direction. This is not a modeling
bug -- it is the correct signal that max-min VALUE is the wrong criterion once
the uncertainty set is expressed in the metric the problem actually supports.

The fix (Sec 5.2(i)) is to work with occupancy measures RELATIVE to a fixed
reference occupancy mu_ref in M(P) (e.g. the calibration population's mean
occupancy, mean_population_occupancy; NOT the point estimate's own optimal
occupancy, which is optimal for every q and makes the problem degenerate --
paper Sec. 4.4): every null direction contributes an IDENTICAL constant to
<theta, Phi^T mu> and <theta, Phi^T mu_ref>, so it cancels exactly and the
centered objective is finite. The inner minimization then has the closed form

    min_{||v||_D <= q} <theta_bar + v, x> = <theta_bar, x> - (1-gamma) q * gauge_D(x),
    x := Phi^T(mu - mu_ref),

where gauge_D is the Minkowski gauge of the occupancy-difference body
D = Phi^T(M(P) - M(P)) (the dual unit ball of ||.||_D up to scale: the paper's
D_F carries an extra 1/(1-gamma), so with this unscaled D and unscaled x the
penalty radius becomes (1-gamma) q, as in the objective below). gauge_D(x) itself has
an LP characterization -- x/t in D for the smallest t >= 0 such that
x = Phi^T(mu' - nu') for some mu', nu' in t*M(P) -- so the whole robust problem
below is a SINGLE LP jointly over (mu, mu', nu', t): no SOCP, no case-split, no
Lagrange dual variable, cheaper than both robust/mdp_robust.py and
robust/value_ball_robust.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from envs.mdp import TabularMDP


@dataclass(frozen=True)
class SafeRobustResult:
    """Output of the two-stage intersection-to-safety pipeline.

    ``nontrivial`` is the legacy name for ``containment_radius < 1`` only.
    It is a necessary threshold check, not evidence of positive worst-case
    advantage or of a departure from the reference. Check those separately.
    """

    occupancy: np.ndarray
    worst_case_advantage: float
    intersection_radius: float
    containment_radius: float
    nontrivial: bool


def _flow_matrices(mdp: TabularMDP):
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
    return A_flow, b_flow


def mean_population_occupancy(mdp: TabularMDP, policies: list[np.ndarray]) -> np.ndarray:
    """A meaningful, non-degenerate default mu_ref (see the WARNING in
    solve_robust_mdp_decision_aware): the average occupancy across a population of
    demonstrator policies (e.g. the calibration split). Ties the robust decision
    to the natural CP-IRL question "does this policy beat what a random draw from
    the demonstrator population would achieve", the same population this project's
    population-fairness reading (plan.md T2-corollary) is about.
    """
    from envs.mdp import occupancy_of_policy
    occs = np.stack([occupancy_of_policy(mdp, pi) for pi in policies], axis=0)
    return occs.mean(axis=0)


def solve_robust_mdp_decision_aware(mdp: TabularMDP, theta_bar: np.ndarray, q: float,
                                     mu_ref: np.ndarray):
    """Returns (d, value): the occupancy measure (S, A) maximizing the worst-case
    RELATIVE value over C_q(theta_bar), centered at mu_ref, and that worst-case
    relative value (advantage over mu_ref under the worst theta in C_q) in the
    same normalization as envs.mdp.occupancy_lp (value = objective / (1 - gamma)).

    WARNING -- a real degeneracy, not a bug, confirmed in
    conformal/tests and documented in CPIRL_FIRST_PRINCIPLES.md addendum: if
    mu_ref is theta_bar's OWN optimal occupancy (occupancy_lp(mdp, theta_bar)),
    then mu_ref already maximizes <theta_bar, Phi^T mu> over ALL mu in M(P), so
    the raw advantage <theta_bar, Phi^T(mu-mu_ref)> is <= 0 for every mu -- the
    optimal solution is mu = mu_ref for EVERY q >= 0, with value == 0 identically.
    Centering at the point estimate's own optimum makes this criterion vacuous:
    it never recommends departing from the point estimate. mu_ref must be a
    genuinely different baseline for this criterion to do anything -- e.g.
    mean_population_occupancy (below), or a fixed safe/incumbent policy.

    q = +inf (uninformative sentinel, mirroring solve_robust_mdp_ball's alpha=pi
    / rho=inf handling): only mu with Phi^T(mu-mu_ref) == 0 keeps the centered
    objective finite, so we use a large finite penalty rather than literal
    infinity to keep the LP well-posed; treat the result as uninformative, as
    with the other solvers.
    """
    theta_bar = np.asarray(theta_bar, dtype=float)
    if theta_bar.shape != (mdp.d,):
        raise ValueError(f"theta_bar must have shape ({mdp.d},), got {theta_bar.shape}")
    if q < 0:
        raise ValueError(f"q must be >= 0, got {q}")

    S, A, gamma = mdp.S, mdp.A, mdp.gamma
    n = S * A
    A_flow, b_flow = _flow_matrices(mdp)
    Phi_flat = mdp.phi.reshape(n, mdp.d)  # (S*A, d)

    mu_ref = np.asarray(mu_ref, dtype=float).reshape(n)
    if abs(mu_ref.sum() - 1.0) > 1e-5 or np.min(mu_ref) < -1e-7:
        raise ValueError("mu_ref must be a valid occupancy measure (nonneg, sums to 1)")

    q_eff = q if np.isfinite(q) else 1e6

    mu = cp.Variable(n, nonneg=True)
    mu_p = cp.Variable(n, nonneg=True)
    nu_p = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)

    x = Phi_flat.T @ (mu - mu_ref)  # (d,) relative occupancy-weighted feature vector
    # Objective is the RELATIVE (centered) criterion itself, <theta_bar, x> - q*gauge(x),
    # not the raw <theta_bar, Phi^T mu> (which differs from it only by the mu-independent
    # constant <theta_bar, Phi^T mu_ref>, so the argmax mu is identical either way, but
    # reporting the centered value directly is the semantically meaningful quantity here).
    # q is calibrated in discounted-return units because ||.||_D includes the
    # factor 1/(1-gamma).  Here x uses raw normalized-occupancy features, so the
    # raw objective must use (1-gamma)*q; dividing prob.value below then recovers
    # <theta_bar,x>/(1-gamma) - q*gauge_D(x).
    objective = cp.Maximize(theta_bar @ x - (1 - gamma) * q_eff * t)
    constraints = [
        A_flow @ mu == b_flow,
        A_flow @ mu_p == t * b_flow,
        A_flow @ nu_p == t * b_flow,
        Phi_flat.T @ (mu_p - nu_p) == x,
    ]

    prob = cp.Problem(objective, constraints)
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if mu.value is not None:
            break
    if mu.value is None:
        raise RuntimeError(f"robust MDP (decision-aware) solve failed: status={prob.status}")

    d_raw = np.asarray(mu.value).reshape(S, A)
    if np.min(d_raw) < -1e-7:
        raise RuntimeError("robust MDP (decision-aware) solver returned a materially negative occupancy")
    d_occ = np.maximum(d_raw, 0.0)
    if np.max(np.abs(A_flow @ d_occ.ravel() - b_flow)) > 1e-5:
        raise RuntimeError("robust MDP (decision-aware) solver returned an infeasible occupancy")

    value = float(prob.value) / (1 - gamma)
    return d_occ, value


def solve_safe_robust_mdp(
    mdp: TabularMDP,
    theta_bar: np.ndarray,
    intersection_radius: float,
    fiber_diameter: float,
    mu_ref: np.ndarray,
    *,
    normalized_fiber_score: bool = False,
) -> SafeRobustResult:
    """Deploy with the *containment* radius implied by inverse-fiber width.

    This closes the two distinct stages that must not be conflated:

    1. split conformal calibrates an intersection radius ``q``;
    2. a certified projective diameter ``eta`` of every normalized Bellman
       fiber inflates ``q`` into a latent-reward containment radius ``R``.

    With the ideal normalized-fiber score, ``R=min(2,q+eta)``.  With the
    tractable cone LP score used by ``calibrate_normalized_decision_aware``,
    ``R=min(2,2q+eta)`` for ``q<1`` and ``R=2`` otherwise.  The center is
    normalized intrinsically to unit occupancy span before robust deployment.

    If the certified latent reward ray lies in the resulting ball, robust
    optimality ensures its value is no worse than ``mu_ref``.  Moreover, a
    unit-span translated seminorm ball with ``R>=1`` cannot yield strictly
    positive worst-case advantage: the reference attains zero and is optimal.
    We return it directly in that mathematically degenerate regime instead of
    asking a numerical solver to rediscover the tie.
    """
    from conformal.decision_metric import (
        containment_radius_from_fiber_width,
        normalize_span,
    )

    q = float(intersection_radius)
    theta_bar_unit = normalize_span(mdp, theta_bar)
    radius = containment_radius_from_fiber_width(
        q,
        fiber_diameter,
        normalized_fiber_score=normalized_fiber_score,
    )

    mu_ref_arr = np.asarray(mu_ref, dtype=float)
    if mu_ref_arr.shape != (mdp.S, mdp.A):
        try:
            mu_ref_arr = mu_ref_arr.reshape(mdp.S, mdp.A)
        except ValueError as exc:
            raise ValueError(
                f"mu_ref must have shape ({mdp.S}, {mdp.A})"
            ) from exc
    if abs(float(mu_ref_arr.sum()) - 1.0) > 1e-5 or np.min(mu_ref_arr) < -1e-7:
        raise ValueError("mu_ref must be a valid occupancy measure (nonnegative, sums to 1)")
    A_flow, b_flow = _flow_matrices(mdp)
    if np.max(np.abs(A_flow @ mu_ref_arr.ravel() - b_flow)) > 1e-5:
        raise ValueError("mu_ref must satisfy the discounted occupancy flow constraints")

    if radius >= 1.0 - 1e-10:
        return SafeRobustResult(
            occupancy=mu_ref_arr.copy(),
            worst_case_advantage=0.0,
            intersection_radius=q,
            containment_radius=radius,
            nontrivial=False,
        )

    occupancy, value = solve_robust_mdp_decision_aware(
        mdp, theta_bar_unit, radius, mu_ref_arr
    )
    # mu_ref is feasible and obtains zero, so a materially negative optimum is
    # a solver failure rather than a property of the model.
    if value < -1e-5:
        raise RuntimeError(
            "safe robust solve returned negative worst-case advantage "
            f"{value:.6g}; the reference guarantees value zero"
        )
    return SafeRobustResult(
        occupancy=occupancy,
        worst_case_advantage=max(float(value), 0.0),
        intersection_radius=q,
        containment_radius=radius,
        nontrivial=True,
    )
