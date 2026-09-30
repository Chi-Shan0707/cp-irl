"""Decision-aware reward metric for CP-IRL: the occupancy-difference span
seminorm ||v||_D, its use as a conformal nonconformity score (distance to the
Bellman-feasible cone), and split-conformal calibration on that score. See
CPIRL_FIRST_PRINCIPLES.md Secs 2, 4 for the derivation.

Core identity (Sec 2.2): for the occupancy polytope M(P) (envs/mdp.py's flow
polytope), the support-function span of the linear functional v is

    ||v||_D := sup_{mu,nu in M(P)} <v, Phi^T(mu - nu)> / (1 - gamma)
             = V*_v(mu0) + V*_{-v}(mu0)

i.e. two value_iteration calls -- no LP needed for the seminorm alone. It is
zero exactly on the annihilator N (shaping AND every other behaviorally-null
direction, not just potential shaping), invariant under feature reparametrization
theta -> A^{-T} theta for any invertible A, and satisfies Reg_r(pi_{r'}) <=
||r - r'||_D with constant 1 -- a strictly tighter and coordinate-free
replacement for conformal/value_gap.py's Lipschitz-constant bound
Reg <= 2*nu_bar*||theta-theta'||_2.

The main two-stage API first normalizes the fitted reward to unit occupancy span
and then uses dist_D(theta_bar_unit, K(pi_k)). This score is dimensionless and
positive-scale invariant while preserving the exact LP/intersection identity.
`containment_radius_from_fiber_width` then records the separate identification
assumption needed to turn intersection into latent reward-ray containment.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from envs.mdp import TabularMDP, value_iteration
from irl.feasible_set import q_linear_operator


def span_seminorm(mdp: TabularMDP, v: np.ndarray) -> float:
    """||v||_D = V*_v(mu0) + V*_{-v}(mu0), via two value_iteration solves.

    Equal to max_{mu in M(P)} <v,Phi^T mu>/(1-gamma) - min_{mu in M(P)}
    <v,Phi^T mu>/(1-gamma): the range of the linear functional v over the
    occupancy polytope. Zero iff v annihilates every occupancy difference (the
    behaviorally-null subspace N, which contains and can strictly exceed the
    potential-shaping subspace S_pot -- CPIRL_FIRST_PRINCIPLES.md Sec 1.3).
    """
    v = np.asarray(v, dtype=float)
    V_pos, _, _ = value_iteration(mdp, v)
    V_neg, _, _ = value_iteration(mdp, -v)
    return float(mdp.mu0 @ V_pos) + float(mdp.mu0 @ V_neg)


def normalize_span(mdp: TabularMDP, theta: np.ndarray, target: float = 1.0,
                   atol: float = 1e-10) -> np.ndarray:
    """Return the positive-scale representative with occupancy span ``target``.

    Exact optimal-policy data do not identify positive reward scale.  Using the
    occupancy span as the numeraire is intrinsic to the MDP and remains exactly
    invariant under invertible feature reparameterizations.  This is preferable
    to Euclidean normalization, whose representative changes when the feature
    coordinates are stretched.

    Behaviorally-null rewards have zero span and no projective representative;
    callers must handle that genuinely degenerate case explicitly.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (mdp.d,):
        raise ValueError(f"theta must have shape ({mdp.d},), got {theta.shape}")
    if not np.isfinite(target) or target <= 0:
        raise ValueError(f"target must be finite and positive, got {target}")
    scale = span_seminorm(mdp, theta)
    if scale <= atol:
        raise ValueError(
            "cannot span-normalize a behaviorally-null reward "
            f"(occupancy span {scale:.3e})"
        )
    return theta * (target / scale)


def span_match_reward(mdp: TabularMDP, theta: np.ndarray,
                      reference: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Positive-rescale ``theta`` to have the same span as ``reference``.

    The returned representative is invariant to the original positive scale of
    ``theta`` and equivariant under a common positive scaling of ``reference``.
    It is the representation-invariant replacement for Euclidean scale matching
    in the containment experiments.
    """
    target = span_seminorm(mdp, np.asarray(reference, dtype=float))
    if target <= atol:
        raise ValueError(
            "reference is behaviorally null and cannot define a reward scale"
        )
    return normalize_span(mdp, theta, target=target, atol=atol)


def projective_span_distance(mdp: TabularMDP, theta: np.ndarray,
                             theta_prime: np.ndarray,
                             atol: float = 1e-10) -> float:
    """Distance between positive reward rays in normalized policy-gap space.

    For non-null rewards this computes

        || theta/||theta||_D - theta'/||theta'||_D ||_D.

    It is unchanged by independent positive rescaling, additive behaviorally-null
    directions, and invertible feature reparameterizations.  Its range is [0, 2]
    and it upper-bounds normalized regret with constant one.  The function is a
    pseudometric in ambient coordinates and a metric on positive reward rays
    after additive-null directions are identified.
    """
    theta_unit = normalize_span(mdp, theta, atol=atol)
    theta_prime_unit = normalize_span(mdp, theta_prime, atol=atol)
    return span_seminorm(mdp, theta_unit - theta_prime_unit)


def _span_seminorm_epigraph(mdp: TabularMDP, v_expr: cp.Expression):
    """cvxpy epigraph of ||v||_D as a function of an affine cvxpy expression
    v_expr (shape (d,)): returns (objective_expr, constraints) with
    objective_expr == ||v||_D at the optimum. This is the LP-dual form of the
    two value_iteration calls in span_seminorm, needed because distance_to_cone
    below folds ||theta - theta_bar||_D into a LARGER joint LP where theta is
    itself a decision variable (so span_seminorm's numeric v is unavailable).
    """
    S, A = mdp.S, mdp.A
    Phi = mdp.phi.reshape(S * A, mdp.d)
    Pflat = mdp.P.reshape(S * A, S)
    rep = np.repeat(np.eye(S), A, axis=0)  # rep[s*A+a, s] = 1, matches Phi/Pflat row order

    V = cp.Variable(S)
    W = cp.Variable(S)
    Rv = Phi @ v_expr  # (S*A,)
    constraints = [
        Rv + mdp.gamma * (Pflat @ V) <= rep @ V,
        -Rv + mdp.gamma * (Pflat @ W) <= rep @ W,
    ]
    return mdp.mu0 @ (V + W), constraints


def distance_to_cone(mdp: TabularMDP, policy: np.ndarray, theta_bar: np.ndarray,
                      theta_bound: float = 1e3, solver=cp.CLARABEL):
    """s_k := inf_{theta in Theta-hat(pi_k)} ||theta - theta_bar||_D, the exact
    decision-aware nonconformity score (CPIRL_FIRST_PRINCIPLES.md Sec 4.1).

    Solved as ONE joint LP over (theta, V, W): Theta-hat(pi_k) is the polyhedral
    cone of reward vectors under which pi_k has no profitable one-step deviation
    (irl/feasible_set.py::q_linear_operator gives its linear constraints via the
    Bellman resolvent), and the objective is the epigraph above. theta_bound is
    a generous safety cap on ||theta||_inf, purely to keep the solver from
    reporting spurious unboundedness along a recession direction of the cone
    that lies in the annihilator (where ||theta-theta_bar||_D is exactly flat,
    not decreasing -- so the true infimum is always attained at some finite
    ||theta||, Sec 4.3); an error is raised if the cap actually binds.

    Returns (s_k, theta_star): the achieved distance and a minimizing theta.
    """
    M_pi = q_linear_operator(mdp, policy)
    theta = cp.Variable(mdp.d)
    obj_expr, epi_cons = _span_seminorm_epigraph(mdp, theta - theta_bar)

    cone_cons = []
    for s in range(mdp.S):
        a_star = policy[s]
        q_star = M_pi[s, a_star] @ theta
        for a in range(mdp.A):
            if a == a_star:
                continue
            cone_cons.append(M_pi[s, a] @ theta <= q_star)

    bound_cons = [cp.norm(theta, "inf") <= theta_bound]

    prob = cp.Problem(cp.Minimize(obj_expr), epi_cons + cone_cons + bound_cons)
    last_err = None
    for slv in (solver, cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=slv)
        except cp.error.SolverError as e:
            last_err = e
            continue
        if prob.value is not None and np.isfinite(prob.value):
            break
    if prob.value is None or not np.isfinite(prob.value):
        raise RuntimeError(
            f"distance_to_cone: solve failed, status={prob.status}"
        ) from last_err

    theta_star = np.asarray(theta.value, dtype=float)
    if np.max(np.abs(theta_star)) > 0.99 * theta_bound:
        raise RuntimeError(
            "distance_to_cone: theta_bound safety cap is active; increase "
            "theta_bound (the reported distance may be inaccurate)"
        )
    return max(float(prob.value), 0.0), theta_star


def calibrate_decision_aware(mdp: TabularMDP, policies_val: list[np.ndarray],
                              theta_bar: np.ndarray, gamma: float,
                              theta_bound: float = 1e3):
    """Split-conformal calibration on the distance_to_cone score -- the exact
    analogue of conformal/value_gap.py::calibrate_value_gap, but with s_k in
    place of the value-gap g_k. Because s_k IS ||theta-theta_bar||_D minimized
    over the feasible cone, s_k <= q  <=>  Theta-hat(pi_k) intersect
    C_q(theta_bar) != empty  holds EXACTLY, with C_q(theta_bar) := {theta :
    ||theta-theta_bar||_D <= q} (CPIRL_FIRST_PRINCIPLES.md Sec 4.2) -- no
    Lipschitz-bound reversal like value_gap.py::ball_radius is needed.

    Returns (q_gamma, s_ks).
    """
    N = len(policies_val)
    if N == 0:
        raise ValueError("policies_val must contain at least one policy")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")

    s_ks = np.array([
        distance_to_cone(mdp, pi, theta_bar, theta_bound=theta_bound)[0]
        for pi in policies_val
    ])

    tau = int(np.ceil(gamma * (N + 1)))
    if tau == N + 1:
        q_gamma = float("inf")
    else:
        q_gamma = float(np.sort(s_ks)[tau - 1])
    return q_gamma, s_ks


def normalized_distance_to_cone(
    mdp: TabularMDP,
    policy: np.ndarray,
    theta_bar: np.ndarray,
    theta_bound: float = 1e3,
    solver=cp.CLARABEL,
):
    """Dimensionless cone score ``dist_D(theta_bar, K(pi))/||theta_bar||_D``.

    Because ``K(pi)`` is a cone saturated along the additive-null subspace, this
    equals the ordinary distance from the unit-span representative of
    ``theta_bar`` to ``K(pi)``.  It lies in [0, 1], remains a single LP, and is
    invariant under ``theta_bar -> c theta_bar + a`` for ``c>0`` and
    ``a in N``.  It is deliberately *not* the generally nonconvex distance to
    the unit-span slice of the cone.

    Returns ``(score, witness, theta_bar_unit)``.
    """
    theta_bar_unit = normalize_span(mdp, theta_bar)
    score, witness = distance_to_cone(
        mdp, policy, theta_bar_unit, theta_bound=theta_bound, solver=solver
    )
    # The zero reward belongs to every Bellman cone, hence the exact score is at
    # most ||theta_bar_unit||_D = 1.  Allow solver tolerance only.
    if score > 1.0 + 1e-5:
        raise RuntimeError(
            f"normalized cone score must lie in [0,1], got {score:.6g}"
        )
    return min(max(float(score), 0.0), 1.0), witness, theta_bar_unit


def calibrate_normalized_decision_aware(
    mdp: TabularMDP,
    policies_val: list[np.ndarray],
    theta_bar: np.ndarray,
    gamma: float,
    theta_bound: float = 1e3,
):
    """Split-conformal calibration of the dimensionless cone score.

    Returns ``(q_gamma, scores, theta_bar_unit)``.  The quantile has no reward
    units and the associated set is

        {theta: ||theta-theta_bar_unit||_D <= q_gamma}.

    This is the scale-fixed presentation of the existing decision-aware method;
    it preserves exact feasible-cone intersection coverage and LP tractability.
    """
    N = len(policies_val)
    if N == 0:
        raise ValueError("policies_val must contain at least one policy")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")

    theta_bar_unit = normalize_span(mdp, theta_bar)
    scores = np.array([
        distance_to_cone(
            mdp, pi, theta_bar_unit, theta_bound=theta_bound
        )[0]
        for pi in policies_val
    ])
    if np.any(scores > 1.0 + 1e-5):
        raise RuntimeError(
            "normalized cone scores must lie in [0,1]; "
            f"largest value was {float(np.max(scores)):.6g}"
        )
    scores = np.clip(scores, 0.0, 1.0)
    tau = int(np.ceil(gamma * (N + 1)))
    q_gamma = (
        float("inf") if tau == N + 1
        else float(np.sort(scores)[tau - 1])
    )
    return q_gamma, scores, theta_bar_unit


def containment_radius_from_fiber_width(
    intersection_radius: float,
    fiber_diameter: float,
    *,
    normalized_fiber_score: bool = False,
) -> float:
    """Inflate an intersection radius into a reward-ray containment radius.

    ``fiber_diameter`` is a certified upper bound (in projective span distance)
    on the unit-span Bellman fiber.  For the ideal, generally nonconvex score
    that measures distance directly to the unit-span fiber, the radius is
    ``q + eta``.  For the tractable cone LP score, normalizing its arbitrary-
    scale witness costs a factor two, giving ``2*q + eta`` whenever ``q < 1``.
    At ``q >= 1`` the zero reward can witness every cone and the only universal
    projective radius is 2.

    The result is clipped at 2, the diameter of the unit sphere of any norm.
    A finite-sample containment claim is valid only when ``fiber_diameter`` is a
    genuine upper bound, not a multistart lower-bound estimate.
    """
    q = float(intersection_radius)
    eta = float(fiber_diameter)
    if q < 0:
        raise ValueError(f"intersection_radius must be nonnegative, got {q}")
    if not 0.0 <= eta <= 2.0:
        raise ValueError(f"fiber_diameter must lie in [0,2], got {eta}")
    if not np.isfinite(q):
        return 2.0
    if normalized_fiber_score:
        return min(2.0, q + eta)
    if q >= 1.0:
        return 2.0
    return min(2.0, 2.0 * q + eta)


def _occupancy_features(mdp: TabularMDP, v: np.ndarray) -> np.ndarray:
    """argmax_{mu in M(P)} <v, Phi^T mu>, returned as the feature vector Phi^T mu
    (normalized by 1/(1-gamma), matching span_seminorm's convention)."""
    from envs.mdp import occupancy_of_policy
    _, _, pi = value_iteration(mdp, v)
    d_occ = occupancy_of_policy(mdp, pi)
    Phi = mdp.phi.reshape(mdp.S * mdp.A, mdp.d)
    return Phi.T @ d_occ.flatten() / (1 - mdp.gamma)


def farthest_in_cone(mdp: TabularMDP, policy: np.ndarray, theta_bar: np.ndarray,
                     n_starts: int = 8, n_iters: int = 15, rng=None,
                     solver=cp.CLARABEL):
    """Legacy heuristic maximum-distance probe.

    A demonstrator acting optimally under its own theta_k has theta_k in the
    Bellman cone, so for any fixed compact normalization set containing theta_k,

        s_k <= ||theta_k - theta_bar||_D <= s~_k        (pointwise, always)

    An *exact* maximum or pointwise upper bound could support a separate
    containment construction. This routine does neither.

    ||.||_D is convex, so unlike distance_to_cone this is a convex MAXIMIZATION
    and not an LP; the maximum sits at an extreme point of the cone-cap. We solve
    it by multistart alternating ascent: for a fixed pair of extremal occupancies
    the inner problem is linear over the cone-cap (an SOCP), and re-deriving the
    extremal occupancies at the new theta never decreases ||.||_D, so the
    iteration ascends monotonically.

    IMPORTANT -- validity direction. The ascent returns a LOWER bound on the true
    maximum. Only an UPPER bound preserves the containment guarantee (over-
    covering is safe; under-covering is not). At the tabular scale used here
    (d <= 8) the exact value is available by enumerating extreme points; callers
    needing a certificate should cross-check against that. See
    conformal/tests/test_decision_metric.py.

    Returns the best value found.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    M_pi = q_linear_operator(mdp, policy)

    theta = cp.Variable(mdp.d)
    g = cp.Parameter(mdp.d)
    cons = [cp.norm(theta, 2) <= 1.0]
    for s in range(mdp.S):
        a_star = policy[s]
        for a in range(mdp.A):
            if a != a_star:
                cons.append(M_pi[s, a] @ theta <= M_pi[s, a_star] @ theta)
    prob = cp.Problem(cp.Maximize(g @ (theta - theta_bar)), cons)

    best = 0.0
    for _ in range(n_starts):
        v = rng.normal(size=mdp.d)
        for _ in range(n_iters):
            g.value = _occupancy_features(mdp, v) - _occupancy_features(mdp, -v)
            try:
                prob.solve(solver=solver)
            except cp.error.SolverError:
                break
            if theta.value is None:
                break
            v_new = np.asarray(theta.value, dtype=float) - theta_bar
            if np.linalg.norm(v_new - v) < 1e-9:
                v = v_new
                break
            v = v_new
        best = max(best, span_seminorm(mdp, v))
    return float(best)


def calibrate_containment(mdp: TabularMDP, policies_val: list[np.ndarray],
                           theta_bar: np.ndarray, gamma: float, rng=None):
    """Heuristic calibration on approximate upper-envelope scores.

    An exact maximum over a justified compact normalization could certify
    containment of a specified latent representative. `farthest_in_cone` is only a
    multistart LOWER bound on that maximum, so this routine does NOT provide the
    theorem's finite-sample certificate.  It is retained for the explicitly
    labelled empirical probe in experiments/run_containment_score.py.

    Returns (q_gamma, s_tilde_ks).
    """
    N = len(policies_val)
    if N == 0:
        raise ValueError("policies_val must contain at least one policy")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")
    rng = rng if rng is not None else np.random.default_rng(0)

    s_tilde = np.array([
        farthest_in_cone(mdp, pi, theta_bar, rng=rng) for pi in policies_val
    ])
    tau = int(np.ceil(gamma * (N + 1)))
    q_gamma = float("inf") if tau == N + 1 else float(np.sort(s_tilde)[tau - 1])
    return q_gamma, s_tilde
