"""Tests for robust/decision_aware_robust.py: the centered max-min LP over the
decision-aware ball C_q(theta_bar). Two things are checked that are NOT obvious
from the module docstring alone: (1) the degeneracy when mu_ref is theta_bar's
own optimum (a real mathematical fact, not a bug -- see the WARNING in the
module), and (2) sane, non-degenerate behavior once mu_ref is a genuinely
different baseline.
"""
import numpy as np
import pytest
import cvxpy as cp

from envs.mdp import occupancy_lp, occupancy_of_policy, random_mdp
from robust.decision_aware_robust import (
    mean_population_occupancy,
    solve_robust_mdp_decision_aware,
    solve_safe_robust_mdp,
)


@pytest.mark.parametrize("seed", range(5))
def test_degenerate_at_own_optimal_reference(seed):
    """WARNING case: mu_ref = theta_bar's own optimal occupancy makes the
    criterion vacuous -- optimal mu is mu_ref itself, value == 0, for ANY q."""
    rng = np.random.default_rng(seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    mu_ref, val_ref = occupancy_lp(mdp, theta_bar)

    for q in (0.0, 0.5, 5.0, float("inf")):
        d_occ, value = solve_robust_mdp_decision_aware(mdp, theta_bar, q, mu_ref)
        assert value == pytest.approx(0.0, abs=1e-4), f"q={q}: value={value}"
        assert np.allclose(d_occ, mu_ref, atol=1e-3), f"q={q}: mu drifted from mu_ref"


@pytest.mark.parametrize("seed", range(6))
def test_reduces_to_raw_advantage_at_q_zero(seed):
    """At q=0 the gauge penalty is switched off entirely, so the LP reduces to
    freely maximizing raw relative value: max_mu <theta_bar, Phi^T(mu-mu_ref)>,
    an exact closed form (point-estimate value minus mu_ref's own value)."""
    rng = np.random.default_rng(100 + seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)

    uniform_policy = np.zeros(mdp.S, dtype=int)  # always action 0: a genuinely
    mu_ref = occupancy_of_policy(mdp, uniform_policy)  # different, non-optimal baseline

    from envs.mdp import policy_value
    _, point_val = occupancy_lp(mdp, theta_bar)
    ref_val = policy_value(mdp, theta_bar, uniform_policy)

    _, value_q0 = solve_robust_mdp_decision_aware(mdp, theta_bar, 0.0, mu_ref)
    assert value_q0 == pytest.approx(point_val - ref_val, abs=1e-4)


@pytest.mark.parametrize("seed", range(6))
def test_value_nonincreasing_in_q_nondegenerate(seed):
    """With a genuinely different (non-optimal) baseline, worst-case relative
    value must be non-increasing as the uncertainty budget q grows -- a bigger
    adversarial ball can never help."""
    rng = np.random.default_rng(200 + seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    uniform_policy = np.zeros(mdp.S, dtype=int)
    mu_ref = occupancy_of_policy(mdp, uniform_policy)

    qs = [0.0, 0.05, 0.2, 1.0, 5.0]
    values = [solve_robust_mdp_decision_aware(mdp, theta_bar, q, mu_ref)[1] for q in qs]
    for a, b in zip(values, values[1:]):
        assert b <= a + 1e-4, f"values not non-increasing: {values}"


@pytest.mark.parametrize("seed", range(4))
def test_mean_population_occupancy_is_valid_occupancy(seed):
    rng = np.random.default_rng(300 + seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    policies = [rng.integers(0, mdp.A, size=mdp.S) for _ in range(10)]
    mu_ref = mean_population_occupancy(mdp, policies)
    assert mu_ref.shape == (mdp.S, mdp.A)
    assert np.min(mu_ref) >= -1e-9
    assert mu_ref.sum() == pytest.approx(1.0, abs=1e-6)

    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    # solver should run without error against this reference and respect the
    # non-increasing-in-q property too
    v0 = solve_robust_mdp_decision_aware(mdp, theta_bar, 0.0, mu_ref)[1]
    v1 = solve_robust_mdp_decision_aware(mdp, theta_bar, 1.0, mu_ref)[1]
    assert v1 <= v0 + 1e-4


def test_infinite_q_sentinel_matches_zero_relative_value_or_worse():
    rng = np.random.default_rng(9)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    uniform_policy = np.zeros(mdp.S, dtype=int)
    mu_ref = occupancy_of_policy(mdp, uniform_policy)

    _, value_inf = solve_robust_mdp_decision_aware(mdp, theta_bar, float("inf"), mu_ref)
    assert value_inf <= 1e-3  # can never beat sticking with mu_ref under an unbounded ball


def test_reported_value_uses_discounted_return_units():
    """Regression for the (1-gamma) factor on q.

    The solver represents occupancy features without 1/(1-gamma), whereas the
    calibrated span radius q is in discounted-return units.  Recompute the
    gauge of the returned occupancy independently and verify the exact stated
    objective theta*x/(1-gamma) - q*gauge(x).
    """
    rng = np.random.default_rng(707)
    mdp = random_mdp(5, 3, 3, gamma=0.8, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    mu_ref = occupancy_of_policy(mdp, np.zeros(mdp.S, dtype=int))
    q = 0.37
    mu, value = solve_robust_mdp_decision_aware(mdp, theta_bar, q, mu_ref)

    n = mdp.S * mdp.A
    phi = mdp.phi.reshape(n, mdp.d)
    x = phi.T @ (mu.reshape(n) - mu_ref.reshape(n))

    # Independent gauge LP: x = Phi^T(mu_p-nu_p), with both scaled occupancy
    # measures carrying total mass t.
    from robust.decision_aware_robust import _flow_matrices
    A_flow, b_flow = _flow_matrices(mdp)
    mu_p = cp.Variable(n, nonneg=True)
    nu_p = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(t), [
        A_flow @ mu_p == t * b_flow,
        A_flow @ nu_p == t * b_flow,
        phi.T @ (mu_p - nu_p) == x,
    ])
    prob.solve(solver=cp.CLARABEL)
    expected = float(theta_bar @ x) / (1 - mdp.gamma) - q * float(t.value)
    assert value == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize("seed", range(4))
def test_safe_wrapper_is_invariant_to_center_scale(seed):
    rng = np.random.default_rng(800 + seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    mu_ref = occupancy_of_policy(mdp, np.zeros(mdp.S, dtype=int))

    result = solve_safe_robust_mdp(
        mdp, theta_bar, intersection_radius=0.1, fiber_diameter=0.1,
        mu_ref=mu_ref, normalized_fiber_score=True,
    )
    scaled = solve_safe_robust_mdp(
        mdp, 7.0 * theta_bar, intersection_radius=0.1,
        fiber_diameter=0.1, mu_ref=mu_ref,
        normalized_fiber_score=True,
    )
    assert result.containment_radius == pytest.approx(0.2)
    assert result.nontrivial
    assert result.worst_case_advantage >= -1e-8
    assert np.allclose(result.occupancy, scaled.occupancy, atol=2e-5)
    assert result.worst_case_advantage == pytest.approx(
        scaled.worst_case_advantage, abs=2e-5
    )


def test_safe_wrapper_guarantees_reference_dominance_for_center_reward():
    rng = np.random.default_rng(811)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    mu_ref = occupancy_of_policy(mdp, np.zeros(mdp.S, dtype=int))
    result = solve_safe_robust_mdp(
        mdp, theta_bar, intersection_radius=0.08, fiber_diameter=0.1,
        mu_ref=mu_ref, normalized_fiber_score=True,
    )

    from conformal.decision_metric import normalize_span
    theta_unit = normalize_span(mdp, theta_bar)
    reward = mdp.reward(theta_unit).reshape(-1)
    improvement = float(
        (result.occupancy.reshape(-1) - mu_ref.reshape(-1)) @ reward
    ) / (1 - mdp.gamma)
    assert improvement >= -2e-5


@pytest.mark.parametrize(
    "q,eta,normalized_score",
    [(0.45, 0.2, False), (0.8, 0.3, True), (1.0, 0.0, False)],
)
def test_safe_wrapper_returns_reference_when_containment_ball_is_degenerate(
    q, eta, normalized_score,
):
    rng = np.random.default_rng(812)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    mu_ref = occupancy_of_policy(mdp, np.zeros(mdp.S, dtype=int))
    result = solve_safe_robust_mdp(
        mdp, theta_bar, intersection_radius=q, fiber_diameter=eta,
        mu_ref=mu_ref, normalized_fiber_score=normalized_score,
    )
    assert result.containment_radius >= 1.0
    assert not result.nontrivial
    assert result.worst_case_advantage == 0.0
    assert np.array_equal(result.occupancy, mu_ref)


def test_safe_wrapper_validates_reference_even_in_degenerate_regime():
    rng = np.random.default_rng(813)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    bad_ref = np.full((mdp.S, mdp.A), 1.0 / (mdp.S * mdp.A))
    with pytest.raises(ValueError, match="flow constraints"):
        solve_safe_robust_mdp(
            mdp,
            rng.normal(size=mdp.d),
            intersection_radius=1.0,
            fiber_diameter=2.0,
            mu_ref=bad_ref,
        )
