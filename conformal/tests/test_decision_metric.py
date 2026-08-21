"""Tests for conformal/decision_metric.py: the span seminorm ||.||_D and the
distance-to-cone conformal score, cross-checked against brute-force enumeration,
shaping-direction nullity, feature-reparametrization invariance, the g_k <= s_k
inequality (value_gap.py's score is a single-witness relaxation of
distance_to_cone), and split-conformal coverage. See
CPIRL_FIRST_PRINCIPLES.md Secs 2, 4.
"""
import itertools

import numpy as np
import pytest

from envs.mdp import (
    TabularMDP, occupancy_lp, occupancy_of_policy, random_mdp, value_iteration,
)
from irl.feasible_set import q_linear_operator
from conformal.decision_metric import (
    calibrate_decision_aware,
    calibrate_normalized_decision_aware,
    containment_radius_from_fiber_width,
    distance_to_cone,
    normalize_span,
    normalized_distance_to_cone,
    projective_span_distance,
    span_match_reward,
    span_seminorm,
)
from conformal.value_gap import value_gap_score


def _brute_force_span(mdp: TabularMDP, v: np.ndarray) -> float:
    """max-min of <v, Phi^T mu> / (1-gamma) over ALL deterministic-policy occupancy
    vertices (the polytope's extreme points), independent of value_iteration."""
    best, worst = -np.inf, np.inf
    for pol in itertools.product(range(mdp.A), repeat=mdp.S):
        pol = np.array(pol)
        mu = occupancy_of_policy(mdp, pol)
        val = float(mu.ravel() @ mdp.phi.reshape(-1, mdp.d) @ v) / (1 - mdp.gamma)
        best, worst = max(best, val), min(worst, val)
    return best - worst


@pytest.mark.parametrize("seed", range(5))
def test_span_seminorm_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    mdp = random_mdp(S=5, A=3, d_feat=4, gamma=0.9, rng=rng)
    v = rng.normal(size=mdp.d)
    assert span_seminorm(mdp, v) == pytest.approx(_brute_force_span(mdp, v), abs=1e-6)


def test_span_seminorm_is_nonnegative_and_zero_at_origin():
    rng = np.random.default_rng(1)
    mdp = random_mdp(S=5, A=3, d_feat=4, gamma=0.9, rng=rng)
    assert span_seminorm(mdp, np.zeros(mdp.d)) == pytest.approx(0.0, abs=1e-9)
    for _ in range(5):
        v = rng.normal(size=mdp.d)
        assert span_seminorm(mdp, v) >= -1e-9


def test_span_seminorm_positive_homogeneity_and_subadditivity():
    rng = np.random.default_rng(2)
    mdp = random_mdp(S=5, A=3, d_feat=4, gamma=0.9, rng=rng)
    u, v = rng.normal(size=mdp.d), rng.normal(size=mdp.d)
    c = 3.7
    assert span_seminorm(mdp, c * u) == pytest.approx(c * span_seminorm(mdp, u), abs=1e-6)
    assert span_seminorm(mdp, u + v) <= span_seminorm(mdp, u) + span_seminorm(mdp, v) + 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_shaping_direction_is_null(seed):
    """A pure potential-shaping feature direction has ||.||_D == 0, exactly (not
    just when restricted to policy vertices -- span_seminorm sup's over the WHOLE
    occupancy polytope), while a generic random direction does not."""
    rng = np.random.default_rng(100 + seed)
    S, A, d = 6, 3, 3
    base = random_mdp(S, A, d, gamma=0.9, rng=rng)
    pot = rng.normal(size=S)
    shaping_feat = base.gamma * (base.P @ pot) - pot[:, None]  # (S, A)
    phi_aug = np.concatenate([base.phi, shaping_feat[:, :, None]], axis=2)
    mdp = TabularMDP(base.P, phi_aug, base.gamma, base.mu0)

    c_dir = np.zeros(mdp.d)
    c_dir[-1] = 1.0
    assert span_seminorm(mdp, c_dir) == pytest.approx(0.0, abs=1e-6)
    assert span_seminorm(mdp, 5.0 * c_dir) == pytest.approx(0.0, abs=1e-6)

    rand_dir = rng.normal(size=mdp.d)
    assert span_seminorm(mdp, rand_dir) > 1e-3


@pytest.mark.parametrize("seed", range(5))
def test_feature_reparametrization_invariance(seed):
    """The GL_d test (CPIRL_FIRST_PRINCIPLES.md Sec 1.3): phi -> A@phi,
    theta -> A^{-T}@theta leaves the reward function, the MDP, and behavior
    identical, so ||.||_D must be exactly invariant while the raw Euclidean norm
    generically is not."""
    rng = np.random.default_rng(200 + seed)
    S, A, d = 5, 3, 4
    mdp = random_mdp(S, A, d, gamma=0.9, rng=rng)
    v = rng.normal(size=d)

    G = rng.normal(size=(d, d))
    while abs(np.linalg.det(G)) < 1e-3:
        G = rng.normal(size=(d, d))
    G_invT = np.linalg.inv(G).T

    phi_tilde = mdp.phi @ G.T  # tilde_phi(s,a) = G @ phi(s,a)
    mdp_tilde = TabularMDP(mdp.P, phi_tilde, mdp.gamma, mdp.mu0)
    v_tilde = G_invT @ v

    # reward function is literally unchanged pointwise
    assert np.allclose(mdp.reward(v), mdp_tilde.reward(v_tilde), atol=1e-8)

    span_before = span_seminorm(mdp, v)
    span_after = span_seminorm(mdp_tilde, v_tilde)
    assert span_after == pytest.approx(span_before, abs=1e-6)

    # a generic reparametrization DOES move the Euclidean norm (sanity check that
    # this test is not vacuous)
    assert not np.isclose(np.linalg.norm(v), np.linalg.norm(v_tilde), rtol=1e-2)


def test_span_normalization_and_matching_are_intrinsic():
    rng = np.random.default_rng(219)
    mdp = random_mdp(S=5, A=3, d_feat=4, gamma=0.9, rng=rng)
    theta = rng.normal(size=mdp.d)
    reference = rng.normal(size=mdp.d)

    theta_unit = normalize_span(mdp, theta)
    theta_matched = span_match_reward(mdp, theta, reference)
    assert span_seminorm(mdp, theta_unit) == pytest.approx(1.0, abs=1e-6)
    assert span_seminorm(mdp, theta_matched) == pytest.approx(
        span_seminorm(mdp, reference), abs=1e-6
    )
    # The original positive scale of the latent reward carries no information.
    assert np.allclose(
        span_match_reward(mdp, 7.3 * theta, reference), theta_matched, atol=1e-7
    )

    G = rng.normal(size=(mdp.d, mdp.d))
    while abs(np.linalg.det(G)) < 1e-3:
        G = rng.normal(size=(mdp.d, mdp.d))
    G_invT = np.linalg.inv(G).T
    mdp_t = TabularMDP(mdp.P, mdp.phi @ G.T, mdp.gamma, mdp.mu0)
    matched_t = span_match_reward(mdp_t, G_invT @ theta, G_invT @ reference)
    assert np.allclose(matched_t, G_invT @ theta_matched, atol=1e-6)


def test_span_normalization_rejects_behaviorally_null_reward():
    rng = np.random.default_rng(220)
    mdp = random_mdp(S=4, A=2, d_feat=3, gamma=0.9, rng=rng)
    with pytest.raises(ValueError, match="behaviorally-null"):
        normalize_span(mdp, np.zeros(mdp.d))


def test_projective_span_distance_invariance_and_regret_bound():
    rng = np.random.default_rng(221)
    mdp = random_mdp(S=5, A=3, d_feat=4, gamma=0.9, rng=rng)
    theta = rng.normal(size=mdp.d)
    theta_prime = rng.normal(size=mdp.d)
    d0 = projective_span_distance(mdp, theta, theta_prime)
    assert 0.0 <= d0 <= 2.0 + 1e-6
    assert projective_span_distance(
        mdp, 4.2 * theta, 0.3 * theta_prime
    ) == pytest.approx(d0, abs=1e-6)

    # Normalized regret is controlled with constant one.
    _, _, pi_prime = value_iteration(mdp, theta_prime)
    mu_prime = occupancy_of_policy(mdp, pi_prime)
    _, value_star = occupancy_lp(mdp, theta)
    value_prime = float(
        mu_prime.ravel() @ mdp.phi.reshape(-1, mdp.d) @ theta
    ) / (1 - mdp.gamma)
    normalized_regret = (value_star - value_prime) / span_seminorm(mdp, theta)
    assert normalized_regret <= d0 + 1e-6


def test_normalized_cone_score_is_positive_affine_invariant():
    rng = np.random.default_rng(222)
    mdp = random_mdp(S=5, A=3, d_feat=3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    theta_k = rng.normal(size=mdp.d)
    _, _, policy = value_iteration(mdp, theta_k)

    s0, _, center0 = normalized_distance_to_cone(mdp, policy, theta_bar)
    s1, _, center1 = normalized_distance_to_cone(mdp, policy, 5.7 * theta_bar)
    assert s0 == pytest.approx(s1, abs=1e-6)
    assert span_seminorm(mdp, center0) == pytest.approx(1.0, abs=1e-6)
    assert span_seminorm(mdp, center1) == pytest.approx(1.0, abs=1e-6)


def test_normalized_calibration_matches_individual_scores():
    rng = np.random.default_rng(223)
    mdp = random_mdp(S=5, A=3, d_feat=3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    policies = []
    for _ in range(8):
        _, _, pi = value_iteration(mdp, rng.normal(size=mdp.d))
        policies.append(pi)
    q, scores, center = calibrate_normalized_decision_aware(
        mdp, policies, theta_bar, gamma=0.7
    )
    direct = np.array([
        normalized_distance_to_cone(mdp, pi, theta_bar)[0]
        for pi in policies
    ])
    assert np.allclose(scores, direct, atol=1e-6)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert span_seminorm(mdp, center) == pytest.approx(1.0, abs=1e-6)
    assert q == pytest.approx(np.sort(scores)[int(np.ceil(0.7 * 9)) - 1])


def test_containment_radius_bridge_cases():
    assert containment_radius_from_fiber_width(
        0.2, 0.1, normalized_fiber_score=True
    ) == pytest.approx(0.3)
    assert containment_radius_from_fiber_width(0.2, 0.1) == pytest.approx(0.5)
    assert containment_radius_from_fiber_width(1.0, 0.0) == pytest.approx(2.0)
    assert containment_radius_from_fiber_width(float("inf"), 0.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        containment_radius_from_fiber_width(0.2, 2.1)


@pytest.mark.parametrize("seed", range(8))
def test_distance_to_cone_theta_star_is_actually_feasible(seed):
    """The minimizing theta returned by distance_to_cone must itself keep `policy`
    a no-profitable-deviation policy (the defining property of Theta-hat(pi_k))."""
    rng = np.random.default_rng(300 + seed)
    mdp = random_mdp(S=5, A=3, d_feat=3, gamma=0.9, rng=rng)
    theta0 = rng.normal(size=mdp.d)
    _, _, policy = value_iteration(mdp, theta0)
    theta_bar = rng.normal(size=mdp.d)

    s_k, theta_star = distance_to_cone(mdp, policy, theta_bar)
    assert s_k >= -1e-6

    M_pi = q_linear_operator(mdp, policy)
    for s in range(mdp.S):
        a_star = policy[s]
        q_star = M_pi[s, a_star] @ theta_star
        for a in range(mdp.A):
            assert M_pi[s, a] @ theta_star <= q_star + 1e-5


@pytest.mark.parametrize("seed", range(8))
def test_value_gap_lower_bounds_distance_to_cone(seed):
    """g_k(theta_bar) <= s_k(theta_bar): value-gap is a single-occupancy-pair
    relaxation of the full distance-to-cone score (CPIRL_FIRST_PRINCIPLES.md
    Sec 2.4)."""
    rng = np.random.default_rng(400 + seed)
    mdp = random_mdp(S=5, A=3, d_feat=3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)

    theta_k = theta_bar + 0.4 * rng.normal(size=mdp.d)
    _, _, policy = value_iteration(mdp, theta_k)

    g_k = value_gap_score(mdp, policy, theta_bar)
    s_k, _ = distance_to_cone(mdp, policy, theta_bar)
    assert g_k <= s_k + 1e-5


def test_calibrate_decision_aware_sentinel_and_shape():
    rng = np.random.default_rng(5)
    mdp = random_mdp(S=5, A=3, d_feat=3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    pols = [rng.integers(0, mdp.A, size=mdp.S) for _ in range(3)]
    q_gamma, s_ks = calibrate_decision_aware(mdp, pols, theta_bar, gamma=0.999)
    assert q_gamma == float("inf")
    assert len(s_ks) == 3
    assert np.all(s_ks >= -1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_coverage_of_decision_aware_score(seed):
    """Empirical split-conformal coverage check, the decision-metric analogue of
    conformal/tests/test_value_gap.py::test_coverage_theorem_v1."""
    rng = np.random.default_rng(500 + seed)
    mdp = random_mdp(S=5, A=3, d_feat=3, gamma=0.9, rng=rng)
    theta_star = rng.normal(size=mdp.d)
    theta_star /= np.linalg.norm(theta_star)

    gamma_target = 0.7
    n_trials = 30
    hits = 0
    for _ in range(n_trials):
        theta_bar = theta_star + rng.normal(0, 0.25, size=mdp.d)
        pols_val = []
        for _ in range(14):
            tk = theta_star + 0.3 * rng.normal(size=mdp.d)
            _, _, pol = value_iteration(mdp, tk)
            pols_val.append(pol)
        q_gamma, _ = calibrate_decision_aware(mdp, pols_val, theta_bar, gamma_target)

        t_new = theta_star + 0.3 * rng.normal(size=mdp.d)
        _, _, pol_new = value_iteration(mdp, t_new)
        s_new, _ = distance_to_cone(mdp, pol_new, theta_bar)
        if s_new <= q_gamma + 1e-6:
            hits += 1

    coverage = hits / n_trials
    assert coverage >= gamma_target - 0.15, f"coverage {coverage} too far below {gamma_target}"


# ---------------------------------------------------------------------------
# Conjugate score / containment coverage (paper Thm. `containment`)
# ---------------------------------------------------------------------------

def test_conjugate_score_dominates_min_score():
    """s_k <= s~_k must hold for every demonstrator: same set, min vs max."""
    from conformal.decision_metric import farthest_in_cone
    rng = np.random.default_rng(7)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)
    for _ in range(4):
        theta_k = rng.normal(size=mdp.d)
        _, _, pi_k = value_iteration(mdp, theta_k)
        s_min, _ = distance_to_cone(mdp, pi_k, theta_bar)
        s_max = farthest_in_cone(mdp, pi_k, theta_bar, rng=rng)
        assert s_max >= s_min - 1e-6, f"max {s_max} < min {s_min}"


def test_conjugate_score_dominates_demonstrator_reward():
    """The inequality the containment theorem rests on:
    ||theta_k - theta_bar||_D <= s~_k, because theta_k lies in its own cone."""
    from conformal.decision_metric import farthest_in_cone
    rng = np.random.default_rng(11)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)
    for _ in range(5):
        theta_k = rng.normal(size=mdp.d)
        theta_k /= np.linalg.norm(theta_k)  # same normalization as the cone-cap
        _, _, pi_k = value_iteration(mdp, theta_k)
        s_max = farthest_in_cone(mdp, pi_k, theta_bar, rng=rng)
        realized = span_seminorm(mdp, theta_k - theta_bar)
        assert realized <= s_max + 1e-5, f"realized {realized} > s~ {s_max}"


def test_containment_calibration_shape_and_monotonicity():
    """calibrate_containment returns a quantile at least as large as the
    intersection-coverage quantile on the same demonstrators."""
    from conformal.decision_metric import calibrate_containment
    rng = np.random.default_rng(13)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.normal(size=mdp.d)
    theta_bar /= np.linalg.norm(theta_bar)
    pols = []
    for _ in range(8):
        th = rng.normal(size=mdp.d)
        _, _, pi = value_iteration(mdp, th)
        pols.append(pi)
    q_contain, s_tilde = calibrate_containment(mdp, pols, theta_bar, 0.8, rng=rng)
    q_inter, s_min = calibrate_decision_aware(mdp, pols, theta_bar, 0.8)
    assert len(s_tilde) == len(pols)
    assert np.all(s_tilde >= s_min - 1e-6)
    assert q_contain >= q_inter - 1e-6
