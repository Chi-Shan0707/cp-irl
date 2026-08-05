"""Correctness checks for the T5 separation construction: classic_irl must converge
toward CIO's degenerate theta_u direction (Lemma 1), not theta_star, as N grows.
"""
import numpy as np
import pytest

from envs.mdp import value_iteration
from irl.example1_mdp import (
    build_example1_mdp, sample_theta_hat_example1,
    theta_u_degenerate_direction, THETA_STAR_EXAMPLE1,
)
from irl.point_estimate import classic_irl


def _population(mdp, N, rng):
    pols = []
    for _ in range(N):
        theta_hat = sample_theta_hat_example1(rng)
        _, _, pi = value_iteration(mdp, theta_hat)
        pols.append(pi)
    return pols


@pytest.mark.parametrize("u", [2.0, 10.0, 50.0])
def test_classic_irl_converges_to_theta_u_not_theta_star(u):
    """CIO Lemma 1's claim, transplanted: as N grows, classic_irl's point estimate
    converges to theta_u = (1/sqrt(1+u^2), u/sqrt(1+u^2)), NOT theta_star -- and the
    gap to theta_star grows with u."""
    rng = np.random.default_rng(0)
    mdp = build_example1_mdp(u)
    policies = _population(mdp, N=300, rng=rng)
    theta_bar = classic_irl(mdp, policies)

    theta_u = theta_u_degenerate_direction(u)
    cos_to_theta_u = theta_bar @ theta_u
    cos_to_theta_star = theta_bar @ (THETA_STAR_EXAMPLE1 / np.linalg.norm(THETA_STAR_EXAMPLE1))

    assert cos_to_theta_u > 0.999, (
        f"u={u}: theta_bar should converge to theta_u, cos_sim={cos_to_theta_u}"
    )
    # as u grows, theta_u drifts away from theta_star, so the point estimate should
    # become progressively LESS aligned with the true theta_star
    assert cos_to_theta_star < 0.99, (
        f"u={u}: theta_bar unexpectedly well-aligned with theta_star "
        f"(cos_sim={cos_to_theta_star}) -- the degenerate-collapse construction "
        f"should NOT recover theta_star"
    )


def test_gap_to_theta_star_grows_with_u():
    rng_seed = 0
    cos_sims = []
    for u in [2.0, 10.0, 50.0, 100.0]:
        rng = np.random.default_rng(rng_seed)
        mdp = build_example1_mdp(u)
        policies = _population(mdp, N=300, rng=rng)
        theta_bar = classic_irl(mdp, policies)
        cos_sims.append(theta_bar @ (THETA_STAR_EXAMPLE1 / np.linalg.norm(THETA_STAR_EXAMPLE1)))
    # monotonically non-increasing alignment with theta_star as u grows
    assert all(cos_sims[i] >= cos_sims[i + 1] - 1e-6 for i in range(len(cos_sims) - 1)), (
        f"expected non-increasing alignment with theta_star as u grows: {cos_sims}"
    )
