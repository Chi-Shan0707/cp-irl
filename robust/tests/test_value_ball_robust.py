"""Tests for robust/value_ball_robust.py: closed-form Euclidean-ball RFO."""
import cvxpy as cp
import numpy as np
import pytest

from envs.mdp import random_mdp, occupancy_lp
from robust.value_ball_robust import solve_robust_mdp_ball


@pytest.mark.parametrize("seed", range(8))
def test_value_shrinks_as_rho_grows(seed):
    rng = np.random.default_rng(seed)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)

    _, val_small = solve_robust_mdp_ball(mdp, theta_bar, rho=0.1)
    _, val_big = solve_robust_mdp_ball(mdp, theta_bar, rho=1.0)

    assert val_big <= val_small + 1e-4


def test_reduces_to_plain_fo_at_rho_zero():
    rng = np.random.default_rng(4)
    mdp = random_mdp(5, 3, 3, gamma=0.9, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)

    _, val_plain = occupancy_lp(mdp, theta_bar)
    _, val_robust = solve_robust_mdp_ball(mdp, theta_bar, rho=0.0)

    assert val_robust == pytest.approx(val_plain, abs=1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_matches_socp_reference(seed):
    """Cross-check the closed-form solve against a direct joint (mu, theta) SOCP
    (a second, independent formulation of the same max-min), the same style of
    correctness check irl/feasible_set.py uses for c_k vs c_k_reference."""
    rng = np.random.default_rng(seed)
    mdp = random_mdp(4, 2, 2, gamma=0.85, rng=rng)
    theta_bar = rng.uniform(0.5, 2.0, size=mdp.d)
    rho = 0.5

    d_fast, val_fast = solve_robust_mdp_ball(mdp, theta_bar, rho)

    # Reference: max_{mu in M} t  s.t.  t <= theta^T Phi^T mu for ALL theta with
    # ||theta - theta_bar|| <= rho -- equivalently (by definition of the inner min's
    # closed form being what we're checking) t <= theta_bar^T x - rho*||x||_2 with
    # x = Phi^T mu.  Build via cvxpy directly from the flow polytope to avoid reusing
    # the module under test's own algebra.
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
    Phi_flat = mdp.phi.reshape(n, mdp.d)

    d_var = cp.Variable(n, nonneg=True)
    x = Phi_flat.T @ d_var
    prob = cp.Problem(cp.Maximize(theta_bar @ x - rho * cp.norm(x, 2)),
                       [A_flow @ d_var == b_flow])
    prob.solve(solver=cp.CLARABEL)
    val_ref = float(prob.value) / (1 - gamma)

    assert val_fast == pytest.approx(val_ref, abs=1e-4)
