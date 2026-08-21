"""Value-gap nonconformity score: an alternative to the angular cap score in
`conformal/calibrate.py` / `irl/feasible_set.py`. See
`notes/2026-08-09_value_gap_redesign.md` for the full derivation (Theorems V1
coverage, V2 shaping-invariance).

Score, for demonstrator k with observed policy pi_k, relative to a FIXED point
estimate theta_bar (no optimization over theta -- unlike c_k):

    c_k := V*_theta_bar(s0) - V^{pi_k}_theta_bar(s0) >= 0

Computed from two pieces already implemented in envs/mdp.py: value_iteration (for
V*_theta_bar) and policy_value (for V^{pi_k}_theta_bar). Calibration is the
textbook split-conformal quantile (Vovk et al.) -- no arccos, no SOCP.
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP, policy_value, value_iteration


def value_gap_score(mdp: TabularMDP, policy: np.ndarray, theta_bar: np.ndarray) -> float:
    """c_k = V*_theta_bar(s0) - V^policy_theta_bar(s0). Always >= 0 (V* is optimal)."""
    v_star, _, _ = value_iteration(mdp, theta_bar)
    v_star_s0 = float(mdp.mu0 @ v_star)
    v_pi = policy_value(mdp, theta_bar, policy)
    gap = v_star_s0 - v_pi
    if gap < -1e-7:
        raise RuntimeError(f"value_gap_score: negative gap {gap} (V* not optimal?)")
    return max(gap, 0.0)


def calibrate_value_gap(mdp: TabularMDP, policies_val: list[np.ndarray],
                         theta_bar: np.ndarray, gamma: float) -> tuple[float, np.ndarray]:
    """Standard split-conformal calibration on the value-gap score.

    Returns (q_gamma, c_ks). q_gamma is the tau-th SMALLEST order statistic,
    tau = ceil(gamma*(N+1)); q_gamma = +inf (uninformative sentinel) if tau == N+1,
    the direct analogue of conformal_calibrate's alpha=pi sentinel.
    """
    N = len(policies_val)
    if N == 0:
        raise ValueError("policies_val must contain at least one policy")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")

    v_star, _, _ = value_iteration(mdp, theta_bar)
    v_star_s0 = float(mdp.mu0 @ v_star)
    c_ks = np.array([
        max(v_star_s0 - policy_value(mdp, theta_bar, pi), 0.0) for pi in policies_val
    ])

    tau = int(np.ceil(gamma * (N + 1)))
    if tau == N + 1:
        q_gamma = float("inf")
    else:
        q_gamma = float(np.sort(c_ks)[tau - 1])
    return q_gamma, c_ks


def dkw_calibrate_value_gap(c_ks: np.ndarray, gamma: float, delta: float = 0.05) -> float:
    """DKW concentration-bound analogue of calibrate_value_gap, the value-gap
    counterpart of conformal/concentration_baseline.py::concentration_calibrate --
    same DKW mechanism, opposite quantile direction (value-gap scores want a
    threshold ABOVE which coverage holds, not below, since larger c_k is worse
    here, not better).

    With probability >= 1-delta over the draw of c_ks (DKW: sup_x|F_hat(x)-F(x)|
    <= eps), the empirical (gamma+eps)-quantile q upper-bounds the true
    gamma-quantile of the score distribution: F_hat(q) >= gamma+eps implies
    F(q) >= F_hat(q) - eps >= gamma. Returns +inf (uninformative sentinel) if
    gamma+eps > 1.
    """
    c_ks = np.asarray(c_ks, dtype=float)
    N = len(c_ks)
    if N == 0:
        raise ValueError("c_ks must contain at least one score")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie strictly between 0 and 1, got {delta}")
    eps = np.sqrt(np.log(2.0 / delta) / (2.0 * N))
    target = gamma + eps
    if target > 1.0:
        return float("inf")
    rank = int(np.ceil(N * target))
    rank = min(max(rank, 1), N)
    return float(np.sort(c_ks)[rank - 1])


def lipschitz_bound(mdp: TabularMDP) -> float:
    """bar_nu = ||phi||_{2,infty} / (1 - beta), the same worst-case Lipschitz constant
    derived and verified in notes/T3_T4_proof.md Sec 2 (Lemma T3-Lip / Corollary
    T3-Lip'). Bounds |V_theta(mu) - V_theta'(mu)| <= bar_nu * ||theta - theta'||_2
    for ANY occupancy measure mu, uniformly.
    """
    phi_norms = np.linalg.norm(mdp.phi.reshape(-1, mdp.d), axis=1)
    return float(phi_norms.max() / (1 - mdp.gamma))


def ball_radius(q_gamma: float, mdp: TabularMDP) -> float:
    """rho = q_gamma / bar_nu (Sec 3 of notes/2026-08-09_value_gap_redesign.md):
    converts a calibrated value-gap threshold into a Euclidean-ball radius via the
    worst-case Lipschitz constant, for use with
    robust/value_ball_robust.py::solve_robust_mdp_ball.
    """
    if not np.isfinite(q_gamma):
        return float("inf")
    return q_gamma / lipschitz_bound(mdp)
