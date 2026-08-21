"""IRL analogue of CIO's Theorem 1 calibration (cio/io_pipeline.py::conformal_calibrate),
built on the resolvent-accelerated c_k from irl/feasible_set.py (plan.md Sec 4 P2,
Sec 5 T1/T2). Population-of-demonstrators setting (plan.md P1, unit A): each
demonstrator k contributes one policy (their observed behavior) and one calibration
point c_k.
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP
from irl.feasible_set import c_k as c_k_fast
from irl.feasible_set import q_linear_operator


def conformal_calibrate(mdp: TabularMDP, policies_val: list[np.ndarray],
                         theta_bar: np.ndarray, gamma: float,
                         n_jobs: int | None = None) -> tuple[float, np.ndarray]:
    """CIO Theorem 1, transplanted: alpha_gamma = arccos(Gamma_tau({c_k})),
    tau = ceil(gamma*(N+1)), Gamma_tau = the tau-th LARGEST value.  When
    tau=N+1, the threshold is the conformal sentinel -1 (the whole unit ball).
    Returns (alpha_gamma, array of c_k). Embarrassingly parallel across demonstrators
    (each c_k only needs that demonstrator's own policy).
    """
    N = len(policies_val)
    if N == 0:
        raise ValueError("policies_val must contain at least one policy")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly between 0 and 1, got {gamma}")
    theta_bar = np.asarray(theta_bar, dtype=float)
    if theta_bar.shape != (mdp.d,):
        raise ValueError(f"theta_bar must have shape ({mdp.d},), got {theta_bar.shape}")
    if not np.isclose(np.linalg.norm(theta_bar), 1.0, atol=1e-7):
        raise ValueError("theta_bar must have unit Euclidean norm")

    tau = int(np.ceil(gamma * (N + 1)))

    if n_jobs is None or n_jobs == 1:
        c_ks = np.array([c_k_fast(mdp, pi, theta_bar) for pi in policies_val])
    else:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            c_ks = np.array(pool.starmap(
                c_k_fast, [(mdp, pi, theta_bar) for pi in policies_val]))

    # If tau=N+1, no calibration order statistic can deliver the requested
    # finite-sample level.  The conformal sentinel -1 returns alpha=pi, i.e.
    # the whole unit ball, which is valid but intentionally uninformative.
    if tau == N + 1:
        c_tau = -1.0
    else:
        sorted_desc = np.sort(c_ks)[::-1]
        c_tau = sorted_desc[tau - 1]
    c_tau = np.clip(c_tau, -1.0, 1.0)
    alpha_gamma = np.arccos(c_tau)
    return float(alpha_gamma), c_ks
