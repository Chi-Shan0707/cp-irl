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
    tau = ceil(gamma*(N+1)), Gamma_tau = the tau-th LARGEST value.
    Returns (alpha_gamma, array of c_k). Embarrassingly parallel across demonstrators
    (each c_k only needs that demonstrator's own policy).
    """
    N = len(policies_val)
    tau = int(np.ceil(gamma * (N + 1)))
    tau = min(max(tau, 1), N)

    if n_jobs is None or n_jobs == 1:
        c_ks = np.array([c_k_fast(mdp, pi, theta_bar) for pi in policies_val])
    else:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            c_ks = np.array(pool.starmap(
                c_k_fast, [(mdp, pi, theta_bar) for pi in policies_val]))

    sorted_desc = np.sort(c_ks)[::-1]
    c_tau = sorted_desc[tau - 1]
    c_tau = np.clip(c_tau, -1.0, 1.0)
    alpha_gamma = np.arccos(c_tau)
    return float(alpha_gamma), c_ks
