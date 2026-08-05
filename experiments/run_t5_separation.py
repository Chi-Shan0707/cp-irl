"""T5 (Separation) demonstration: the MDP analogue of CIO's degenerate Example 1
(plan.md Sec 5, "the paper's punch"). Shows classic IRL's AOG diverging as the
scaling parameter u grows, while conformal IO (CP-IRL) stays bounded/shrinks -- the
core qualitative claim motivating the whole project, now demonstrated for an actual
MDP (not just CIO's own shortest-path LP, which Phase 1 already reproduced).

Three-way comparison (added per plan.md Phase 5's baseline list): classic
point-estimate IRL, the "honest" concentration-based robust MDP baseline (DKW,
conformal/concentration_baseline.py -- valid but not calibration-efficient), and
CP-IRL's exact split-conformal robust MDP. Expect: classic diverges, DKW-robust
stays bounded but is measurably WORSE than conformal-robust (looser calibration ->
larger alpha -> more conservative/costly hedging), conformal-robust is the best of
the three at large u.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_t5_separation.py
"""
from __future__ import annotations

import numpy as np

from envs.mdp import value_iteration, occupancy_lp
from irl.example1_mdp import (
    build_example1_mdp, sample_theta_hat_example1, THETA_STAR_EXAMPLE1,
)
from irl.point_estimate import classic_irl
from irl.feasible_set import c_k as c_k_fast
from conformal.calibrate import conformal_calibrate
from conformal.concentration_baseline import concentration_calibrate
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8


def gen_population(mdp, N, rng):
    policies = []
    for _ in range(N):
        theta_hat = sample_theta_hat_example1(rng)
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
    return policies


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(u, seed, gamma_target=0.8, N_train=100, N_val=100, delta=0.05):
    rng = np.random.default_rng(seed)
    mdp = build_example1_mdp(u)
    theta_star_unit = THETA_STAR_EXAMPLE1 / np.linalg.norm(THETA_STAR_EXAMPLE1)

    pols_train = gen_population(mdp, N_train, rng)
    pols_val = gen_population(mdp, N_val, rng)

    theta_bar = classic_irl(mdp, pols_train)
    alpha_conformal, c_ks = conformal_calibrate(mdp, pols_val, theta_bar, gamma_target,
                                                 n_jobs=N_JOBS)
    alpha_dkw = concentration_calibrate(c_ks, gamma_target, delta=delta)

    d_classic, _ = occupancy_lp(mdp, theta_bar)
    d_conformal, _ = solve_robust_mdp(mdp, theta_bar, alpha_conformal)
    d_dkw, _ = solve_robust_mdp(mdp, theta_bar, alpha_dkw)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aog_classic = v_star - occ_value(mdp, d_classic, theta_star_unit)
    aog_conformal = v_star - occ_value(mdp, d_conformal, theta_star_unit)
    aog_dkw = v_star - occ_value(mdp, d_dkw, theta_star_unit)
    return aog_classic, aog_conformal, aog_dkw, alpha_conformal, alpha_dkw


def main():
    us = [2, 5, 10, 25, 50, 100]
    seeds = list(range(10))

    print(f"{'u':>5} {'AOG_classic':>14} {'AOG_conformal':>15} {'AOG_DKW':>12} "
          f"{'alpha_conf':>11} {'alpha_dkw':>11}")
    for u in us:
        aogs_c, aogs_cio, aogs_dkw, alphas_c, alphas_d = [], [], [], [], []
        for seed in seeds:
            ac, acio, adkw, alpha_c, alpha_d = run_once(u, seed)
            aogs_c.append(ac)
            aogs_cio.append(acio)
            aogs_dkw.append(adkw)
            alphas_c.append(alpha_c)
            alphas_d.append(alpha_d)
        print(f"{u:>5} {np.mean(aogs_c):>14.4f} {np.mean(aogs_cio):>15.4f} "
              f"{np.mean(aogs_dkw):>12.4f} {np.mean(alphas_c):>11.4f} "
              f"{np.mean(alphas_d):>11.4f}")

    print("\nExpected pattern: AOG_classic grows unboundedly with u; AOG_conformal "
          "and AOG_DKW both stay bounded (both are valid robust hedges), but "
          "AOG_DKW should be worse than AOG_conformal since alpha_dkw >= alpha_conf "
          "(DKW's concentration bound is looser than exact split-conformal "
          "calibration -- more conservative hedging costs more AOG).")


if __name__ == "__main__":
    main()
