"""Phase 5 ablation (plan.md Sec 6): gamma sweep on the gridworld env, for the
LP-IRL point estimator (the one showing the largest classic-vs-robust gap in
notes/phase5_gridworld_result.md, hence the most informative for seeing how alpha
and coverage respond to the target confidence level).

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_phase5_gamma_ablation.py
"""
from __future__ import annotations

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.mdp import value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from irl.feasible_set import c_k as c_k_fast
from conformal.calibrate import conformal_calibrate
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8


def gen_population(mdp, theta_star, N, rng, noise_std=0.5, eps0=0.05):
    policies = []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
    return policies


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(seed, gamma_target, size=6, n_regions=4, feature_dim=4, N=30):
    rng = np.random.default_rng(seed)
    mdp, _ = build_region_gridworld(size=size, n_regions=n_regions,
                                     feature_dim=feature_dim, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    pols_train = gen_population(mdp, theta_star, N, rng)
    pols_val = gen_population(mdp, theta_star, N, rng)
    theta_bar = classic_irl(mdp, pols_train)
    alpha, _ = conformal_calibrate(mdp, pols_val, theta_bar, gamma_target, n_jobs=N_JOBS)

    pols_cov = gen_population(mdp, theta_star, N, rng)
    from multiprocessing import Pool
    with Pool(N_JOBS) as pool:
        c_ks_cov = np.array(pool.starmap(c_k_fast, [(mdp, pi, theta_bar) for pi in pols_cov]))
    coverage = float(np.mean(c_ks_cov >= np.cos(alpha) - 1e-9))

    d_classic, _ = occupancy_lp(mdp, theta_bar)
    d_conformal, _ = solve_robust_mdp(mdp, theta_bar, alpha)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aog_classic = v_star - occ_value(mdp, d_classic, theta_star_unit)
    aog_conformal = v_star - occ_value(mdp, d_conformal, theta_star_unit)
    return alpha, coverage, aog_classic, aog_conformal


def main():
    gammas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    seeds = list(range(5))

    print(f"{'gamma':>6} {'mean alpha':>11} {'mean coverage':>14} "
          f"{'mean AOG_classic':>17} {'mean AOG_conformal':>19}")
    for gamma in gammas:
        alphas, covs, aog_cs, aog_rs = [], [], [], []
        for seed in seeds:
            a, c, ac, ar = run_once(seed, gamma)
            alphas.append(a)
            covs.append(c)
            aog_cs.append(ac)
            aog_rs.append(ar)
        print(f"{gamma:>6.2f} {np.mean(alphas):>11.4f} {np.mean(covs):>14.3f} "
              f"{np.mean(aog_cs):>17.4f} {np.mean(aog_rs):>19.4f}")


if __name__ == "__main__":
    main()
