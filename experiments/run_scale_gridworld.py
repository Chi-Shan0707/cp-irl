"""Scale check beyond the paper's toy regime (|S| <= 36, d <= 5).

Reuses experiments/run_phase5_gridworld.py's pipeline (three point estimators x
three decision rules, plus behavior cloning) but parameterizes grid size and
feature dimension, and reports:
  (a) a wall-clock scaling table across |S| = 36, 100, 225, 400 to show the
      pipeline (classic/MaxEnt/Bayesian IRL, conformal calibration, robust MDP)
      remains tractable well past the paper's largest reported instance, and
  (b) full AOG/POG/coverage metrics at a genuinely larger instance
      (size=15 -> |S|=225, feature_dim=7 -> d=7), addressing the reviewer
      concern that all reported experiments are toy-scale tabular MDPs.

This is still a tabular experiment (not function approximation / deep RL) --
that scope limitation is not resolved here and should stay stated as such in
the paper; this script only tests whether the *tabular* claims hold up at
larger |S| and d than currently reported.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_scale_gridworld.py
"""
from __future__ import annotations

import time

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.mdp import value_iteration, occupancy_lp, occupancy_of_policy
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl
from irl.bayesian_irl import bayesian_irl
from irl.behavior_cloning import behavior_cloning_policy
from irl.feasible_set import c_k as c_k_fast
from conformal.calibrate import conformal_calibrate
from conformal.concentration_baseline import concentration_calibrate
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8

POINT_ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=150),
    "Bayesian": lambda mdp, pols: bayesian_irl(mdp, pols, n_samples=400, burn_in=150,
                                                rng=np.random.default_rng(0)),
}


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


def run_once(seed, size, n_regions, feature_dim, gamma_target=0.8,
             N_train=15, N_val=15, N_test=15):
    rng = np.random.default_rng(seed)
    mdp, _ = build_region_gridworld(size=size, n_regions=n_regions,
                                     feature_dim=feature_dim, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    pols_train = gen_population(mdp, theta_star, N_train, rng)
    pols_val = gen_population(mdp, theta_star, N_val, rng)
    theta_hats_test = []
    for _ in range(N_test):
        p = rng.uniform(0.5, 2.0, size=feature_dim)
        eps = rng.normal(0, 0.5, size=feature_dim)
        th = np.maximum(theta_star * p + eps, 0.0) + 0.05
        theta_hats_test.append(th)

    _, v_star = occupancy_lp(mdp, theta_star_unit)

    rows = []
    for est_name, estimator in POINT_ESTIMATORS.items():
        t0 = time.time()
        theta_bar = estimator(mdp, pols_train)
        t_fit = time.time() - t0

        t0 = time.time()
        alpha_conf, c_ks = conformal_calibrate(mdp, pols_val, theta_bar, gamma_target,
                                                n_jobs=N_JOBS)
        t_calib = time.time() - t0
        alpha_dkw = concentration_calibrate(c_ks, gamma_target, delta=0.05)

        pols_cov = gen_population(mdp, theta_star, N_test, rng)
        from multiprocessing import Pool
        with Pool(N_JOBS) as pool:
            c_ks_cov = np.array(pool.starmap(c_k_fast, [(mdp, pi, theta_bar) for pi in pols_cov]))
        coverage = float(np.mean(c_ks_cov >= np.cos(alpha_conf) - 1e-9))

        d_classic, _ = occupancy_lp(mdp, theta_bar)
        d_conformal, _ = solve_robust_mdp(mdp, theta_bar, alpha_conf)
        d_dkw, _ = solve_robust_mdp(mdp, theta_bar, alpha_dkw)

        for policy_name, d_occ in [("classic", d_classic), ("conformal", d_conformal),
                                    ("dkw", d_dkw)]:
            aog = v_star - occ_value(mdp, d_occ, theta_star_unit)
            pog_vals = []
            for th in theta_hats_test:
                _, v_star_perceived = occupancy_lp(mdp, th)
                pog_vals.append(v_star_perceived - occ_value(mdp, d_occ, th))
            rows.append(dict(seed=seed, estimator=est_name, policy=policy_name,
                              aog=aog, pog=float(np.mean(pog_vals)),
                              alpha=alpha_conf if policy_name == "conformal" else
                              (alpha_dkw if policy_name == "dkw" else 0.0),
                              coverage=coverage, t_fit=t_fit, t_calib=t_calib))

    bc_pi = behavior_cloning_policy(mdp, pols_train)
    d_bc = occupancy_of_policy(mdp, bc_pi)
    aog_bc = v_star - occ_value(mdp, d_bc, theta_star_unit)
    pog_bc_vals = [occ_value(mdp, d_bc, th) for th in theta_hats_test]
    _, v_star_perceived_list = zip(*[occupancy_lp(mdp, th) for th in theta_hats_test])
    pog_bc = float(np.mean([vp - pb for vp, pb in zip(v_star_perceived_list, pog_bc_vals)]))
    rows.append(dict(seed=seed, estimator="none", policy="behavior_cloning",
                      aog=aog_bc, pog=pog_bc, alpha=0.0, coverage=float("nan"),
                      t_fit=0.0, t_calib=0.0))

    return rows


def print_summary(all_rows):
    print(f"{'estimator':>10} {'policy':>14} {'mean AOG':>10} {'mean POG':>10} "
          f"{'mean alpha':>11} {'mean coverage':>14} {'mean t_fit':>11}")
    combos = sorted(set((r["estimator"], r["policy"]) for r in all_rows))
    for est, pol in combos:
        subset = [r for r in all_rows if r["estimator"] == est and r["policy"] == pol]
        aogs = [r["aog"] for r in subset]
        pogs = [r["pog"] for r in subset]
        alphas = [r["alpha"] for r in subset]
        covs = [r["coverage"] for r in subset if not np.isnan(r["coverage"])]
        t_fits = [r["t_fit"] for r in subset]
        cov_str = f"{np.mean(covs):.3f}" if covs else "n/a"
        print(f"{est:>10} {pol:>14} {np.mean(aogs):>10.4f} {np.mean(pogs):>10.4f} "
              f"{np.mean(alphas):>11.4f} {cov_str:>14} {np.mean(t_fits):>11.3f}")


def scaling_table():
    print("=== Wall-clock scaling across |S| (single seed, one estimator fit + one "
          "calibration pass, N_train=N_val=10) ===")
    print(f"{'size':>5} {'|S|':>6} {'d':>3} {'t_fit(LP)':>11} {'t_calib':>10} {'t_robust':>10}")
    for size, feature_dim in [(6, 4), (10, 5), (15, 7), (20, 8)]:
        rng = np.random.default_rng(0)
        mdp, _ = build_region_gridworld(size=size, n_regions=min(8, size), feature_dim=feature_dim,
                                         gamma=0.9, rng=rng)
        theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
        pols_train = gen_population(mdp, theta_star, 10, rng)
        pols_val = gen_population(mdp, theta_star, 10, rng)

        t0 = time.time()
        theta_bar = classic_irl(mdp, pols_train)
        t_fit = time.time() - t0

        t0 = time.time()
        alpha, _ = conformal_calibrate(mdp, pols_val, theta_bar, 0.8, n_jobs=N_JOBS)
        t_calib = time.time() - t0

        t0 = time.time()
        solve_robust_mdp(mdp, theta_bar, alpha)
        t_robust = time.time() - t0

        print(f"{size:>5} {size * size:>6} {feature_dim:>3} {t_fit:>11.2f} "
              f"{t_calib:>10.2f} {t_robust:>10.2f}")


def main():
    scaling_table()

    print("\n=== Full metrics at a scaled-up instance: size=15 (|S|=225), "
          "feature_dim=7 (d=7), n_regions=8 -- 6.25x larger than the paper's "
          "largest reported |S|=36 instance, and d exceeds the reported d<=5 ===")
    seeds = list(range(5))
    all_rows = []
    t0 = time.time()
    for seed in seeds:
        all_rows.extend(run_once(seed, size=15, n_regions=8, feature_dim=7))
    elapsed = time.time() - t0
    print(f"Total: {elapsed:.1f}s for {len(seeds)} seeds\n")
    print_summary(all_rows)


if __name__ == "__main__":
    main()
