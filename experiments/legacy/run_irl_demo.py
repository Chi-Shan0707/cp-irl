"""Phase 2 gate: end-to-end CP-IRL on an actual MDP (not the shortest-path LP used to
reproduce CIO in Phase 1) -- population of demonstrators -> classic IRL point estimate
-> conformal calibration (resolvent-accelerated c_k) -> robust MDP -> AOG/POG.

Setting simplification (plan.md Sec 4 P1, unit A, as committed): all demonstrators
share the SAME MDP (P, phi, mu0) -- no varying per-demonstrator context u_k. This means
AOG (measured against the fixed ground-truth theta_star) has no population-level
randomness to average over -- it's a single number per (mdp, theta_star) draw, and we
get statistical variation across SEEDS (different mdp/theta_star draws), not across a
test set. POG *does* retain population variation, since each test demonstrator has
their own theta_hat_k. This is reported honestly below, not glossed over -- adding
varying context u_k (plan's P1 unit B) would restore per-instance AOG variation and is
a documented future extension, not implemented here.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_irl_demo.py
"""
from __future__ import annotations

import time

import numpy as np

from envs.mdp import random_mdp, value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from conformal.calibrate import conformal_calibrate
from irl.feasible_set import c_k as c_k_fast
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8


def gen_population(mdp, theta_star, N, rng, noise_std=1.0, eps0=0.1, p_range=(0.5, 2.0)):
    policies, theta_hats = [], []
    for _ in range(N):
        p = rng.uniform(*p_range, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, policy = value_iteration(mdp, theta_hat)
        policies.append(policy)
        theta_hats.append(theta_hat)
    return policies, theta_hats


def occupancy_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(seed, gamma, N_train, N_val, N_test, S=10, A=4, d_feat=5):
    rng = np.random.default_rng(seed)
    mdp = random_mdp(S, A, d_feat, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=d_feat)

    policies_train, _ = gen_population(mdp, theta_star, N_train, rng)
    policies_val, _ = gen_population(mdp, theta_star, N_val, rng)
    _, theta_hats_test = gen_population(mdp, theta_star, N_test, rng)

    theta_bar = classic_irl(mdp, policies_train)
    alpha, _ = conformal_calibrate(mdp, policies_val, theta_bar, gamma, n_jobs=N_JOBS)

    # coverage on a fresh held-out sample
    policies_cov, _ = gen_population(mdp, theta_star, N_test, rng)
    from multiprocessing import Pool
    with Pool(N_JOBS) as pool:
        c_ks_cov = np.array(pool.starmap(
            c_k_fast, [(mdp, pi, theta_bar) for pi in policies_cov]))
    coverage = float(np.mean(c_ks_cov >= np.cos(alpha) - 1e-9))

    d_classic, _ = occupancy_lp(mdp, theta_bar)
    d_robust, _ = solve_robust_mdp(mdp, theta_bar, alpha)

    _, val_star = occupancy_lp(mdp, theta_star)  # V*_{theta*}
    aog_classic = val_star - occupancy_value(mdp, d_classic, theta_star)
    aog_cio = val_star - occupancy_value(mdp, d_robust, theta_star)

    pog_classic_vals, pog_cio_vals = [], []
    for theta_hat in theta_hats_test:
        _, val_star_perceived = occupancy_lp(mdp, theta_hat)
        pog_classic_vals.append(val_star_perceived - occupancy_value(mdp, d_classic, theta_hat))
        pog_cio_vals.append(val_star_perceived - occupancy_value(mdp, d_robust, theta_hat))

    return dict(seed=seed, gamma=gamma, alpha=alpha, coverage=coverage,
                aog_classic=aog_classic, aog_cio=aog_cio,
                pog_classic=float(np.mean(pog_classic_vals)),
                pog_cio=float(np.mean(pog_cio_vals)))


def main():
    seeds = list(range(8))
    gammas = [0.5, 0.7, 0.9]
    N_train, N_val, N_test = 30, 30, 30

    print(f"{'seed':>4} {'gamma':>6} {'alpha':>7} {'coverage':>9} "
          f"{'AOG_classic':>12} {'AOG_cio':>9} {'AOG_impr%':>10} "
          f"{'POG_classic':>12} {'POG_cio':>9} {'POG_impr%':>10}")

    rows = []
    t0 = time.time()
    for seed in seeds:
        for gamma in gammas:
            r = run_once(seed, gamma, N_train, N_val, N_test)
            aog_impr = 100 * (r["aog_classic"] - r["aog_cio"]) / max(abs(r["aog_classic"]), 1e-9)
            pog_impr = 100 * (r["pog_classic"] - r["pog_cio"]) / max(abs(r["pog_classic"]), 1e-9)
            rows.append({**r, "aog_impr": aog_impr, "pog_impr": pog_impr})
            print(f"{seed:>4} {gamma:>6.2f} {r['alpha']:>7.3f} {r['coverage']:>9.3f} "
                  f"{r['aog_classic']:>12.4f} {r['aog_cio']:>9.4f} {aog_impr:>10.1f} "
                  f"{r['pog_classic']:>12.4f} {r['pog_cio']:>9.4f} {pog_impr:>10.1f}")
    elapsed = time.time() - t0
    print(f"\n{len(rows)} runs in {elapsed:.1f}s ({elapsed/len(rows):.2f}s/run)")

    print("\n--- Gate check summary ---")
    for gamma in gammas:
        covs = [r["coverage"] for r in rows if r["gamma"] == gamma]
        aog_imprs = [r["aog_impr"] for r in rows if r["gamma"] == gamma]
        pog_imprs = [r["pog_impr"] for r in rows if r["gamma"] == gamma]
        print(f"gamma={gamma:.2f}: mean coverage={np.mean(covs):.3f} (target {gamma:.2f}), "
              f"mean AOG improvement={np.mean(aog_imprs):.1f}% (std {np.std(aog_imprs):.1f}), "
              f"mean POG improvement={np.mean(pog_imprs):.1f}% (std {np.std(pog_imprs):.1f})")


if __name__ == "__main__":
    main()
