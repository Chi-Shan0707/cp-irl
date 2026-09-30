"""Phase 5 ablation (plan.md Sec 6): trajectory count per demonstrator -- the last
of the six ablation axes. Instead of observing each demonstrator's FULL policy,
observe only M rollout trajectories of length H, giving partial (visited-state-only)
observation. Uses irl/feasible_set.py::c_k_partial (T2_proof.md's "wrong object, not
wrong guarantee" relaxation: valid but less informative than full-policy
observation).

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_phase5_trajectory_ablation.py
"""
from __future__ import annotations

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.mdp import value_iteration, occupancy_lp
from irl.point_estimate import classic_irl
from irl.feasible_set import c_k_partial
from robust.mdp_robust import solve_robust_mdp

N_JOBS = 8


def rollout_trajectory(mdp, policy, H, rng):
    """Roll out H steps under `policy` (with the MDP's own transition stochasticity),
    starting from mu0. Returns dict {visited state -> observed action}."""
    s = rng.choice(mdp.S, p=mdp.mu0)
    visited = {}
    for _ in range(H):
        a = int(policy[s])
        visited[int(s)] = a
        s = rng.choice(mdp.S, p=mdp.P[s, a])
    return visited


def gen_population_partial(mdp, theta_star, N, M, H, rng, noise_std=0.5, eps0=0.05):
    """Each demonstrator's FULL policy is generated (for evaluating the *true*
    demonstrator behavior), but only M trajectories of length H are "observed"
    (visited states merged across all M rollouts)."""
    demo_visited = []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, pi = value_iteration(mdp, theta_hat)
        visited = {}
        for _ in range(M):
            visited.update(rollout_trajectory(mdp, pi, H, rng))
        demo_visited.append(visited)
    return demo_visited


def classic_irl_partial(mdp, demo_visited, normalize_l2=True):
    """classic_irl requires a full policy array; approximate a full policy from
    partial observations by filling unvisited states with a placeholder action 0
    (irrelevant -- classic_irl's LP only uses the occupancy measure of visited
    states in spirit, but our existing classic_irl takes full arrays, so we
    reconstruct a full policy using visited actions where known and a fixed
    default elsewhere -- this is a simplification noted in the ablation's honest
    reporting, not the theoretically ideal partial-trajectory point estimator)."""
    policies = []
    for visited in demo_visited:
        pi = np.zeros(mdp.S, dtype=int)
        for s, a in visited.items():
            pi[s] = a
        policies.append(pi)
    return classic_irl(mdp, policies, normalize_l2=normalize_l2)


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(seed, M, H, gamma_target=0.8, size=6, n_regions=4, feature_dim=4, N=30):
    rng = np.random.default_rng(seed)
    mdp, _ = build_region_gridworld(size=size, n_regions=n_regions,
                                     feature_dim=feature_dim, gamma=0.9, rng=rng)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    demo_train = gen_population_partial(mdp, theta_star, N, M, H, rng)
    demo_val = gen_population_partial(mdp, theta_star, N, M, H, rng)
    theta_bar = classic_irl_partial(mdp, demo_train)

    c_ks = np.array([c_k_partial(mdp, v, theta_bar) for v in demo_val])
    tau = int(np.ceil(gamma_target * (len(c_ks) + 1)))
    tau = min(max(tau, 1), len(c_ks))
    c_tau = np.clip(np.sort(c_ks)[::-1][tau - 1], -1.0, 1.0)
    alpha = np.arccos(c_tau)

    mean_visited_frac = np.mean([len(v) / mdp.S for v in demo_val])

    d_classic, _ = occupancy_lp(mdp, theta_bar)
    d_conformal, _ = solve_robust_mdp(mdp, theta_bar, alpha)
    _, v_star = occupancy_lp(mdp, theta_star_unit)

    aog_classic = v_star - occ_value(mdp, d_classic, theta_star_unit)
    aog_conformal = v_star - occ_value(mdp, d_conformal, theta_star_unit)
    return alpha, mean_visited_frac, aog_classic, aog_conformal


def main():
    configs = [(1, 3), (2, 5), (5, 8), (10, 15), (20, 30)]  # (M trajectories, H length)
    seeds = list(range(5))

    print(f"{'M':>3} {'H':>3} {'mean alpha':>11} {'mean visited_frac':>18} "
          f"{'mean AOG_classic':>17} {'mean AOG_conformal':>19}")
    for M, H in configs:
        alphas, fracs, aog_cs, aog_rs = [], [], [], []
        for seed in seeds:
            a, f, ac, ar = run_once(seed, M, H)
            alphas.append(a)
            fracs.append(f)
            aog_cs.append(ac)
            aog_rs.append(ar)
        print(f"{M:>3} {H:>3} {np.mean(alphas):>11.4f} {np.mean(fracs):>18.3f} "
              f"{np.mean(aog_cs):>17.4f} {np.mean(aog_rs):>19.4f}")


if __name__ == "__main__":
    main()
