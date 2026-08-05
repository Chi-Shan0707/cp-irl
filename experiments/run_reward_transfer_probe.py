"""Reward-transfer probe (paper/draft.md Sec 7, "Discussion and Limitations" --
flagged as an open question: "whether a feasible reward set calibrated on one MDP
transfers, with a degraded but still meaningful coverage guarantee, to a related
MDP with different dynamics").

This is NOT the full weighted-conformal-under-covariate-shift theorem the
Discussion sketches (that requires importance-weighting calibration by a task/MDP
density ratio, per Tibshirani et al. 2019, combined with Schlaginhaufen &
Kamgarpour's 2024 principal-angle transferability conditions -- a real piece of
follow-up theory, not attempted here). This IS the necessary FIRST empirical
step that theory should be judged against: does NAIVE (unweighted) transfer of a
conformal reward set from a source MDP to a target MDP with different transition
dynamics (but the SAME feature map, so the reward parametrization is meaningful
in both) break coverage, and by how much? If naive transfer already worked, the
weighted-conformal machinery would be unnecessary; if it breaks badly, that
motivates exactly the fix the Discussion proposes.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_reward_transfer_probe.py
"""
from __future__ import annotations

import numpy as np

from envs.gridworld import build_region_gridworld, _DELTAS, _cell_to_state, _clip
from envs.mdp import TabularMDP, value_iteration
from irl.point_estimate import classic_irl
from irl.feasible_set import c_k as c_k_fast
from conformal.calibrate import conformal_calibrate

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


def _build_shared_feature_pair(size, n_regions, feature_dim, gamma, rng, slip_source,
                                slip_target):
    """Builds two TabularMDPs sharing IDENTICAL phi (constructed once from the
    region layout, using the INTENDED (non-slip) successor's region feature --
    i.e. phi(s,a) does not depend on slip_prob at all) but different P (source vs
    target slip probability). This decouples the reward parametrization from the
    dynamics, isolating exactly the transfer question: build_region_gridworld's
    own phi is an EXPECTED successor feature weighted by slip probabilities, so
    two calls with different slip_prob give different phi even from the same
    region layout -- unsuitable for a transfer experiment where the reward
    meaning must stay fixed while only dynamics change."""
    S, A = size * size, 4

    seeds = rng.choice(S, size=n_regions, replace=False)
    seed_coords = np.array([(s // size, s % size) for s in seeds])
    region_id = np.zeros(S, dtype=int)
    for s in range(S):
        r, c = s // size, s % size
        dists = np.sum((seed_coords - np.array([r, c])) ** 2, axis=1)
        region_id[s] = np.argmin(dists)
    region_features = rng.normal(size=(n_regions, feature_dim))
    region_features /= np.linalg.norm(region_features, axis=1, keepdims=True)

    # phi from the INTENDED (deterministic) successor only -- independent of slip
    phi = np.zeros((S, A, feature_dim))
    for r in range(size):
        for c in range(size):
            s = _cell_to_state(r, c, size)
            for a, (dr, dc) in enumerate(_DELTAS):
                nr, nc = _clip(r + dr, c + dc, size)
                s_next = _cell_to_state(nr, nc, size)
                phi[s, a, :] = region_features[region_id[s_next]]

    def build_P(slip_prob):
        P = np.zeros((S, A, S))
        for r in range(size):
            for c in range(size):
                s = _cell_to_state(r, c, size)
                for a, (dr, dc) in enumerate(_DELTAS):
                    targets = []
                    nr, nc = _clip(r + dr, c + dc, size)
                    targets.append((1 - slip_prob, _cell_to_state(nr, nc, size)))
                    perp = [(dc, dr), (-dc, -dr)] if (dr, dc) != (0, 0) else [(0, 0), (0, 0)]
                    for (pdr, pdc) in perp:
                        pr, pc = _clip(r + pdr, c + pdc, size)
                        targets.append((slip_prob / 2, _cell_to_state(pr, pc, size)))
                    for prob, s_next in targets:
                        P[s, a, s_next] += prob
        return P

    mu0 = np.zeros(S)
    mu0[0] = 1.0
    mdp_source = TabularMDP(build_P(slip_source), phi, gamma, mu0)
    mdp_target = TabularMDP(build_P(slip_target), phi, gamma, mu0)
    return mdp_source, mdp_target


def run_once(seed, gamma_target, dynamics_shift, size=6, n_regions=4, feature_dim=4, N=30):
    """dynamics_shift: the SOURCE MDP uses slip_prob=0.1 (the project default);
    the TARGET MDP uses slip_prob = 0.1 + dynamics_shift. Both share the exact
    same phi (see _build_shared_feature_pair), so only the dynamics differ."""
    rng_layout = np.random.default_rng(seed)
    mdp_source, mdp_target = _build_shared_feature_pair(
        size, n_regions, feature_dim, 0.9, rng_layout, 0.1, 0.1 + dynamics_shift)
    assert np.array_equal(mdp_source.phi, mdp_target.phi)

    rng = np.random.default_rng(seed + 10_000)
    theta_star = rng.uniform(0.5, 2.0, size=feature_dim)

    pols_train_source = gen_population(mdp_source, theta_star, N, rng)
    pols_val_source = gen_population(mdp_source, theta_star, N, rng)
    theta_bar = classic_irl(mdp_source, pols_train_source)
    alpha, _ = conformal_calibrate(mdp_source, pols_val_source, theta_bar, gamma_target,
                                    n_jobs=N_JOBS)

    # in-distribution coverage check: fresh SOURCE-MDP population
    pols_cov_source = gen_population(mdp_source, theta_star, N, rng)
    c_ks_source = np.array([c_k_fast(mdp_source, pi, theta_bar) for pi in pols_cov_source])
    coverage_source = float(np.mean(c_ks_source >= np.cos(alpha) - 1e-9))

    # transfer coverage check: fresh TARGET-MDP population (different dynamics,
    # naive transfer -- theta_bar and alpha computed entirely from the source)
    pols_cov_target = gen_population(mdp_target, theta_star, N, rng)
    c_ks_target = np.array([c_k_fast(mdp_target, pi, theta_bar) for pi in pols_cov_target])
    coverage_target = float(np.mean(c_ks_target >= np.cos(alpha) - 1e-9))

    return coverage_source, coverage_target, alpha


def main():
    gamma_target = 0.8
    shifts = [0.0, 0.05, 0.1, 0.2, 0.3]
    seeds = list(range(8))

    print("Naive reward-set transfer probe: calibrate on a SOURCE MDP, check "
          "coverage on a TARGET MDP with perturbed transition dynamics but the "
          "SAME feature map (no reweighting applied).\n")
    print(f"{'dynamics_shift':>15} {'mean cov_source':>16} {'mean cov_target':>16} "
          f"{'mean coverage_drop':>19}")
    for shift in shifts:
        cov_s, cov_t, drops = [], [], []
        for seed in seeds:
            cs, ct, _ = run_once(seed, gamma_target, shift)
            cov_s.append(cs)
            cov_t.append(ct)
            drops.append(cs - ct)
        print(f"{shift:>15.2f} {np.mean(cov_s):>16.3f} {np.mean(cov_t):>16.3f} "
              f"{np.mean(drops):>19.3f}")

    print(f"\n(target gamma = {gamma_target}; coverage_drop = source coverage minus "
          "target coverage -- positive means naive transfer under-covers relative "
          "to what was calibrated for)")


if __name__ == "__main__":
    main()
