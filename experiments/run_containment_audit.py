"""Two-stage CIO-style audit for CP-IRL.

Stage I span-normalizes the fitted reward, calibrates its LP distance to each
Bellman cone, and checks the exact intersection event. Stage II accepts a
*certified* normalized inverse-fiber diameter eta, forms the expanded radius R,
and solves the reference-relative safe robust LP. Outputs keep intersection,
projective latent containment, reference dominance, and R<1 nontriviality
separate.

The default eta=2 is the universal bound and is intentionally vacuous: it is a
framework test, not evidence of improvement. Angular and value-gap methods are
retained only as legacy baselines; their Euclidean geometry does not certify
Stage II.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_containment_audit.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from envs.gridworld import build_region_gridworld
from envs.objectworld import build_objectworld
from envs.mdp import random_mdp, value_iteration, occupancy_lp, occupancy_of_policy
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl
from irl.bayesian_irl import bayesian_irl
from irl.feasible_set import c_k as angular_c_k
from conformal.calibrate import conformal_calibrate
from conformal.value_gap import calibrate_value_gap, value_gap_score, ball_radius
from conformal.decision_metric import (
    calibrate_normalized_decision_aware,
    containment_radius_from_fiber_width,
    distance_to_cone,
    normalize_span,
    normalized_distance_to_cone,
    span_match_reward,
    span_seminorm,
)
from robust.mdp_robust import solve_robust_mdp
from robust.value_ball_robust import solve_robust_mdp_ball
from robust.decision_aware_robust import (
    mean_population_occupancy,
    solve_robust_mdp_decision_aware,
    solve_safe_robust_mdp,
)

N_JOBS = 8
GAMMA_TARGET = 0.8

POINT_ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=150),
    "Bayesian": lambda mdp, pols: bayesian_irl(mdp, pols, n_samples=400, burn_in=150,
                                                rng=np.random.default_rng(0)),
}


def build_random_mdp_env(rng):
    return random_mdp(6, 3, 3, gamma=0.9, rng=rng)


def build_gridworld_env(rng):
    mdp, _ = build_region_gridworld(size=6, n_regions=4, feature_dim=4, gamma=0.9, rng=rng)
    return mdp


def build_objectworld_env(rng):
    return build_objectworld(size=6, n_colors=4, n_objects_per_color=2, gamma=0.9, rng=rng)


ENV_BUILDERS = {
    "random_mdp": build_random_mdp_env,
    "gridworld": build_gridworld_env,
    "objectworld": build_objectworld_env,
}
_ENV_SEED_OFFSET = {"random_mdp": 0, "gridworld": 100_000, "objectworld": 200_000}


def gen_population(mdp, theta_star, N, rng, noise_std=0.5, eps0=0.05):
    policies, thetas = [], []
    for _ in range(N):
        p = rng.uniform(0.5, 2.0, size=theta_star.shape)
        eps = rng.normal(0, noise_std, size=theta_star.shape)
        theta_hat = np.maximum(theta_star * p + eps, 0.0) + eps0
        _, _, pi = value_iteration(mdp, theta_hat)
        policies.append(pi)
        thetas.append(theta_hat)
    return policies, thetas


def occ_value(mdp, d_occ, theta):
    R = mdp.reward(theta)
    return float(d_occ.flatten() @ R.flatten()) / (1 - mdp.gamma)


def run_once(env_name, seed, N_train, N_val, N_test, fiber_diameter=2.0,
             estimator_names=None):
    rng = np.random.default_rng(_ENV_SEED_OFFSET[env_name] + seed)
    mdp = ENV_BUILDERS[env_name](rng)
    theta_star = rng.uniform(0.5, 2.0, size=mdp.d)
    # All decision-aware evaluation uses an intrinsic unit-span representative.
    # Euclidean normalization remains only inside the explicitly coordinate-
    # dependent angular/value-ball baselines.
    theta_star_span = normalize_span(mdp, theta_star)

    pols_train, _ = gen_population(mdp, theta_star, N_train, rng)
    pols_val, _ = gen_population(mdp, theta_star, N_val, rng)
    pols_test, thetas_test = gen_population(mdp, theta_star, N_test, rng)

    _, v_star = occupancy_lp(mdp, theta_star_span)
    mu_ref = mean_population_occupancy(mdp, pols_val)

    rows = []
    if estimator_names is None:
        estimator_names = tuple(POINT_ESTIMATORS)
    for est_name in estimator_names:
        estimator = POINT_ESTIMATORS[est_name]
        theta_bar = estimator(mdp, pols_train)
        theta_bar_unit = theta_bar / max(np.linalg.norm(theta_bar), 1e-12)

        # ---- calibrate all three scores on the SAME validation population ----
        alpha_conf, c_ks_ang = conformal_calibrate(mdp, pols_val, theta_bar_unit,
                                                     GAMMA_TARGET, n_jobs=N_JOBS)
        q_vg, _ = calibrate_value_gap(mdp, pols_val, theta_bar, GAMMA_TARGET)
        rho_vg = ball_radius(q_vg, mdp)
        q_da, _, theta_bar_span = calibrate_normalized_decision_aware(
            mdp, pols_val, theta_bar, GAMMA_TARGET
        )
        containment_radius = containment_radius_from_fiber_width(
            q_da, fiber_diameter, normalized_fiber_score=False
        )

        # ---- robust policies ----
        d_point, _ = occupancy_lp(mdp, theta_bar)
        d_angular, _ = solve_robust_mdp(mdp, theta_bar_unit, alpha_conf)
        rho_vg_capped = rho_vg if np.isfinite(rho_vg) else 10.0 * np.linalg.norm(theta_bar)
        d_ball, _ = solve_robust_mdp_ball(mdp, theta_bar, rho_vg_capped)
        # Stage I deployment uses the intersection set and has no latent safety
        # interpretation.  The normalized sentinel q=1 is already behaviorally
        # uninformative and triggers the reference-degeneracy boundary.
        q_da_capped = q_da if np.isfinite(q_da) else 1.0
        d_da, _ = solve_robust_mdp_decision_aware(
            mdp, theta_bar_span, q_da_capped, mu_ref
        )
        # Stage II uses the fiber-width-inflated containment radius.  With the
        # universal certified eta=2 this deliberately returns the reference;
        # pass a smaller *certified* eta only when the application supplies one.
        safe_result = solve_safe_robust_mdp(
            mdp,
            theta_bar,
            q_da,
            fiber_diameter,
            mu_ref,
            normalized_fiber_score=False,
        )
        d_safe = safe_result.occupancy

        aog = {
            "point_estimate": v_star - occ_value(mdp, d_point, theta_star_span),
            "angular_cap": v_star - occ_value(mdp, d_angular, theta_star_span),
            "value_ball": v_star - occ_value(mdp, d_ball, theta_star_span),
        }
        # decision-aware solver targets RELATIVE value vs mu_ref, not absolute
        # occupancy directly comparable to the others' AOG in the same units --
        # still perfectly fine to score its OWN occupancy d_da against theta_star:
        aog["decision_aware"] = v_star - occ_value(mdp, d_da, theta_star_span)
        aog["decision_safe"] = v_star - occ_value(mdp, d_safe, theta_star_span)

        # ---- E2: calibration-event coverage vs true containment, on fresh draws ----
        # For angular and decision-aware scores the event is exactly feasible-set
        # intersection.  The value-gap event is only score coverage: converting
        # its threshold to a Euclidean ball inverts a one-way Lipschitz bound.
        ang_intersect = ang_contain = vg_score_event = vg_contain = 0
        da_intersect = da_contain = da_expanded_contain = 0
        safe_reference_dominance = 0
        ceiling_violations = 0
        # Reward is only identified up to POSITIVE SCALE (theta and c*theta, c>0,
        # induce the same optimal policy -- gen_population's multiplicative noise
        # `p in [0.5, 2.0]` means raw theta_t and theta_bar routinely differ by a
        # 4x scale factor that carries NO behavioral information whatsoever).
        # Comparing raw theta_t against theta_bar therefore conflates this
        # scale-identifiability gap with the direction/shaping gap the sets are
        # actually meant to calibrate. We scale-match theta_t to theta_bar's own
        # norm before checking Euclidean/decision-aware containment (the angular
        # check is already scale-free by construction, both sides unit-normalized).
        for pol_t, theta_t in zip(pols_test, thetas_test):
            theta_t_unit = theta_t / np.linalg.norm(theta_t)
            theta_t_scaled = theta_t * (np.linalg.norm(theta_bar) / np.linalg.norm(theta_t))
            theta_t_span = span_match_reward(mdp, theta_t, theta_bar_span)

            c_t = angular_c_k(mdp, pol_t, theta_bar_unit)
            ang_intersect += int(c_t >= np.cos(alpha_conf) - 1e-6)
            ang_contain += int(theta_t_unit @ theta_bar_unit >= np.cos(alpha_conf) - 1e-6)

            g_t = value_gap_score(mdp, pol_t, theta_bar)
            vg_score_event += int(g_t <= q_vg + 1e-6)
            vg_contain += int(np.linalg.norm(theta_t_scaled - theta_bar) <= rho_vg + 1e-6
                               if np.isfinite(rho_vg) else True)

            s_t, _, _ = normalized_distance_to_cone(mdp, pol_t, theta_bar)
            da_intersect += int(s_t <= q_da + 1e-6)
            span_t_scaled = span_seminorm(mdp, theta_t_span - theta_bar_span)
            da_contain += int(span_t_scaled <= q_da + 1e-6 if np.isfinite(q_da) else True)
            da_expanded_contain += int(span_t_scaled <= containment_radius + 1e-6)

            safe_value = occ_value(mdp, d_safe, theta_t)
            ref_value = occ_value(mdp, mu_ref, theta_t)
            safe_reference_dominance += int(safe_value + 1e-6 >= ref_value)

            # Prop. 1 regret bound (the "ceiling"), Reg_{theta_t}(pi_thetabar) <= ||theta_t-theta_bar||_D.
            # MUST use the RAW (not scale-matched) theta_t here: regret is
            # scale-SENSITIVE (Reg_{c*theta} = c*Reg_theta), unlike the
            # scale-identifiability question containment above is isolating.
            span_t_raw = span_seminorm(mdp, theta_t - theta_bar)
            _, v_star_t = occupancy_lp(mdp, theta_t)
            reg_point_t = v_star_t - occ_value(mdp, d_point, theta_t)
            if reg_point_t > span_t_raw + 1e-4:
                ceiling_violations += 1

        n_test = len(pols_test)
        rows.append(dict(
            schema_version="projective-v2", gamma=GAMMA_TARGET,
            n_train=N_train, n_val=N_val, n_test_split=N_test,
            env=env_name, seed=seed, estimator=est_name,
            alpha_angular=alpha_conf, q_value_gap=q_vg, rho_value_gap=rho_vg,
            q_decision_aware=q_da, fiber_diameter=fiber_diameter,
            containment_radius=containment_radius,
            safe_nontrivial=safe_result.nontrivial,
            aog_point=aog["point_estimate"], aog_angular=aog["angular_cap"],
            aog_value_ball=aog["value_ball"], aog_decision_aware=aog["decision_aware"],
            aog_decision_safe=aog["decision_safe"],
            angular_intersect_cov=ang_intersect / n_test, angular_containment=ang_contain / n_test,
            value_gap_score_event_cov=vg_score_event / n_test, value_gap_containment=vg_contain / n_test,
            decision_aware_intersect_cov=da_intersect / n_test,
            decision_aware_containment=da_contain / n_test,
            decision_aware_expanded_containment=da_expanded_contain / n_test,
            safe_reference_dominance=safe_reference_dominance / n_test,
            ceiling_violations=ceiling_violations, n_test=n_test,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument(
        "--estimators", type=str, default="MaxEnt,Bayesian",
        help=("Comma-separated point estimators. The projective audit defaults "
              "to MaxEnt and Bayesian because the exact-fit LP estimator can "
              "return a behaviorally-null center, which has no unit-span "
              "representative. Available: LP-IRL, MaxEnt, Bayesian."),
    )
    ap.add_argument(
        "--fiber-diameter", type=float, default=2.0,
        help=("Certified upper bound eta in projective span distance. "
              "The universal eta=2 is valid but intentionally degenerate."),
    )
    ap.add_argument("--out", type=str, default="experiments/containment_audit_results.json")
    args = ap.parse_args()

    estimator_names = tuple(x.strip() for x in args.estimators.split(",") if x.strip())
    unknown = sorted(set(estimator_names) - set(POINT_ESTIMATORS))
    if not estimator_names or unknown:
        ap.error(
            "--estimators must name one or more of LP-IRL, MaxEnt, Bayesian; "
            f"unknown={unknown}"
        )

    t0 = time.time()
    all_rows = []
    for env_name in ENV_BUILDERS:
        for seed in range(args.n_seeds):
            all_rows.extend(run_once(
                env_name, seed, args.n_train, args.n_val, args.n_test,
                fiber_diameter=args.fiber_diameter,
                estimator_names=estimator_names,
            ))
    elapsed = time.time() - t0

    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=1)
    print(f"Total: {elapsed:.1f}s for {len(ENV_BUILDERS)}x{args.n_seeds} seeds "
          f"x{len(estimator_names)} estimators "
          f"({len(all_rows)} rows) -> {args.out}\n")

    print("=== E2: calibration-event coverage vs TRUE CONTAINMENT (target = "
          f"{GAMMA_TARGET}) ===")
    print(f"{'construction':>16} {'event cov':>14} {'containment':>13}")
    for name, ikey, ckey in [
        ("angular cap", "angular_intersect_cov", "angular_containment"),
        ("value-gap ball", "value_gap_score_event_cov", "value_gap_containment"),
        ("decision-aware", "decision_aware_intersect_cov", "decision_aware_containment"),
    ]:
        i_mean = np.mean([r[ikey] for r in all_rows])
        c_mean = np.mean([r[ckey] for r in all_rows])
        print(f"{name:>16} {i_mean:>14.3f} {c_mean:>13.3f}")

    total_test = sum(r["n_test"] for r in all_rows)
    total_viol = sum(r["ceiling_violations"] for r in all_rows)
    print(f"\n=== Prop. 1 regret-bound (ceiling) violations: {total_viol}/{total_test} "
          "(expect 0) ===\n")

    print("=== Stage II: fiber-width containment and reference safety ===")
    mean_R = np.mean([r["containment_radius"] for r in all_rows])
    nontrivial = np.mean([r["safe_nontrivial"] for r in all_rows])
    expanded_cov = np.mean([
        r["decision_aware_expanded_containment"] for r in all_rows
    ])
    safe_cov = np.mean([r["safe_reference_dominance"] for r in all_rows])
    print(f"eta={args.fiber_diameter:.3f}  mean R={mean_R:.3f}  "
          f"R<1 fraction={nontrivial:.3f}  expanded containment={expanded_cov:.3f}  "
          f"reference dominance={safe_cov:.3f}\n")

    print("=== AOG by env x estimator (point / angular / value_ball / decision_aware) ===")
    combos = sorted(set((r["env"], r["estimator"]) for r in all_rows))
    for env, est in combos:
        subset = [r for r in all_rows if r["env"] == env and r["estimator"] == est]
        p = np.mean([r["aog_point"] for r in subset])
        a = np.mean([r["aog_angular"] for r in subset])
        v = np.mean([r["aog_value_ball"] for r in subset])
        d = np.mean([r["aog_decision_aware"] for r in subset])
        q_da = np.mean([r["q_decision_aware"] for r in subset if np.isfinite(r["q_decision_aware"])])
        ds = np.mean([r["aog_decision_safe"] for r in subset])
        print(f"{env:>12} {est:>9}  point={p:8.4f}  angular={a:8.4f}  "
              f"value_ball={v:8.4f}  decision_aware={d:8.4f}  safe={ds:8.4f} "
              f"  (mean q_da={q_da:.3f})")


if __name__ == "__main__":
    main()
