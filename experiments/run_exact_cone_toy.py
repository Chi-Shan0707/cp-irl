"""Post-acceptance supplement: exact cone certificates in a one-state MDP.

The regular-polygon features make each unit-span Bellman cone a line segment.
Its endpoints give exact farthest scores and diameters; random reward samples
are used only for calibration/evaluation, never to certify the cone width.
See paper/post_acceptance_toy.tex and paper/POST_ACCEPTANCE_CHANGES.md.

Run in rlenv: PYTHONPATH=. python experiments/run_exact_cone_toy.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from conformal.decision_metric import distance_to_cone
from envs.mdp import TabularMDP
from robust.decision_aware_robust import (
    solve_robust_mdp_decision_aware,
    solve_safe_robust_mdp,
)


def polygon_mdp(n_actions: int, discount: float = 0.9) -> TabularMDP:
    if n_actions < 4 or n_actions % 4:
        raise ValueError("n_actions must be a positive multiple of four, at least four")
    angles = 2 * np.pi * np.arange(n_actions) / n_actions
    features = np.column_stack((np.cos(angles), np.sin(angles)))
    return TabularMDP(np.ones((1, n_actions, 1)), features[None],
                      discount, np.ones(1))


def span(mdp: TabularMDP, rewards: np.ndarray) -> np.ndarray:
    """Independent exact finite-action formula, also vectorized over rewards."""
    values = np.asarray(rewards) @ mdp.phi[0].T / (1 - mdp.gamma)
    return np.ptp(values, axis=-1)


def cone_endpoints(mdp: TabularMDP, action: int) -> np.ndarray:
    half_angle = np.pi / mdp.A
    angles = 2 * half_angle * action + np.array([-half_angle, half_angle])
    return ((1 - mdp.gamma) / (2 * np.cos(half_angle))
            * np.column_stack((np.cos(angles), np.sin(angles))))


def cone_score_lp(mdp: TabularMDP, center: np.ndarray, action: int) -> float:
    """Independent SciPy LP: two reward coordinates and a span epigraph."""
    phi = mdp.phi[0]
    normals = 2 * phi / (1 - mdp.gamma)  # central symmetry gives span = max normals @ v
    constraints = np.vstack((np.column_stack((normals, -np.ones(mdp.A))),
                             np.column_stack((phi - phi[action], np.zeros(mdp.A)))))
    rhs = np.concatenate((normals @ center, np.zeros(mdp.A)))
    result = linprog([0., 0., 1.], A_ub=constraints, b_ub=rhs,
                     bounds=[(None, None), (None, None), (0, None)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return max(0., float(result.fun))


def ball_worst_advantage(mdp, center, radius, occupancy, reference):
    """Independent inner LP audits the library's reported robust objective."""
    feature_difference = ((occupancy - reference).ravel() @ mdp.phi[0]
                          / (1 - mdp.gamma))
    result = linprog(feature_difference, A_ub=2 * mdp.phi[0] / (1 - mdp.gamma),
                     b_ub=np.full(mdp.A, radius), bounds=[(None, None)] * 2,
                     method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(center @ feature_difference + result.fun)


def exact_geometry(mdp, center):
    endpoints = np.array([cone_endpoints(mdp, j) for j in range(mdp.A)])
    minimum = np.array([cone_score_lp(mdp, center, j) for j in range(mdp.A)])
    maximum = span(mdp, endpoints - center).max(axis=1)
    # For m divisible by four the diameter is attained in an action direction.
    eta = 2. if mdp.A == 4 else float(2 * np.tan(np.pi / mdp.A))
    np.testing.assert_allclose(span(mdp, endpoints), 1., atol=1e-12)
    np.testing.assert_allclose(span(mdp, endpoints[:, 0] - endpoints[:, 1]),
                               eta, atol=1e-12)
    for j in range(mdp.A):
        values = endpoints[j] @ mdp.phi[0].T
        if np.max(values - values[:, j, None]) > 1e-12:
            raise AssertionError("an analytic endpoint violates its Bellman cone")
        library_score, _ = distance_to_cone(mdp, np.array([j]), center)
        np.testing.assert_allclose(library_score, minimum[j], atol=2e-6)
    if np.any(maximum > 2 * minimum + eta + 1e-10):
        raise AssertionError("pointwise radius bridge failed")
    return endpoints, minimum, maximum, eta


def quantile(scores, target):
    rank = int(np.ceil(target * (len(scores) + 1)))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


def population(mdp, count, rng, control=False):
    actions = (np.zeros(count, dtype=int) if control else
               rng.choice([-1, 0, 1], size=count, p=[.25, .5, .25]) % mdp.A)
    # Strictly inside the cone: no optimal-action ties in the sampled population.
    angles = 2 * np.pi / mdp.A * (actions + rng.uniform(-.45, .45, count))
    rewards = np.column_stack((np.cos(angles), np.sin(angles)))
    rewards /= span(mdp, rewards)[:, None]
    np.testing.assert_array_equal((rewards @ mdp.phi[0].T).argmax(axis=1), actions)
    return actions, rewards


def run_case(n_actions, trials, n_calibration, n_test, target, seed, control=False):
    mdp = polygon_mdp(n_actions)
    # Prespecified independently of every calibration and test draw. No estimator
    # comparison is intended; this isolates the calibration-to-decision bridge.
    center = np.array([(1 - mdp.gamma) / 2, 0.])
    reference = np.full((1, mdp.A), 1 / mdp.A)
    endpoints, minimum, maximum, eta = exact_geometry(mdp, center)
    policy_cache = {}
    rows = []
    for trial in range(trials):
        cal_rng, test_rng = [np.random.default_rng(s) for s in
                             np.random.SeedSequence([seed, n_actions, trial]).spawn(2)]
        cal_actions, cal_rewards = population(mdp, n_calibration, cal_rng, control)
        test_actions, test_rewards = population(mdp, n_test, test_rng, control)
        q = quantile(minimum[cal_actions], target)
        q_max = quantile(maximum[cal_actions], target)
        # Oracle uses latent CALIBRATION rewards only; test rewards never set a radius.
        q_oracle = quantile(span(mdp, cal_rewards - center), target)
        radius = min(2., 2 * q + eta)
        if not q_oracle <= q_max + 1e-10 or not q_max <= radius + 1e-10:
            raise AssertionError("oracle <= exact maximum <= width bridge failed")
        cache_key = (q, q_max)
        if cache_key not in policy_cache:
            safe = solve_safe_robust_mdp(mdp, center, q, eta, reference)
            # Use the same explicit reference tie-break at R=1 as the safe wrapper.
            exact_occ, exact_value = ((reference.copy(), 0.) if q_max >= 1 - 1e-10
                else solve_robust_mdp_decision_aware(mdp, center, q_max, reference))
            for r, occ, value in [(radius, safe.occupancy, safe.worst_case_advantage),
                                   (q_max, exact_occ, exact_value)]:
                np.testing.assert_allclose(ball_worst_advantage(mdp, center, r, occ, reference),
                                           value, atol=2e-6)
            policy_cache[cache_key] = (safe.occupancy, safe.worst_case_advantage,
                                      exact_occ, exact_value)
        safe_occ, safe_value, exact_occ, exact_value = policy_cache[cache_key]
        test_distances = span(mdp, test_rewards - center)
        safe_advantages = test_rewards @ ((safe_occ - reference).ravel() @ mdp.phi[0]) / (1 - mdp.gamma)
        rows.append(dict(
            trial=trial, intersection_radius=q, exact_max_radius=q_max,
            oracle_radius=q_oracle, bridge_radius=radius,
            intersection_coverage=float(np.mean(minimum[test_actions] <= q + 1e-9)),
            stage1_containment=float(np.mean(test_distances <= q + 1e-9)),
            exact_containment=float(np.mean(test_distances <= q_max + 1e-9)),
            oracle_containment=float(np.mean(test_distances <= q_oracle + 1e-9)),
            bridge_containment=float(np.mean(test_distances <= radius + 1e-9)),
            below_degeneracy_threshold=bool(radius < 1),
            positive_worst_case_advantage=bool(safe_value > 1e-7),
            bridge_worst_case_advantage=float(safe_value),
            exact_worst_case_advantage=float(exact_value),
            reference_feature_distance=float(np.linalg.norm((safe_occ - reference).ravel() @ mdp.phi[0])),
            reference_dominance=float(np.mean(safe_advantages >= -1e-7)),
            minimum_test_advantage=float(safe_advantages.min()),
        ))
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "trial"}
    standard_errors = {
        k: float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(trials))
        for k in summary
    } if trials > 1 else {}
    result = dict(n_actions=n_actions, discount=mdp.gamma, feature_dimension=2,
                  center=center.tolist(), reference=reference.ravel().tolist(),
                  policy_support=[0] if control else [n_actions - 1, 0, 1],
                  certified_width=eta, cone_min_scores=minimum.tolist(),
                  cone_max_scores=maximum.tolist(), endpoints=endpoints.tolist(),
                  summary=summary, standard_errors_over_trials=standard_errors,
                  policies=[dict(intersection_radius=q, exact_max_radius=q_max,
                                 bridge_occupancy=values[0].ravel().tolist(),
                                 exact_occupancy=values[2].ravel().tolist())
                            for (q, q_max), values in policy_cache.items()], trials=rows)
    if control:
        # In the control every demonstrator chooses action 0. Optimize over that
        # entire normalized feasible cone directly: two endpoints suffice for
        # every linear policy-value objective, with no latent-reward oracle.
        returns = endpoints[0] @ mdp.phi[0].T / (1 - mdp.gamma)
        ref_returns = returns @ reference.ravel()
        lp = linprog(np.r_[np.zeros(mdp.A), -1.],
                     A_ub=np.column_stack((-returns, np.ones(2))), b_ub=-ref_returns,
                     A_eq=np.array([np.r_[np.ones(mdp.A), 0.]]), b_eq=[1.],
                     bounds=[(0, None)] * mdp.A + [(None, None)], method="highs")
        if not lp.success:
            raise RuntimeError(lp.message)
        result["direct_cone_worst_case_advantage"] = float(-lp.fun)
        result["direct_cone_occupancy"] = lp.x[:-1].tolist()
        np.testing.assert_allclose(-lp.fun, .5, atol=1e-10)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-calibration", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--target", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--out", type=Path, default=Path("experiments/exact_cone_toy_results.json"))
    args = parser.parse_args()
    if min(args.trials, args.n_calibration, args.n_test) < 1 or not 0 < args.target < 1:
        parser.error("sample counts must be positive and target must lie in (0,1)")
    if np.ceil(args.target * (args.n_calibration + 1)) > args.n_calibration:
        parser.error("this finite-radius diagnostic requires a nonsentinel quantile")
    config = {k: v for k, v in vars(args).items() if k != "out"}
    cases = [run_case(m, args.trials, args.n_calibration, args.n_test, args.target,
                      args.seed, control=control) for m, control in [(16, False), (4, True)]]
    output = dict(schema_version="exact-cone-toy-v1", status="post-acceptance supplement",
                  config=config, cases=cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(json.dumps({"n_actions": case["n_actions"], "eta": case["certified_width"],
                          **case["summary"]}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
