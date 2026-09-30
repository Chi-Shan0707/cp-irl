"""Does swapping min -> max in the decision-aware score upgrade INTERSECTION
coverage to genuine CONTAINMENT coverage?

The existing score (conformal/decision_metric.py::distance_to_cone) is
    s_k := min_{theta in Theta-hat(pi_k)} ||theta - theta_bar||_D,
which yields only  P[Theta-hat(pi_new) ∩ C_q != empty] >= gamma.

Replace the min by a max over the same feasible cone:
    w_k := max_{theta in Theta-hat(pi_k), ||theta||_2 <= 1} ||theta - theta_bar||_D.
Because the demonstrator acts optimally under its own theta_k, we have
theta_k in Theta-hat(pi_k), hence ||theta_k - theta_bar||_D <= w_k pointwise.
Split conformal on w_k gives P[w_new <= q] >= gamma, and w_new <= q implies
||theta_new - theta_bar||_D <= q. That is CONTAINMENT, not intersection.

CAVEAT (stated honestly): ||.||_D is convex, so w_k is a convex MAXIMIZATION --
not an LP. Its maximum is attained at an extreme point of the cone-cap. We
compute it by multistart alternating ascent, which returns a LOWER bound; a
lower bound does NOT preserve the validity implication (an upper bound would).
This script therefore measures two things separately:
  (1) the empirical containment coverage achieved by the computed w_k, and
  (2) how often the computed w_k actually dominates ||theta_k - theta_bar||_D,
      which is the assumption the ascent must meet to be sound.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_containment_score.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from conformal.decision_metric import (
    span_seminorm, distance_to_cone, farthest_in_cone,
)
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl
from experiments.run_containment_audit import (
    ENV_BUILDERS, _ENV_SEED_OFFSET, GAMMA_TARGET, gen_population,
)

ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=150),
}


def conformal_quantile(scores, gamma):
    N = len(scores)
    tau = int(np.ceil(gamma * (N + 1)))
    return float("inf") if tau == N + 1 else float(np.sort(scores)[tau - 1])


def run_once(env_name, est_name, seed, N_train, N_val, N_test):
    rng = np.random.default_rng(_ENV_SEED_OFFSET[env_name] + seed)
    mdp = ENV_BUILDERS[env_name](rng)
    theta_star = rng.uniform(0.5, 2.0, size=mdp.d)

    pols_tr, _ = gen_population(mdp, theta_star, N_train, rng)
    pols_val, _ = gen_population(mdp, theta_star, N_val, rng)
    pols_te, thetas_te = gen_population(mdp, theta_star, N_test, rng)

    theta_bar = ESTIMATORS[est_name](mdp, pols_tr)
    # Keep theta_bar inside the unit ball so the cone-cap can contain it.
    theta_bar = theta_bar / max(np.linalg.norm(theta_bar), 1.0)
    ascent_rng = np.random.default_rng(1000 + seed)

    w_val = np.array([farthest_in_cone(mdp, pi, theta_bar, rng=ascent_rng)
                      for pi in pols_val])
    s_val = np.array([distance_to_cone(mdp, pi, theta_bar)[0] for pi in pols_val])
    q_w = conformal_quantile(w_val, GAMMA_TARGET)
    q_s = conformal_quantile(s_val, GAMMA_TARGET)

    contain_w = contain_s = intersect_s = dominated = 0
    for pi_t, th_t in zip(pols_te, thetas_te):
        # Reward is identified only up to positive scale; match to theta_bar's
        # norm, as run_containment_audit.py does, before measuring distance.
        th_scaled = th_t / np.linalg.norm(th_t) * np.linalg.norm(theta_bar)
        t_k = span_seminorm(mdp, th_scaled - theta_bar)
        contain_w += int(t_k <= q_w + 1e-6)
        contain_s += int(t_k <= q_s + 1e-6)
        s_t, _ = distance_to_cone(mdp, pi_t, theta_bar)
        intersect_s += int(s_t <= q_s + 1e-6)
        w_t = farthest_in_cone(mdp, pi_t, theta_bar, rng=ascent_rng)
        dominated += int(w_t >= t_k - 1e-6)

    n = len(pols_te)
    return dict(env=env_name, estimator=est_name, seed=seed,
                q_s=q_s, q_w=q_w,
                containment_minscore=contain_s / n,
                containment_maxscore=contain_w / n,
                intersection_minscore=intersect_s / n,
                ascent_dominates=dominated / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--out", type=str,
                    default="experiments/containment_score_results.json")
    args = ap.parse_args()

    rows = []
    hdr = (f"{'env':>12} {'est':>7} {'q_s':>7} {'q_w':>7} "
           f"{'intersect(min)':>14} {'contain(min)':>12} {'contain(MAX)':>12} {'dom':>5}")
    print(hdr)
    for env_name in ENV_BUILDERS:
        for est_name in ESTIMATORS:
            for seed in range(args.seeds):
                rows.append(run_once(env_name, est_name, seed,
                                     args.n_train, args.n_val, args.n_test))
            r = [x for x in rows if x["env"] == env_name and x["estimator"] == est_name]
            m = lambda k: np.mean([x[k] for x in r])
            print(f"{env_name:>12} {est_name:>7} {m('q_s'):7.3f} {m('q_w'):7.3f} "
                  f"{m('intersection_minscore'):14.3f} {m('containment_minscore'):12.3f} "
                  f"{m('containment_maxscore'):12.3f} {m('ascent_dominates'):5.2f}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\ntarget gamma = {GAMMA_TARGET}\nwrote {args.out}")


if __name__ == "__main__":
    main()
