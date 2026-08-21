"""Diagnostic probe for the inverse-fiber width needed by Stage II.

For each validation demonstrator, this script compares the tractable normalized
cone score s_k with the intrinsic projective latent distance t_k. The residual
max(t_k-2*s_k, 0) is a sample lower bound on the eta needed by the bridge
t_k <= 2*s_k+eta. It is not a certificate: eta must be supplied by external
structure or a valid uniform bound before the safe-prescription theorem applies.

All normalization uses the occupancy span, so the probe is invariant to
positive reward scale and invertible feature reparameterization.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_cone_width_probe.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from conformal.decision_metric import (
    normalize_span,
    normalized_distance_to_cone,
    projective_span_distance,
)
from experiments.run_containment_audit import (
    ENV_BUILDERS, _ENV_SEED_OFFSET, GAMMA_TARGET, gen_population,
)
from irl.point_estimate import classic_irl
from irl.maxent import maxent_irl


ESTIMATORS = {
    "LP-IRL": lambda mdp, pols: classic_irl(mdp, pols),
    "MaxEnt": lambda mdp, pols: maxent_irl(mdp, pols, n_iters=150),
}


def conformal_quantile(scores, gamma):
    N = len(scores)
    tau = int(np.ceil(gamma * (N + 1)))
    return float("inf") if tau == N + 1 else float(np.sort(scores)[tau - 1])


def run_once(env_name, est_name, seed, N_train, N_val):
    rng = np.random.default_rng(_ENV_SEED_OFFSET[env_name] + seed)
    mdp = ENV_BUILDERS[env_name](rng)
    theta_star = rng.uniform(0.5, 2.0, size=mdp.d)

    pols_train, _ = gen_population(mdp, theta_star, N_train, rng)
    pols_val, thetas_val = gen_population(mdp, theta_star, N_val, rng)

    theta_bar = ESTIMATORS[est_name](mdp, pols_train)
    theta_bar_span = normalize_span(mdp, theta_bar)

    s_ks, t_ks, bridge_residuals = [], [], []
    for pi_k, theta_k in zip(pols_val, thetas_val):
        s_k, _, _ = normalized_distance_to_cone(mdp, pi_k, theta_bar)
        # Compare intrinsic unit-span reward rays.  Euclidean matching would
        # change under an invertible feature reparameterization.
        theta_k_scaled = normalize_span(mdp, theta_k)
        s_ks.append(s_k)
        t_k = projective_span_distance(mdp, theta_k_scaled, theta_bar_span)
        t_ks.append(t_k)
        # The tractable cone-score bridge has t_k <= 2*s_k + eta.  This is only
        # the residual required by the sampled latent reward, hence a diagnostic
        # lower bound on the necessary fiber width, not a certificate.
        bridge_residuals.append(max(t_k - 2.0 * s_k, 0.0))

    s_ks, t_ks = np.array(s_ks), np.array(t_ks)
    # Sanity: s_k <= t_k must hold for every k, by theta_k in Theta-hat(pi_k).
    # (Checked on the scale-matched theta_k, which stays in the cone since the
    # cone is scale-invariant.)
    violations = int(np.sum(s_ks > t_ks + 1e-6))

    q_s = conformal_quantile(s_ks, GAMMA_TARGET)
    q_t = conformal_quantile(t_ks, GAMMA_TARGET)
    return dict(
        env=env_name, estimator=est_name, seed=seed,
        q_s=q_s, q_t=q_t,
        ratio=(q_t / q_s) if q_s > 1e-9 else float("inf"),
        mean_s=float(s_ks.mean()), mean_t=float(t_ks.mean()),
        max_sample_bridge_residual=float(np.max(bridge_residuals)),
        containment_at_q_s=float(np.mean(t_ks <= q_s + 1e-6)),
        order_violations=violations,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--out", type=str,
                    default="experiments/cone_width_probe_results.json")
    args = ap.parse_args()

    rows = []
    for env_name in ENV_BUILDERS:
        for est_name in ESTIMATORS:
            for seed in range(args.seeds):
                rows.append(run_once(env_name, est_name, seed,
                                     args.n_train, args.n_val))
            r = [x for x in rows if x["env"] == env_name
                 and x["estimator"] == est_name]
            print(f"{env_name:>12} {est_name:>8}  "
                  f"q_s={np.mean([x['q_s'] for x in r]):8.4f}  "
                  f"q_t={np.mean([x['q_t'] for x in r]):8.4f}  "
                  f"inflation={np.mean([x['ratio'] for x in r]):6.2f}x  "
                  f"bridge-residual={np.mean([x['max_sample_bridge_residual'] for x in r]):6.3f}  "
                  f"containment@q_s={np.mean([x['containment_at_q_s'] for x in r]):.3f}  "
                  f"order_viol={sum(x['order_violations'] for x in r)}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
