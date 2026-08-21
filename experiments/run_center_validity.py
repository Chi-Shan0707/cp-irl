"""Audit whether a point estimator defines a projective reward center.

The unit-span construction is defined only for rewards whose occupancy span is
positive.  Exact-fit IRL objectives can select an additive-null reward even when
the demonstrations are informative, because all zero-loss optima are tied.  This
script measures that failure rather than silently adding a coordinate-dependent
epsilon or changing the estimator after seeing the result.

Run:
    PYTHONPATH=. python experiments/run_center_validity.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from conformal.decision_metric import span_seminorm
from experiments.run_containment_audit import (
    ENV_BUILDERS,
    _ENV_SEED_OFFSET,
    gen_population,
)
from irl.point_estimate import classic_irl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--out", default="experiments/center_validity_results.json"
    )
    args = parser.parse_args()

    rows = []
    for env_name, builder in ENV_BUILDERS.items():
        for seed in range(args.seeds):
            rng = np.random.default_rng(_ENV_SEED_OFFSET[env_name] + seed)
            mdp = builder(rng)
            theta_star = rng.uniform(0.5, 2.0, size=mdp.d)
            policies, _ = gen_population(mdp, theta_star, args.n_train, rng)
            theta_bar = classic_irl(mdp, policies)
            span = span_seminorm(mdp, theta_bar)
            rows.append(
                {
                    "schema_version": "projective-center-v1",
                    "env": env_name,
                    "seed": seed,
                    "estimator": "LP-IRL",
                    "n_train": args.n_train,
                    "occupancy_span": span,
                    "projective_center_valid": bool(span > args.atol),
                    "atol": args.atol,
                }
            )

    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=2)

    print("projective-center validity of exact-fit LP-IRL")
    for env_name in ENV_BUILDERS:
        subset = [r for r in rows if r["env"] == env_name]
        valid = sum(r["projective_center_valid"] for r in subset)
        print(f"{env_name:>12}: {valid}/{len(subset)} valid")
    total = sum(r["projective_center_valid"] for r in rows)
    print(f"{'pooled':>12}: {total}/{len(rows)} valid")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
