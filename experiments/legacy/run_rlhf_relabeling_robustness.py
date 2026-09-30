"""Robustness check for the RLHF-relabeling claim in paper/draft.md's Discussion
section ("the deployed policy is aligned with a gamma-fraction of the annotator
population's own unobserved preferences"), which is currently supported by a
single configuration (6 responses, 4 attributes, 8 seeds; coverage 0.947-0.994).

A reviewer concern: that single-config, 8-seed result is thin support for a claim
phrased as a general property of the relabeling. This script reruns
run_rlhf_relabeling_demo.py's exact pipeline (unchanged) across MULTIPLE
(n_responses, n_attributes, N) configurations and MORE seeds, to check whether
the empirical coverage >= gamma pattern holds broadly or was a lucky config.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/legacy/run_rlhf_relabeling_robustness.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_rlhf_relabeling_demo import run_once  # noqa: E402  (sibling in legacy/)

CONFIGS = [
    dict(n_responses=6, n_attributes=4, N=40),   # original config
    dict(n_responses=4, n_attributes=3, N=25),   # smaller population, fewer options
    dict(n_responses=10, n_attributes=6, N=60),  # larger response set, higher-dim preference
    dict(n_responses=6, n_attributes=4, N=15),   # original config, small-sample regime
]


def main():
    seeds = list(range(30))
    gammas = [0.7, 0.8, 0.9]

    print("RLHF-relabeling robustness sweep: same pipeline as "
          "run_rlhf_relabeling_demo.py, across multiple configs and 30 seeds.\n")
    header = (f"{'config':>28} {'gamma':>6} {'mean coverage':>14} {'min coverage':>13} "
              f"{'frac(cov>=gamma)':>17} {'mean alpha':>11}")
    print(header)
    all_ok = True
    for cfg in CONFIGS:
        cfg_label = f"nr={cfg['n_responses']},na={cfg['n_attributes']},N={cfg['N']}"
        for gamma in gammas:
            covs, alphas = [], []
            for seed in seeds:
                c, a, _, _ = run_once(seed, gamma, n_responses=cfg["n_responses"],
                                       n_attributes=cfg["n_attributes"], N=cfg["N"])
                covs.append(c)
                alphas.append(a)
            covs = np.array(covs)
            frac_ok = float(np.mean(covs >= gamma - 1e-9))
            all_ok = all_ok and frac_ok >= 0.9  # allow a small finite-sample slack
            print(f"{cfg_label:>28} {gamma:>6.2f} {covs.mean():>14.3f} {covs.min():>13.3f} "
                  f"{frac_ok:>17.3f} {np.mean(alphas):>11.4f}")

    print("\nInterpretation: 'frac(cov>=gamma)' is the fraction of the 30 seeds "
          "(per config/gamma) where the empirical coverage met or exceeded the "
          "target gamma -- this is what the finite-sample conformal guarantee "
          "promises (marginally, not per-seed, so occasional undershoots are "
          "expected and not a failure of the theorem).")
    print(f"\nAll configs/gammas at >=90% of seeds meeting target: {all_ok}")


if __name__ == "__main__":
    main()
