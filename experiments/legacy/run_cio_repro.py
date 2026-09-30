"""Phase 1 gate (plan.md Sec 6): reproduce the CIO paper's two qualitative claims on
our shortest-path instance:
  1. Empirical coverage of the calibrated uncertainty set tracks the target gamma.
  2. Conformal IO achieves lower AOG (and POG) than classic (point-estimate) IO.

This is a smoke-scale run (modest N, one network size) meant to validate the pipeline
end-to-end honestly before committing to the paper-scale sweep (many seeds x gamma x
N_val, matching CIO's Fig. 3/4). Run:

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    python experiments/run_cio_repro.py
"""
from __future__ import annotations

import time

import numpy as np

from cio.network import build_layered_network
from cio.io_pipeline import generate_population, classic_io, conformal_calibrate
from cio.shortest_path import solve_forward
from cio.robust import solve_rfo


def aog_pog(net, demos_test, theta_star, x_bar_fn):
    """AOG = E[f(theta*, xbar(u)) - f(theta*, x*(u))]
       POG = E[f(theta_hat, xbar(u)) - f(theta_hat, x*(theta_hat, u))]
    x_bar_fn(s, t) -> decision (edge-flow vector) for the prescribed policy.
    """
    aog_vals, pog_vals = [], []
    for dem in demos_test:
        s, t = dem.s, dem.t
        x_bar = x_bar_fn(s, t)

        x_star_true = solve_forward(net, theta_star, s, t)
        aog_vals.append(theta_star @ x_bar - theta_star @ x_star_true)

        x_star_perceived = solve_forward(net, dem.theta_hat, s, t)
        pog_vals.append(dem.theta_hat @ x_bar - dem.theta_hat @ x_star_perceived)

    return float(np.mean(aog_vals)), float(np.mean(pog_vals))


def run_once(seed: int, gamma: float, N_train: int, N_val: int, N_test: int,
             net_shape=(2, 5, 5, 2)) -> dict:
    rng = np.random.default_rng(seed)
    net = build_layered_network(*net_shape, skip_connections=True)
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    demos_train = generate_population(net, theta_star, N_train, rng)
    demos_val = generate_population(net, theta_star, N_val, rng)
    demos_test = generate_population(net, theta_star, N_test, rng)

    theta_bar = classic_io(net, demos_train)
    alpha, _ = conformal_calibrate(net, demos_val, theta_bar, gamma)

    # empirical coverage on a FRESH held-out sample (not train/val/test decisions,
    # but the same generating process) -- does the calibrated cap contain a theta
    # that would make a next observed decision optimal?
    demos_cov = generate_population(net, theta_star, N_test, rng)
    n_covered = 0
    from cio.io_pipeline import _c_k
    for dem in demos_cov:
        c = _c_k(net, dem, theta_bar)
        if c >= np.cos(alpha) - 1e-9:
            n_covered += 1
    coverage = n_covered / len(demos_cov)

    aog_classic, pog_classic = aog_pog(
        net, demos_test, theta_star_unit,
        lambda s, t: solve_forward(net, theta_bar, s, t),
    )
    aog_cio, pog_cio = aog_pog(
        net, demos_test, theta_star_unit,
        lambda s, t: solve_rfo(net, theta_bar, alpha, s, t),
    )

    return dict(seed=seed, gamma=gamma, alpha=alpha, coverage=coverage,
                aog_classic=aog_classic, aog_cio=aog_cio,
                pog_classic=pog_classic, pog_cio=pog_cio)


def main():
    N_train, N_val, N_test = 60, 60, 60
    gammas = [0.5, 0.7, 0.9]
    seeds = [0, 1, 2]

    print(f"{'seed':>4} {'gamma':>6} {'alpha':>7} {'coverage':>9} "
          f"{'AOG_classic':>12} {'AOG_cio':>9} {'AOG_impr%':>10} "
          f"{'POG_classic':>12} {'POG_cio':>9} {'POG_impr%':>10}")

    rows = []
    t0 = time.time()
    for seed in seeds:
        for gamma in gammas:
            r = run_once(seed, gamma, N_train, N_val, N_test)
            aog_impr = 100 * (r["aog_classic"] - r["aog_cio"]) / max(r["aog_classic"], 1e-9)
            pog_impr = 100 * (r["pog_classic"] - r["pog_cio"]) / max(r["pog_classic"], 1e-9)
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
              f"mean AOG improvement={np.mean(aog_imprs):.1f}%, "
              f"mean POG improvement={np.mean(pog_imprs):.1f}%")


if __name__ == "__main__":
    main()
