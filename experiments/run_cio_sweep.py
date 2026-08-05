"""Phase 1 gate, full version (plan.md Sec 6): reproduce CIO's Fig. 3 (coverage vs
N_val, showing over-coverage shrinking as N_val grows -- "asymptotically exact") and
Fig. 4 (conformal IO beats classic IO on AOG/POG across gamma), with enough seeds to
say something statistically honest (10, matching the paper).

Parallelized: the per-point conformal calibration score c_k (Theorem 1) and the
coverage-check c_k's are embarrassingly parallel across the 20 physical cores
available on this machine (see plan.md Sec 3).

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python experiments/run_cio_sweep.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cio.network import build_layered_network
from cio.io_pipeline import generate_population, classic_io, conformal_calibrate, _c_k
from cio.shortest_path import solve_forward
from cio.robust import solve_rfo

N_JOBS = 16
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "cio_sweep"


def aog_pog(net, demos_test, theta_star_unit, x_bar_fn):
    aog_vals, pog_vals = [], []
    for dem in demos_test:
        s, t = dem.s, dem.t
        x_bar = x_bar_fn(s, t)
        x_star_true = solve_forward(net, theta_star_unit, s, t)
        aog_vals.append(theta_star_unit @ x_bar - theta_star_unit @ x_star_true)
        x_star_perceived = solve_forward(net, dem.theta_hat, s, t)
        pog_vals.append(dem.theta_hat @ x_bar - dem.theta_hat @ x_star_perceived)
    return float(np.mean(aog_vals)), float(np.mean(pog_vals))


def coverage_check(net, theta_bar, alpha, demos_cov, n_jobs=N_JOBS):
    from multiprocessing import Pool
    with Pool(n_jobs) as pool:
        c_ks = pool.starmap(_c_k, [(net, dem, theta_bar) for dem in demos_cov])
    c_ks = np.array(c_ks)
    return float(np.mean(c_ks >= np.cos(alpha) - 1e-9))


def run_once(seed: int, gamma: float, N_train: int, N_val: int, N_test: int,
             net_shape=(2, 5, 5, 2)) -> dict:
    rng = np.random.default_rng(seed)
    net = build_layered_network(*net_shape, skip_connections=True)
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    theta_star_unit = theta_star / np.linalg.norm(theta_star)

    demos_train = generate_population(net, theta_star, N_train, rng)
    demos_val = generate_population(net, theta_star, N_val, rng)
    demos_test = generate_population(net, theta_star, N_test, rng)
    demos_cov = generate_population(net, theta_star, N_test, rng)

    theta_bar = classic_io(net, demos_train)
    alpha, _ = conformal_calibrate(net, demos_val, theta_bar, gamma, n_jobs=N_JOBS)
    coverage = coverage_check(net, theta_bar, alpha, demos_cov)

    aog_classic, pog_classic = aog_pog(
        net, demos_test, theta_star_unit, lambda s, t: solve_forward(net, theta_bar, s, t))
    aog_cio, pog_cio = aog_pog(
        net, demos_test, theta_star_unit, lambda s, t: solve_rfo(net, theta_bar, alpha, s, t))

    return dict(seed=seed, gamma=gamma, N_val=N_val, alpha=alpha, coverage=coverage,
                aog_classic=aog_classic, aog_cio=aog_cio,
                pog_classic=pog_classic, pog_cio=pog_cio)


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(10))
    N_train, N_test = 60, 50

    print("=== Sweep 1: coverage vs N_val (target gamma=0.8, fixed) ===")
    gamma_fixed = 0.8
    n_val_grid = [50, 100, 200]
    rows_nval = []
    t0 = time.time()
    for N_val in n_val_grid:
        covs = []
        for seed in seeds:
            r = run_once(seed, gamma_fixed, N_train, N_val, N_test)
            covs.append(r["coverage"])
            rows_nval.append(r)
        print(f"N_val={N_val:>4}: mean coverage={np.mean(covs):.3f} "
              f"std={np.std(covs):.3f}  (target={gamma_fixed}, over-coverage should shrink)")
    print(f"  [{time.time()-t0:.1f}s]")

    print("\n=== Sweep 2: AOG/POG improvement across gamma (fixed N_val=100) ===")
    N_val_fixed = 100
    gammas = [0.5, 0.7, 0.9]
    rows_gamma = []
    t0 = time.time()
    for gamma in gammas:
        aog_imprs, pog_imprs, covs = [], [], []
        for seed in seeds:
            r = run_once(seed, gamma, N_train, N_val_fixed, N_test)
            aog_impr = 100 * (r["aog_classic"] - r["aog_cio"]) / max(r["aog_classic"], 1e-9)
            pog_impr = 100 * (r["pog_classic"] - r["pog_cio"]) / max(r["pog_classic"], 1e-9)
            aog_imprs.append(aog_impr)
            pog_imprs.append(pog_impr)
            covs.append(r["coverage"])
            rows_gamma.append({**r, "aog_impr": aog_impr, "pog_impr": pog_impr})
        print(f"gamma={gamma:.2f}: mean coverage={np.mean(covs):.3f}, "
              f"AOG improvement={np.mean(aog_imprs):.1f}% (std {np.std(aog_imprs):.1f}), "
              f"POG improvement={np.mean(pog_imprs):.1f}% (std {np.std(pog_imprs):.1f})")
    print(f"  [{time.time()-t0:.1f}s]")

    with open(RUNS_DIR / "sweep_nval.json", "w") as f:
        json.dump(rows_nval, f, indent=2)
    with open(RUNS_DIR / "sweep_gamma.json", "w") as f:
        json.dump(rows_gamma, f, indent=2)
    print(f"\nRaw results written to {RUNS_DIR}/")


if __name__ == "__main__":
    main()
