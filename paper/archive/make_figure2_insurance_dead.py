"""Figure 2 for the paper: the Q2 "insurance cost" finding -- robust hedging
cuts true-reward regret under systematic bias but costs regret under ordinary
noise. Currently this headline empirical claim exists only as numbers scattered
across the Experiments/Discussion prose; this figure puts it in one place.

Panel (a) reuses experiments/run_t5_separation.py's degenerate construction (the
same data source as Figure 1) at two representative scaling parameters u=2 and
u=100, LP-IRL point estimate, plain vs. conformal-robust decision.

Panel (b) reuses experiments/run_irl_demo.py (random MDP), the gridworld numbers
in notes/phase5_gridworld_result.md, and the Objectworld numbers in
notes/phase5_objectworld_result.md -- all LP-IRL, gamma_target=0.8, plain vs.
conformal-robust decision, matched protocol across the three environments.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python paper/make_figure2_insurance.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_t5_separation import run_once as run_biased_once
from experiments.run_irl_demo import run_once as run_random_mdp_once

# Okabe-Ito colorblind-safe palette (matches make_figure1.py)
COLOR_CLASSIC = "#D55E00"   # vermillion
COLOR_CIRL = "#0072B2"      # blue


def biased_panel_data():
    us = [2, 100]
    seeds = list(range(10))
    means_classic, means_cirl = [], []
    for u in us:
        acs, arobs = [], []
        for seed in seeds:
            ac, acio, _, _, _ = run_biased_once(u, seed)
            acs.append(ac)
            arobs.append(acio)
        means_classic.append(np.mean(acs))
        means_cirl.append(np.mean(arobs))
    return us, means_classic, means_cirl


def noisy_panel_data():
    # random MDP: freshly computed here, matched protocol (gamma=0.8, N=30, LP-IRL)
    seeds = list(range(5))
    ac_rmdp, ar_rmdp = [], []
    for seed in seeds:
        out = run_random_mdp_once(seed, gamma=0.8, N_train=30, N_val=30, N_test=30)
        ac_rmdp.append(out["aog_classic"])
        ar_rmdp.append(out["aog_cio"])
    # gridworld and Objectworld: from notes/phase5_gridworld_result.md and
    # notes/phase5_objectworld_result.md (LP-IRL rows, gamma_target=0.8)
    envs = ["Random MDP", "Gridworld", "Objectworld"]
    means_classic = [np.mean(ac_rmdp), 0.5814, 0.1194]
    means_cirl = [np.mean(ar_rmdp), 1.9960, 1.2745]
    return envs, means_classic, means_cirl


def main():
    us, biased_classic, biased_cirl = biased_panel_data()
    envs, noisy_classic, noisy_cirl = noisy_panel_data()

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # Panel (a): systematic bias -- robustification helps
    # Transform to log10 so ticks are equally spaced integers
    log_biased_classic = np.log10(np.maximum(biased_classic, 1e-12))
    log_biased_cirl    = np.log10(np.maximum(biased_cirl, 1e-12))
    x = np.arange(len(us))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w / 2, log_biased_classic, w, color=COLOR_CLASSIC, label="Classic IRL")
    ax.bar(x + w / 2, log_biased_cirl, w, color=COLOR_CIRL, label="CP-IRL")
    ticks_a = [-3, -2, -1, 0, 1, 2, 3]
    ax.set_yticks(ticks_a)
    ax.set_yticklabels([str(t) for t in ticks_a])
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$u={u}$" for u in us])
    ax.set_ylabel(r"$\lg$(True-reward regret / AOG)")
    ax.set_title("(a) Systematic bias:\nrobustification helps", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.04, 0.04), frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)

    # Panel (b): ordinary noise -- robustification hurts
    ax = axes[1]
    x = np.arange(len(envs))
    ax.bar(x - w / 2, noisy_classic, w, color=COLOR_CLASSIC, label="Classic IRL")
    ax.bar(x + w / 2, noisy_cirl, w, color=COLOR_CIRL, label="CP-IRL")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=7.5)
    ax.set_ylabel("True-reward regret (AOG)")
    ax.set_title("(b) Ordinary noise:\nrobustification hurts", fontsize=9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "figures" / "fig2_insurance.pdf"
    fig.savefig(out_path)
    out_path_png = out_path.with_suffix(".png")
    fig.savefig(out_path_png, dpi=200)
    print(f"Wrote {out_path} and {out_path_png}")
    print("Panel (a):", dict(zip(us, zip(biased_classic, biased_cirl))))
    print("Panel (b):", dict(zip(envs, zip(noisy_classic, noisy_cirl))))


if __name__ == "__main__":
    main()
