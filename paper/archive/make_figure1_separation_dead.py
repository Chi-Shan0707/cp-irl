"""Figure 1 for the paper (paper/draft.md's checklist flags this as the priority
next step): the T5 separation result -- classic IRL's true-reward regret (AOG)
diverges unboundedly as the scaling parameter u grows, while CP-IRL's stays
bounded and shrinks toward zero. This is the single most visually compelling
result in the paper (per the ml-paper-writing skill's guidance that Figure 1
should convey the core idea/most compelling result).

Reuses experiments/run_t5_separation.py's exact data-generating pipeline (10
seeds) rather than re-deriving numbers, so the figure matches the reported table
in notes/phase2_t5_construction.md exactly.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlenv
    PYTHONPATH=. python paper/make_figure1.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_t5_separation import run_once

# Okabe-Ito colorblind-safe palette
COLOR_CLASSIC = "#D55E00"   # vermillion
COLOR_CIRL = "#0072B2"      # blue


def main():
    us = [2, 5, 10, 25, 50, 100]
    seeds = list(range(10))

    mean_classic, std_classic = [], []
    mean_cirl, std_cirl = [], []

    for u in us:
        aogs_c, aogs_r = [], []
        for seed in seeds:
            ac, ar, _adkw, _alpha_c, _alpha_d = run_once(u, seed)
            aogs_c.append(ac)
            aogs_r.append(ar)
        mean_classic.append(np.mean(aogs_c))
        std_classic.append(np.std(aogs_c))
        mean_cirl.append(np.mean(aogs_r))
        std_cirl.append(np.std(aogs_r))

    us = np.array(us)
    mean_classic = np.array(mean_classic)
    std_classic = np.array(std_classic)
    mean_cirl = np.array(mean_cirl)
    std_cirl = np.array(std_cirl)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    # Transform to log10 so the linear axis is equally spaced at integer exponents
    log_mean_classic = np.log10(np.maximum(mean_classic, 1e-12))
    log_std_classic  = std_classic / (np.maximum(mean_classic, 1e-12) * np.log(10))
    log_mean_cirl    = np.log10(np.maximum(mean_cirl, 1e-12))
    log_std_cirl     = std_cirl / (np.maximum(mean_cirl, 1e-12) * np.log(10))

    ax.plot(us, log_mean_classic, marker="o", color=COLOR_CLASSIC, linewidth=2,
             label="Classic (point-estimate) IRL", zorder=3)
    ax.fill_between(us,
                     log_mean_classic - log_std_classic,
                     log_mean_classic + log_std_classic,
                     color=COLOR_CLASSIC, alpha=0.15)

    ax.plot(us, log_mean_cirl, marker="s", color=COLOR_CIRL, linewidth=2,
             label="CP-IRL (calibrated, robust)", zorder=3)
    ax.fill_between(us,
                     log_mean_cirl - log_std_cirl,
                     log_mean_cirl + log_std_cirl,
                     color=COLOR_CIRL, alpha=0.15)

    # Linear axis with integer log10 tick labels
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("Scaling parameter $u$")
    ax.set_ylabel(r"$\lg$(True-reward regret / AOG)")
    ax.set_title("Separation: classic IRL's regret diverges,\nCP-IRL's shrinks (10 seeds, shaded = std)", fontsize=10)
    # Legend at the bottom-left in the empty space
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.95,
              loc="lower left", bbox_to_anchor=(0.04, 0.04), fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", alpha=0.25)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "figures" / "fig1_separation.pdf"
    fig.savefig(out_path)
    out_path_png = out_path.with_suffix(".png")
    fig.savefig(out_path_png, dpi=300)
    print(f"Saved {out_path} and {out_path_png}")
    print("u:", list(us))
    print("mean_classic:", list(np.round(mean_classic, 4)))
    print("mean_cirl:", list(np.round(mean_cirl, 4)))


if __name__ == "__main__":
    main()
