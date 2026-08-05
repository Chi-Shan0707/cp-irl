# CP-IRL

CP-IRL extends conformal inverse optimization from one-shot decisions to inverse
reinforcement learning. It calibrates a finite-sample, distribution-free reward
uncertainty set from a population of demonstrators and computes a robust policy
for the induced Markov decision process.

The repository contains:

- tabular MDP, gridworld, and Objectworld environments;
- classical, maximum-entropy, and Bayesian IRL estimators;
- conformal reward-set calibration based on Bellman-optimal feasible rewards;
- an occupancy-measure solver for robust MDPs with reward ambiguity;
- experiment scripts for coverage, regret, separation, and ablation studies;
- the NeurIPS 2026 workshop paper and its figure source.

## Paper

The current manuscript is available at [`paper/main.pdf`](paper/main.pdf).

## Environment

The implementation uses Python with NumPy, SciPy, and CVXPY. Tests can be run
from the repository root with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
  tests cio/tests irl/tests robust/tests conformal/tests -q
```

## Repository layout

```text
envs/          tabular MDP environments
cio/           conformal inverse optimization reference implementation
irl/           inverse reinforcement learning estimators and feasible sets
conformal/     calibration and concentration baselines
robust/        robust MDP optimization
experiments/   experiment and ablation scripts
tests/         environment tests
paper/         manuscript, bibliography, and figures
```

## Scope

The current implementation targets tabular MDPs with rewards linear in known
features. The experiments show that robust reward hedging is most useful when
the point estimator is systematically biased; with approximately unbiased but
noisy estimates, robustness can incur additional true-reward regret.
