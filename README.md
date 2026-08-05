<div align="center">

# CP-IRL

**Conformal Prediction for Inverse Reinforcement Learning**

*From Inverse Optimization to Sequential Decisions*

[![License: MIT](https://img.shields.io/badge/License-MIT-8a3b2b.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#environment)
[![Paper](https://img.shields.io/badge/paper-PDF-8a3b2b.svg)](paper/main.pdf)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-555.svg)](https://chi-shan0707.github.io/cp-irl/)

[Paper (PDF)](paper/main.pdf) &nbsp;·&nbsp;
[Math walkthrough](math_summary.html) &nbsp;·&nbsp;
[Project page](https://chi-shan0707.github.io/cp-irl/) &nbsp;·&nbsp;
[Citation](#citation) &nbsp;·&nbsp;
[License](#license)

</div>

---

CP-IRL extends Conformal Inverse Optimization (CIO) from one-shot decisions to
inverse reinforcement learning. It calibrates a finite-sample,
distribution-free reward uncertainty set from a population of demonstrators
and computes a robust policy for the induced Markov decision process.

<details>
<summary><strong>Abstract</strong></summary>
<br>

Inverse reinforcement learning (IRL) is inverse optimization with a sequential
forward problem: demonstrations reveal an unknown reward, and the recovered
reward is subsequently optimized to prescribe a policy. This perspective
exposes the same weakness identified by Conformal Inverse Optimization (CIO)
for one-shot decisions: collapsing heterogeneous objectives to one point
estimate can yield a bad downstream decision. We introduce **CP-IRL**, which
transfers CIO's calibrate-then-robustify principle to Markov decision
processes. From a population of demonstrators, CP-IRL conformally calibrates a
set of reward parameters consistent with a new demonstrator and optimizes a
policy against the resulting reward ambiguity. The sequential setting requires
new machinery. We reduce each calibration score to a reward-dimensional convex
program using the Bellman resolvent, and prove that a fixed transition kernel
makes the robust MDP convex for any compact reward ambiguity set; a two-state
counterexample shows that this property can fail when ambiguity also enters
the dynamics. We establish finite-sample coverage, regret bounds, and a
separation example in which point-estimate IRL has unbounded regret while
CP-IRL's regret vanishes. Experiments on random MDPs, gridworld, and
Objectworld identify the method's scope: robustification reduces true-reward
regret under systematic estimation bias, but incurs an insurance cost when the
point estimate is merely noisy and approximately unbiased.

</details>

## What's here

- tabular MDP, gridworld, and Objectworld environments
- classical, maximum-entropy, and Bayesian IRL estimators
- conformal reward-set calibration based on Bellman-optimal feasible rewards
- an occupancy-measure solver for robust MDPs with reward ambiguity
- experiment scripts for coverage, regret, separation, and ablation studies
- the NeurIPS 2026 workshop paper and its figure source

## Paper

The current manuscript is at [`paper/main.pdf`](paper/main.pdf). A math
walkthrough of the core theory — definitions, full proofs, and a worked
geometric counterexample — is at
[`math_summary.html`](math_summary.html), also published via
[GitHub Pages](https://chi-shan0707.github.io/cp-irl/).

## Environment

The implementation uses Python with NumPy, SciPy, and CVXPY. Tests run from
the repository root with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
  tests cio/tests irl/tests robust/tests conformal/tests -q
```

## Repository layout

| Path            | Contents                                             |
|-----------------|-------------------------------------------------------|
| `envs/`         | tabular MDP, gridworld, Objectworld environments       |
| `cio/`          | conformal inverse optimization reference implementation |
| `irl/`          | inverse reinforcement learning estimators and feasible sets |
| `conformal/`    | calibration and concentration baselines                |
| `robust/`       | robust MDP optimization                                |
| `experiments/`  | experiment and ablation scripts                        |
| `tests/`        | environment tests                                       |
| `paper/`        | manuscript, bibliography, and figures                  |

## Scope

The current implementation targets tabular MDPs with rewards linear in known
features. The experiments show that robust reward hedging is most useful when
the point estimator is systematically biased; with approximately unbiased but
noisy estimates, robustness can incur additional true-reward regret.

## Citation

This is a working manuscript (NeurIPS 2026 workshop submission); it is not yet
formally published. If you use this code or build on the results, please cite
it as:

```bibtex
@misc{chi2026cpirl,
  title  = {From Inverse Optimization to Sequential Decisions: Conformal
            Prediction for Inverse Reinforcement Learning},
  author = {Chi, Yuhan},
  year   = {2026},
  note   = {Manuscript in preparation, Fudan University},
  url    = {https://github.com/Chi-Shan0707/cp-irl}
}
```

A machine-readable citation is also provided in
[`CITATION.cff`](CITATION.cff) (GitHub renders this as a "Cite this
repository" button). This entry will be updated with a venue and DOI once the
paper is formally published — check back for the canonical citation if you're
citing this after that point.

## License

Code is released under the [MIT License](LICENSE). The paper text and figures
in `paper/` are © Yuhan Chi and are made available for reading and citation;
if you'd like to reuse figures or text beyond fair use/citation, please reach
out (yhchi25@m.fudan.edu.cn).
