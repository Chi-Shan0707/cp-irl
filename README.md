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

CP-IRL asks which parts of Conformal Inverse Optimization (CIO) survive the
move from one-shot decisions to inverse reinforcement learning, and answers
component by component. The conformal calibration itself transfers untouched —
it is split conformal prediction, and no claim here improves on it. Two things
do not transfer: CIO's angular score is measured in reward coordinates that
behavior does not fix, so we replace it with a distance in the span seminorm
induced by the occupancy polytope; and the calibrated event ("some reward in
the set explains the new demonstrator") is weaker than what a robust
prescription needs ("the demonstrator's own reward is in the set"). Closing
that second gap takes an explicit identification assumption — a certified
inverse-fiber diameter — which expands the intersection radius into a latent
reward-ray containment radius used by a reference-relative robust policy.

<details>
<summary><strong>Abstract</strong></summary>
<br>

**CP-IRL** uses one geometric object — the occupancy-difference span — to
calibrate Bellman inverse fibers and to derive the downstream robust penalty.
Unit-span normalization makes the construction invariant to positive reward
scale, additive null directions, and invertible feature reparameterizations
while keeping the score an LP; a feature reparameterization that leaves every
reward and policy unchanged moves CIO's angular radius by 24.3° and ours by
2e-8. Split conformal gives exact inverse-fiber intersection coverage (measured
0.82 at target 0.80), but containment of the demonstrator's own reward at the
same radius reaches only 0.44. Given a certified fiber diameter `eta`, the
implementation converts the calibrated radius `q` into a containment radius `R`
and solves a safe reference-relative robust LP. If `R >= 1`, the guarantee is
provably degenerate and the solver returns the reference explicitly.

</details>

## What's here

- tabular MDP, gridworld, and Objectworld environments
- classical, maximum-entropy, and Bayesian IRL estimators
- conformal reward-set calibration based on Bellman-optimal feasible rewards
- intrinsic reward-ray normalization and a two-stage `q, eta -> R` bridge
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

The implementation uses Python 3.10 with NumPy, SciPy, and CVXPY. Install the
tested dependency ranges with:

```bash
python -m pip install -r requirements.txt
```

Tests run from the repository root with:

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
features. A useful safe prescription requires an application-specific
certificate satisfying `2*q + eta < 1`; the universal `eta = 2` bound is valid
but forces the robust policy back to the reference. The framework is complete,
while learning or certifying a small fiber diameter is left to the intended
application's additional information.

## Citation

This is a working manuscript (NeurIPS 2026 workshop submission); it is not yet
formally published. If you use this code or build on the results, please cite
it as:

```bibtex
@misc{chi2026cpirl,
  title  = {Calibrating What Behavior Identifies: Conformal Reward Sets for
            Inverse Reinforcement Learning},
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
