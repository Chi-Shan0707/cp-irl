# Post-acceptance author revision — 6 October 2026

This is an author revision **after acceptance**, not the document on which the
workshop decision was based. The new experiment below was not peer-reviewed
as part of that submission. No OpenReview submission or decision was changed.

## Version boundary

The accepted/reviewed artifacts remain byte-for-byte unchanged:

| File | SHA-256 |
|---|---|
| `main_full9_revised.tex` | `b6846566e4a61a2d7d7de7f0329dc905675512225fa50a215524217f37177bf4` |
| `main_full9_revised.pdf` | `79fd2a882937df7396c370db13e06e1a58c7737903f09de4bd8fdae5e44f5e45` |

The pre-update public camera-ready files are preserved in Git at
[`4553b426876222a6b16cd0029bf954425a806bd0`](https://github.com/Chi-Shan0707/cp-irl/tree/4553b426876222a6b16cd0029bf954425a806bd0/paper):

| File at that commit | SHA-256 |
|---|---|
| `main.tex` | `5038522112543dbbcbebe5b3f4a6032a3078973029a236dd3b062d53941b4689` |
| `main.pdf` | `08a7a0b1c7fa2292a2585dac8d5d13f4f8284fb0521f6137901157d4871fb2de` |

`main.tex` / `main.pdf` now identify themselves as the dated author revision.
The original E1–E3 result JSONs, figure files, and Table 2 numbers are unchanged.
Private reviewer text is not included in the public change record.

## 1. Attribution to CIO

The abstract, introduction, Table 1, related work and discussion now explicitly
acknowledge that Lin, Delage and Chan already formulate intersection coverage
and assume an inverse-feasible-set diameter bound. The primary source is
[Conformal Inverse Optimization, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7423902b5534e2b267438c85444a54b1-Paper-Conference.pdf):

- §3.1.1, Assumption 2 (printed p. 3): Euclidean inverse-set diameter ≤ η.
- §4 opening and Theorem 2 (printed pp. 6–7): coverage of an intersection event.
- Theorem 3, equations (13)–(14) (printed p. 7): decision-quality bounds use η.

CIO **assumes** this bound; it does not supply a general estimator of it.
Our normalized span-width assumption is an adaptation to a different geometry,
not the first use of an identification-width assumption. We removed the
unsupported generalization that one-shot inverse-feasible sets are necessarily
thin. The specific invariant Bellman-cone LP score, normalized `2q + η` bridge,
and reference-relative ball degeneracy analysis remain the claimed adaptations.
The existing bibliography entry is unchanged.

## 2. Scope and mathematical clarifications

- The `R ≥ 1` statement concerns the paper's translated span ball and relative
  robust objective. It is not an information-theoretic impossibility for every
  uncertainty set or decision rule. At `R = 1` other optima may tie the reference;
  at `R > 1` every optimum is feature-equivalent to it.
- `R < 1` is necessary, not sufficient, for strictly positive worst-case
  advantage. E3's 85.0/76.7/36.7% values retain their numbers but are described as
  threshold-eligible fractions, not demonstrated improvement rates. The legacy
  `SafeRobustResult.nontrivial` field keeps its behavior; its docstring clarifies
  that it reports this threshold only. No production solver behavior changes.
- Degeneration to the reference does not predict whether the reference beats
  or loses to a point estimator. Table 2's losses remain empirical findings.
- A sharper width can be computed from a known MDP in special cases; it need
  not always come from external data. Side information that restricts a cone
  would require updating both the feasible set and its score/certificate.
- The split-conformal proof replaces an incorrect equivalence in the presence
  of tied scores by the sufficient one-way implication from randomized rank to
  threshold coverage. The coverage theorem and algorithm do not change.
- A witness-distance display uses `≤`, rather than `=`, when substituting its
  upper bound `q`. The containment theorem is unchanged.
- Exact-maximum calibration is stated with nonnull latent rewards. The
  abstract no longer conflates unchanged absolute values with unchanged value
  differences, and specifies κ = 200 for the 24° result.
- The E2 figure caption distinguishes score coverage from set intersection for
  the value-gap baseline, and identifies the three containment geometries.
  The original 1,200 evaluations reuse 600 distinct test rewards across the
  two estimators; the block-based confidence intervals are unchanged.

## 3. New, separately labelled Appendix D

`post_acceptance_toy.tex` adds a one-state self-looping MDP with regular-polygon
action features and two reward parameters. Each normalized Bellman cone is a
line segment. Its analytic endpoints give exact maximum-distance scores and a
full-cone diameter `η = 2 tan(π/m)` for `m` divisible by four. Neither quantity
is certified using sampled latent rewards.

The fixed experiment has discount 0.9, target 0.8, seed 20261006 and 100
independent splits, each with 20 calibration and 100 held-out rewards. The
unit-span center and uniform reference are fixed before data generation.

| Case | Certificate and outcome |
|---|---|
| 16 actions; demonstrators select −1, 0, 1 with probabilities ¼, ½, ¼ | η = 0.397825, q = 0.191342, R = 0.780508; worst-case advantage = 0.109746, with an actual departure from the reference. Exact-maximum calibration gives radius 0.581596 and worst-case advantage 0.209202. Both choose the same action. |
| 4 actions; all demonstrators select action 0 | η = 2, q = 0, R = 2; exact maximum radius = 1. Both ball prescriptions have zero worst-case advantage. Direct optimization over the full normalized action-0 cone certifies advantage 0.5. |

The first case exercises a positive certificate; the second isolates set-shape
conservatism. The control's direct-cone comparison is for the explicitly stated
single-policy population, not a new general population-coverage method.
An oracle quantile uses latent **calibration** rewards only and is labelled
non-deployable. Test rewards are used exclusively for evaluation. Finite-sample
guarantees come from the proof, not the observed coverage rates.

The experiment does not establish useful widths for the original benchmarks,
new general computational guarantees, superiority to point-estimate IRL, or
results for learned dynamics / finite demonstrations.

## Reproduction and validation

Use the repository's `rlenv` environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlenv
make toy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
  tests/test_exact_cone_toy.py robust/tests/test_decision_aware_robust.py -q
make paper
```

`experiments/exact_cone_toy_results.json` records configuration, all cone
endpoints/scores, computed occupancies, every trial, and standard errors across
independent trial means. Analytic endpoint extrema are checked against LPs over
the full normalized cone. Cone scores and robust inner objectives are also
cross-checked with independent SciPy LP formulations. See
[PROVENANCE.md](PROVENANCE.md) for the boundary between original and new results.

Validation on 2026-10-06: the default toy run completed; all 36 focused tests
above passed. The revised PDF builds to 16 pages with no undefined references,
undefined citations, or overfull boxes. The dated notice and Appendix D table
were checked in the rendered PDF. Accepted-artifact hashes match the table
above; original E1–E3 JSONs and figures have no diff.

## 4. Public explanation and overview figures

The README, project page and Chinese math walkthrough now match the revised
attribution, guarantee scope, experiment accounting and separately labelled
Appendix D. The accepted submission remains linked alongside the author revision.

Two new web overview figures (`figures/core_geometry.*` and
`figures/core_prescription.*`) are explanatory additions after acceptance, not
figures from the reviewed paper. They do not replace original Figures 1–2 and
are not inserted into the accepted PDF. `make overview` generates SVG, PDF and
PNG from `make_core_figures.py` without running experiments.

- Geometry uses the exact Appendix D 16-action MDP in coordinates
  `w = 2θ/(1−β)`, for which the occupancy-span norm is `max_j p_jᵀw`.
  The balls are polygons. A separate SciPy LP computes the nearest-cone
  distance; assertions check unit normalization, the latent-reward miss,
  the exact fiber diameter, and containment of both fiber endpoints.
- The decision curve `G(R) = max{0, (1−R)/2}` is specific to this MDP,
  action-0 center and uniform reference. The plotting source derives it
  from the gauge of the action-feature polygon. It is labelled as an
  example, distinct from the general `R ≥ 1` degeneracy statement.
- Captions state the assumptions and marginal nature of coverage, and
  distinguish the sphere/ball prescription from other reward-set shapes.
  Exported figures are visually checked for readable, separated labels;
  text, including axis labels, ticks, legends and annotation text, also
  receives a bounding-box overlap check. Local browser checks at 1280 px and
  390 px verified that the SVGs load and the page has no horizontal overflow.
