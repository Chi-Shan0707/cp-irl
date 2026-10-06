# Where each number in the paper comes from

The original E1–E3/Table 2 values below were recomputed from the recorded output
on 2026-09-30 and remain unchanged in the 2026-10-06 author revision.
The new Appendix D results are listed separately below and postdate acceptance.
The recorded JSONs are committed, so `make figures`
and these checks work from a fresh clone. Rerunning the scripts (`make
experiments`) regenerates them.

| Paper location | Number(s) | Script | Recorded output |
|---|---|---|---|
| Abstract, §1, §5 E1, Fig. 1a | angular radius moves 0.424 rad (24.3°) on average and 1.071 worst at κ=200; span radius ≤ 2.0×10⁻⁸ at κ=200; reward discrepancy ≤ 2.1×10⁻¹⁴ | `experiments/run_reparam_invariance.py` | `experiments/reparam_invariance_results.json` |
| Abstract, §1, §5 E2, Fig. 1b | intersection 0.812 / 0.838 / 0.819; containment 0.368 / 0.004 / 0.438 (95% t over 30 environment–seed blocks) | `experiments/run_containment_audit.py` | `experiments/containment_audit_results_v2.json` |
| §5 E3, Fig. 2 | q_γ mean 0.316, median 0.289, range [0.019, 0.871]; R<1 eligibility 85.0 / 76.7 / 36.7 % at η = 0, 0.2, 0.5 (necessary, not sufficient, for positive robust advantage); 95 / 75 / 60 % by environment at η=0.2; η=2 gives R=2, containment 1.000, dominance 1.000, eligibility 0 | `experiments/run_containment_audit.py` | `experiments/containment_audit_results_v2.json` |
| App. B, Table 2 | all 30 regret cells (`aog_*` fields, mean over 10 seeds) | `experiments/run_containment_audit.py` | `experiments/containment_audit_results_v2.json` |
| §5 setup | exact-fit LP baseline: span-zero center in 1 of 30 fits (gridworld, seed 6) | `experiments/run_center_validity.py` | `experiments/center_validity_results.json` |
| App. B, concentration baseline (value gap) | DKW radius infinite in 270/270 runs; 4–460× regret at the 10‖θ̄‖₂ cap | `experiments/legacy/run_value_gap_full_suite.py` | `experiments/value_gap_full_suite_results.json` (not committed; regenerate with the script) |
| App. B, concentration baseline (angular) | DKW cap never narrower than conformal | `experiments/legacy/run_phase5_gridworld.py` (prints to stdout); `conformal/tests/test_concentration_baseline.py` | — |
| App. A, after Prop. 5 | resolvent score 2–5.5× faster than the joint-(θ, V) reference for \|S\| = 20–100; agreement to solver precision | agreement: `irl/tests/test_feasible_set.py`; timing was measured once during development and no timing script was kept | — |
| App. C, all displayed identities | numerical checks | `paper/verify_formulas.py` | stdout |

The regenerated JSONs contain every run, not only the summaries reported in
the paper. Confidence intervals follow the figure scripts: 95% Student-t over
environment–seed blocks, after averaging the two estimators within each block.

## Appendix D: added after acceptance on 2026-10-06

Script: `experiments/run_exact_cone_toy.py` (`make toy`).
Recorded output: `experiments/exact_cone_toy_results.json`.
Source and analytic certificate: `paper/post_acceptance_toy.tex`.

| Quantity | Recorded field / value |
|---|---|
| Experiment design | `config`: seed 20261006, 100 trials, 20 calibration and 100 test rewards, target 0.8; each case also records its MDP, center, reference and policy support |
| 16-action exact width | `certified_width` = 0.397824734759316, analytically 2 tan(π/16) |
| 16-action radii | `intersection_radius` = 0.191342, `exact_max_radius` = 0.581596, `bridge_radius` = 0.780508; each common to all trials to shown precision |
| 16-action worst-case advantages | `bridge_worst_case_advantage` = 0.109746; `exact_worst_case_advantage` = 0.209202; both select action 0 to solver tolerance |
| 16-action containment | `stage1_containment` = 0.5039; `exact_containment` and `bridge_containment` = 1; `oracle_containment` = 0.8173, SE 0.0087 over independent trial means |
| 16-action oracle mean radius | `oracle_radius` = 0.428040; calibrated only on latent calibration rewards, never on test rewards |
| 4-action control | exact width 2, q = 0, exact maximum radius 1, bridge radius 2, ball worst-case advantages 0; `direct_cone_worst_case_advantage` = 0.5 |

The analytic endpoints certify the *full* cones for every action. Raw per-trial
values and solved occupancies are included; random draws do not estimate η or
the maximum score. Coverage is descriptive and can be conservative because
policy-based scores have ties. See [POST_ACCEPTANCE_CHANGES.md](POST_ACCEPTANCE_CHANGES.md)
for version preservation and the limits of this new diagnostic.
