"""Independent checks for the post-acceptance polygon diagnostic."""
import numpy as np
import pytest
from scipy.optimize import linprog

from experiments.run_exact_cone_toy import cone_endpoints, polygon_mdp, run_case


@pytest.mark.parametrize("n_actions", [4, 16])
def test_analytic_endpoints_match_full_cone_linear_programs(n_actions):
    mdp = polygon_mdp(n_actions)
    phi = mdp.phi[0]
    for action in [0, 1, n_actions - 1]:
        endpoints = cone_endpoints(mdp, action)
        # Inside this Bellman cone central symmetry gives
        # span(theta)=2*phi[action]@theta/(1-discount). This equality plus
        # ALL cone inequalities defines the entire normalized feasible set.
        for direction in phi:
            result = linprog(-direction, A_ub=phi - phi[action],
                             b_ub=np.zeros(n_actions), A_eq=phi[action][None],
                             b_eq=[(1 - mdp.gamma) / 2], bounds=[(None, None)] * 2,
                             method="highs")
            assert result.success
            np.testing.assert_allclose(-result.fun, (endpoints @ direction).max(), atol=1e-10)


def test_positive_certificate_and_set_shape_control():
    positive = run_case(16, 3, 20, 20, .8, 20261006)
    control = run_case(4, 3, 20, 20, .8, 20261006, control=True)
    assert positive["summary"]["positive_worst_case_advantage"] == 1
    assert positive["summary"]["bridge_worst_case_advantage"] > .1
    assert positive["summary"]["reference_feature_distance"] > .99
    assert control["summary"]["bridge_worst_case_advantage"] == 0
    assert control["summary"]["reference_feature_distance"] == 0
    np.testing.assert_allclose(control["direct_cone_worst_case_advantage"], .5, atol=1e-10)


def test_invalid_polygon_is_rejected():
    with pytest.raises(ValueError):
        polygon_mdp(6)
