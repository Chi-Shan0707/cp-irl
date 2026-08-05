"""Sanity checks for the robust forward problem (RFO)."""
import numpy as np
import pytest

from cio.network import build_layered_network
from cio.shortest_path import solve_forward
from cio.support_function import support_value
from cio.robust import solve_rfo


@pytest.mark.parametrize("seed", range(8))
def test_rfo_value_at_least_fo_value_under_theta_bar(seed):
    """RFO hedges against a whole set containing theta_bar, so its worst-case cost
    must be >= the plain FO cost achieved by just solving under theta_bar directly
    (the plain-FO solution is one feasible point for RFO's inner max, evaluated at
    theta_bar itself, but RFO's objective is the WORST case over the whole cap, so
    RFO's optimal worst-case value is >= h_C of the FO-optimal x, which is >= the
    plain value since h_C(x) >= theta_bar . x for any x, with equality only if the
    cap has just this one direction, i.e. alpha=0)."""
    rng = np.random.default_rng(seed)
    net = build_layered_network(m_s=2, w1=4, w2=4, m_t=2)
    theta_bar = rng.uniform(0.5, 2.0, size=net.d)
    theta_bar /= np.linalg.norm(theta_bar)
    alpha = rng.uniform(0.1, np.pi / 2 - 0.1)
    s, t = net.sources[0], net.sinks[0]

    x_fo = solve_forward(net, theta_bar, s, t)
    x_rfo = solve_rfo(net, theta_bar, alpha, s, t)

    val_fo_worst = support_value(x_fo, theta_bar, alpha)
    val_rfo_worst = support_value(x_rfo, theta_bar, alpha)

    # RFO minimizes the worst case, so it should do at least as well as the FO
    # solution's worst case.
    assert val_rfo_worst <= val_fo_worst + 1e-4


@pytest.mark.parametrize("seed", range(5))
def test_rfo_worst_case_value_grows_with_alpha(seed):
    """A bigger uncertainty set (larger alpha) can only make the worst case >= smaller alpha's."""
    rng = np.random.default_rng(seed + 100)
    net = build_layered_network(m_s=2, w1=4, w2=4, m_t=2)
    theta_bar = rng.uniform(0.5, 2.0, size=net.d)
    theta_bar /= np.linalg.norm(theta_bar)
    s, t = net.sources[0], net.sinks[0]

    x_small = solve_rfo(net, theta_bar, alpha=0.1, s=s, t=t)
    x_big = solve_rfo(net, theta_bar, alpha=1.0, s=s, t=t)

    val_small = support_value(x_small, theta_bar, 0.1)
    val_big = support_value(x_big, theta_bar, 1.0)

    assert val_big >= val_small - 1e-4


def test_rfo_reduces_to_fo_as_alpha_to_zero():
    rng = np.random.default_rng(7)
    net = build_layered_network(m_s=2, w1=4, w2=4, m_t=2)
    theta_bar = rng.uniform(0.5, 2.0, size=net.d)
    theta_bar /= np.linalg.norm(theta_bar)
    s, t = net.sources[0], net.sinks[0]

    x_fo = solve_forward(net, theta_bar, s, t)
    x_rfo = solve_rfo(net, theta_bar, alpha=1e-4, s=s, t=t)

    assert theta_bar @ x_rfo == pytest.approx(theta_bar @ x_fo, abs=1e-2)
