"""Forward LP / dual LP strong-duality check, and sanity of the network builder."""
import numpy as np
import pytest

from cio.network import build_layered_network
from cio.shortest_path import solve_forward, dual_potentials, path_edges


def test_network_edge_count():
    net = build_layered_network(m_s=3, w1=6, w2=6, m_t=3)
    # m_s*w1 + w1*w2 + w2*m_t
    assert net.d == 3 * 6 + 6 * 6 + 6 * 3
    assert net.num_nodes == 3 + 6 + 6 + 3


@pytest.mark.parametrize("seed", range(15))
def test_strong_duality_holds(seed):
    rng = np.random.default_rng(seed)
    net = build_layered_network(m_s=2, w1=4, w2=4, m_t=2)
    theta = rng.uniform(0.1, 5.0, size=net.d)  # positive costs, well-posed DAG shortest path
    s, t = net.sources[0], net.sinks[-1]

    x = solve_forward(net, theta, s, t)
    y = dual_potentials(net, theta, s, t)

    primal_val = theta @ x
    dual_val = net.rhs(s, t) @ y
    assert primal_val == pytest.approx(dual_val, abs=1e-5)


@pytest.mark.parametrize("seed", range(10))
def test_forward_solution_is_a_single_path(seed):
    """DAG shortest path with positive costs -> integral 0/1 optimal solution."""
    rng = np.random.default_rng(seed)
    net = build_layered_network(m_s=2, w1=4, w2=4, m_t=2)
    theta = rng.uniform(0.1, 5.0, size=net.d)
    s, t = net.sources[0], net.sinks[0]

    x = solve_forward(net, theta, s, t)
    on_path = path_edges(x)
    # exactly one edge leaving each layer boundary on the path (3-hop path: src->l1->l2->sink)
    assert on_path.sum() == pytest.approx(3, abs=0)
    assert np.all((x < 1e-6) | (np.abs(x - 1) < 1e-6))
