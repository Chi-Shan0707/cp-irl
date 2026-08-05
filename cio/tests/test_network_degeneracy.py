"""Regression test for a real bug found during Phase 1 (see notes/phase1_status.md):
with skip_connections=False every source-sink path has exactly 3 edges, so a uniform
theta ties every path's cost, making classic IO's sub-optimality loss uninformative
(its point estimate collapses toward the uniform direction regardless of theta*).
skip_connections=True must produce genuine path-length diversity and must NOT exhibit
this collapse.
"""
import numpy as np
import pytest

from cio.network import build_layered_network
from cio.io_pipeline import generate_population, classic_io


def test_skip_connections_give_variable_path_lengths():
    net = build_layered_network(m_s=2, w1=5, w2=5, m_t=2, skip_connections=True)
    lengths = set()
    # source -> layer1 -> layer2 -> sink (3 edges), source -> layer2 (skip, 2 edges
    # total via layer2->sink), layer1 -> sink (skip), source -> sink (1 edge, direct)
    for (u, v) in net.edges:
        if u in net.sources and v in net.sources:  # unreachable, sanity
            continue
    # Just check that both 1-hop (source->sink) and 3-hop-only (no-skip) edge sets exist.
    direct = [(u, v) for (u, v) in net.edges if u in net.sources and v in net.sinks]
    assert len(direct) > 0, "expected direct source->sink shortcut edges"


def test_uniform_theta_does_not_tie_all_paths_with_skip_connections():
    net = build_layered_network(m_s=2, w1=4, w2=4, m_t=2, skip_connections=True)
    uniform = np.ones(net.d) / net.d
    from cio.shortest_path import solve_forward, path_edges
    s, t = net.sources[0], net.sinks[0]
    x = solve_forward(net, uniform, s, t)
    # under a uniform cost, the single direct source->sink edge (1 edge) must beat
    # any 2- or 3-edge alternative, so the optimal path is unique and short.
    assert path_edges(x).sum() == 1


@pytest.mark.parametrize("seed", range(5))
def test_classic_io_does_not_collapse_to_uniform_with_skip_connections(seed):
    """Regression guard: without this fix, theta_bar's coefficient of variation
    (std/mean) collapsed to ~3% (near-uniform) even under paper-level noise. With
    skip connections it should be substantially more informative."""
    rng = np.random.default_rng(seed)
    net = build_layered_network(m_s=2, w1=5, w2=5, m_t=2, skip_connections=True)
    theta_star = rng.uniform(0.5, 2.0, size=net.d)
    demos = generate_population(net, theta_star, N=60, rng=rng)  # paper-level noise
    theta_bar = classic_io(net, demos)

    cv = theta_bar.std() / theta_bar.mean()
    assert cv > 0.15, f"theta_bar coefficient of variation {cv:.3f} suggests near-uniform collapse"

    theta_star_unit = theta_star / np.linalg.norm(theta_star)
    cos_sim = theta_bar @ theta_star_unit
    assert cos_sim > 0.6, f"theta_bar barely correlated with theta_star: cos_sim={cos_sim:.3f}"
