"""Layered shortest-path network used to reproduce the CIO shortest-path experiment.

Structure: m_s source nodes -> w1 hidden nodes -> w2 hidden nodes -> m_t sink nodes,
fully connected between consecutive layers. Every edge (u, v) with u a tail node and
v a head node contributes one column to the node-edge incidence matrix A, defined so
that (A @ x)[i] = inflow(i) - outflow(i) for an edge-flow vector x.

Context u = (s, t), a source-sink pair; d = number of edges.

With `skip_connections=False` (default), every source-sink path has exactly 3 edges
(source -> layer1 -> layer2 -> sink). This is a real identifiability trap for
sub-optimality-loss IO: since every feasible path has the same edge COUNT, a perfectly
uniform cost vector theta = c * ones(d) makes every path cost exactly 3c, i.e. every
path is simultaneously optimal (zero sub-optimality loss) regardless of which paths
were actually observed. classic_io's point estimate then collapses toward this
uninformative uniform direction (found empirically — see notes/network_degeneracy.md)
rather than recovering theta*'s actual direction. `skip_connections=True` adds
source->layer2 and layer1->sink "shortcut" edges, giving genuine path-length diversity
(1, 2, and 3-edge paths coexist) so a uniform theta no longer ties every alternative,
which is what real road networks look like and is required for classic IO's
sub-optimality loss to be informative at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShortestPathNetwork:
    edges: list[tuple[int, int]]  # (tail, head) node indices
    A: np.ndarray  # (num_nodes, num_edges)
    sources: list[int]
    sinks: list[int]
    num_nodes: int

    @property
    def d(self) -> int:
        return len(self.edges)

    def rhs(self, s: int, t: int) -> np.ndarray:
        b = np.zeros(self.num_nodes)
        b[s] = -1.0
        b[t] = 1.0
        return b

    def all_od_pairs(self) -> list[tuple[int, int]]:
        return [(s, t) for s in self.sources for t in self.sinks]


def build_layered_network(m_s: int, w1: int, w2: int, m_t: int,
                           skip_connections: bool = False) -> ShortestPathNetwork:
    sources = list(range(m_s))
    layer1 = list(range(m_s, m_s + w1))
    layer2 = list(range(m_s + w1, m_s + w1 + w2))
    sinks = list(range(m_s + w1 + w2, m_s + w1 + w2 + m_t))
    num_nodes = m_s + w1 + w2 + m_t

    edges: list[tuple[int, int]] = []
    for u in sources:
        for v in layer1:
            edges.append((u, v))
    for u in layer1:
        for v in layer2:
            edges.append((u, v))
    for u in layer2:
        for v in sinks:
            edges.append((u, v))
    if skip_connections:
        for u in sources:
            for v in layer2:
                edges.append((u, v))
        for u in layer1:
            for v in sinks:
                edges.append((u, v))
        for u in sources:
            for v in sinks:
                edges.append((u, v))

    A = np.zeros((num_nodes, len(edges)))
    for e, (u, v) in enumerate(edges):
        A[v, e] += 1.0
        A[u, e] -= 1.0

    return ShortestPathNetwork(edges=edges, A=A, sources=sources, sinks=sinks,
                                num_nodes=num_nodes)
