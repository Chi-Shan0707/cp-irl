"""Behavior cloning baseline (plan.md Phase 5 baseline list): the simplest possible
non-reward-based imitation policy -- majority-vote the population's actions per
state, with no notion of reward, uncertainty, or robustness at all. Included to
distinguish "CP-IRL's benefit comes from reasoning about a calibrated reward set"
from "any form of aggregating demonstrator behavior helps."
"""
from __future__ import annotations

import numpy as np

from envs.mdp import TabularMDP


def behavior_cloning_policy(mdp: TabularMDP, policies: list[np.ndarray]) -> np.ndarray:
    """Per-state majority vote across the training population's policies. Ties
    broken by smallest action index (matching value_iteration's convention).
    Returns a deterministic policy, shape (S,).
    """
    S, A = mdp.S, mdp.A
    votes = np.zeros((S, A), dtype=int)
    for pi in policies:
        votes[np.arange(S), pi] += 1
    return np.argmax(votes, axis=1)
