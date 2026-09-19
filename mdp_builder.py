"""
mdp_builder.py — turn a TreasureHuntEnvironment instance into MDP tensors
==========================================================================

This is the ONLY place in the codebase that constructs P and R. Every
policy engine (policy_iteration_lab.py, value_iteration_solution.py)
receives P and R already built — engines never look at pygame, at
`env.walls`, or at grid coordinates. See MDP_INPUT_OUTPUT_FORMAT.md for
the full convention this follows.

Quick summary:
  states, state_to_index = enumerate_states(env)   # S = len(states)
  P, R, states, state_to_index = build_mdp_tensors(env)

  P.shape == (A, S, S)   P[a, s, s'] = probability of landing in s'
                         given you took action a in state s
  R.shape == (S, A)      R[s, a]    = expected immediate reward for
                         taking action a in state s (averaged over the
                         stochastic outcome, since the actual reward
                         depends on which of the 3 possible actual
                         actions the environment samples)
"""

import numpy as np

NUM_ACTIONS = 4


def enumerate_states(env):
    """
    Deterministic list of every (position, has_treasure) state on this
    environment's (possibly randomized) grid, plus the state -> index
    map every other module uses. Order depends only on env.rows/env.cols
    /env.walls, so it is stable for a given layout but will differ
    between two randomly generated layouts — that's expected.
    """
    states = env.all_states()
    state_to_index = {state: i for i, state in enumerate(states)}
    return states, state_to_index


def build_mdp_tensors(env):
    """
    Build (P, R) for the given environment instance.

    Terminal handling: once `position == env.goal`, the state is made
    absorbing — every action loops back to the same state with reward 0.
    Without this, policy/value iteration would keep trying to extract
    reward from a state that no longer exists in play.
    """
    states, state_to_index = enumerate_states(env)
    S = len(states)
    A = NUM_ACTIONS

    P = np.zeros((A, S, S))
    R = np.zeros((S, A))

    for s_idx, state in enumerate(states):
        if env.is_terminal(state):
            for action in range(A):
                P[action, s_idx, s_idx] = 1.0
                R[s_idx, action] = 0.0
            continue

        for action in range(A):
            expected_reward = 0.0
            for actual_action, prob in env.action_outcomes(action):
                next_state, reward = env.simulate_transition(state, actual_action)
                next_idx = state_to_index[next_state]
                P[action, s_idx, next_idx] += prob
                expected_reward += prob * reward
            R[s_idx, action] = expected_reward

    _sanity_check(P)
    return P, R, states, state_to_index


def _sanity_check(P):
    """Every P[a, s, :] must sum to 1 — a broken transition function is
    the single most common bug in a lab like this, so fail loudly."""
    row_sums = P.sum(axis=2)
    if not np.allclose(row_sums, 1.0):
        bad = np.argwhere(~np.isclose(row_sums, 1.0))
        raise ValueError(
            f"P is not a valid transition tensor — rows not summing to 1 "
            f"at (action, state) pairs: {bad[:5].tolist()} (showing up to 5)"
        )
