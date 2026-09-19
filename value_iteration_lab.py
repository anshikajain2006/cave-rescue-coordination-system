"""
value_iteration_lab.py — LAB EXERCISE: implement Value Iteration
====================================================================

This is the file you edit for the value-iteration version of the lab.
Fill in _bellman_backup() and _extract_policy() below.

Algorithm you're implementing:

    V_0(s) = 0 for all s
    repeat:
        for every state s and action a:
            Q(s,a) = R(s,a) + discount * sum_s' P(a,s,s') * V(s)
        V_new(s) = max_a Q(s,a)          <- this is the "Bellman backup"
    until max_s |V_new(s) - V(s)| < self.tolerance, or max_sweeps reached

    policy(s) = argmax_a Q(s,a)          <- policy extraction, using the
                                            FINAL converged V

Note the key structural difference from policy_iteration_lab.py: there
is no separate "evaluate the current policy" step here — every sweep
immediately takes the max over actions, so V converges directly to V*
without ever fixing a policy along the way. Policy extraction only
happens once, at the very end.

Test yourself with:
    python compare_engines.py --engine-a vi_lab --engine-b vi_solution
which checks your implementation against a known-correct reference.
"""

import numpy as np

from policy_engine import PolicyEngine, PolicyEngineResult


class ValueIterationLabEngine(PolicyEngine):

    name = "Value Iteration (yours)"

    def __init__(self, tolerance=1e-8, max_sweeps=10000):
        self.tolerance = tolerance
        self.max_sweeps = max_sweeps

    def solve(self, P, R, discount=0.95, **kwargs):
        A, S, _ = P.shape
        value = np.zeros(S)

        sweep = 0
        for sweep in range(1, self.max_sweeps + 1):
            q_values = self._bellman_backup(P, R, discount, value)  # (S, A)
            new_value = q_values.max(axis=1)

            delta = np.max(np.abs(new_value - value))
            value = new_value

            if delta < self.tolerance:
                break

        q_values = self._bellman_backup(P, R, discount, value)
        policy = self._extract_policy(q_values)

        return PolicyEngineResult(policy, value, sweep, self.name)

    # ------------------------------------------------------------
    # TODO 1: BELLMAN BACKUP
    # ------------------------------------------------------------
    def _bellman_backup(self, P, R, discount, value):
        """
        Return Q, a numpy array of shape (S, A):
            Q[s, a] = R[s, a] + discount * sum_s' P[a, s, s'] * value[s']

        P has shape (A, S, S); R has shape (S, A); value has shape (S,).

        Hint: for a fixed action a, P[a] is an (S, S) matrix, so
        P[a].dot(value) gives you the (S,) vector of expected future
        values for every starting state under action a in one line.
        Do this for each of the A actions and stack the results into
        columns of a (S, A) array (np.stack(..., axis=1) is your friend
        here — you did exactly this in value_iteration_solution.py if
        you've read it, but try from scratch first).
        """
        A, S, _ = P.shape

        # TODO: replace this with the real (S, A) Q-value computation.
        # For each action a: Q[:, a] = R[:, a] + discount * P[a] @ value
        # P[a] is (S, S), so P[a].dot(value) gives (S,) expected future values.
        # Stack A columns into (S, A).
        return np.stack(
            [R[:, a] + discount * P[a].dot(value) for a in range(A)],
            axis=1,
        )

    # ------------------------------------------------------------
    # TODO 2: POLICY EXTRACTION
    # ------------------------------------------------------------
    def _extract_policy(self, q_values):
        """
        Given the FINAL Q, shape (S, A), return policy: shape (S,), the
        action index that maximizes Q in each state.
        """
        # TODO: replace this placeholder with an argmax over the action axis.
        return q_values.argmax(axis=1)
