"""
policy_iteration_lab.py — LAB EXERCISE: implement Policy Iteration
=====================================================================

This is the file you edit. Fill in _policy_evaluation() and
_policy_improvement() below. Everything else (the environment, the
tensors, the reference solver, the comparison harness) is already
provided and should NOT need to change.

Algorithm you're implementing:

    1. Start with an arbitrary policy (below: "always take action 0").
    2. POLICY EVALUATION
       Compute V(s) for the CURRENT policy by repeatedly applying:
           V(s) <- R(s, policy[s]) + discount * sum_s' P(policy[s], s, s') * V(s')
       until V stops changing much (< self.eval_tolerance), or you've
       done self.max_eval_sweeps sweeps.
    3. POLICY IMPROVEMENT
       For every state s, set:
           policy_new(s) = argmax_a [ R(s,a) + discount * sum_s' P(a,s,s') * V(s') ]
    4. If policy_new == policy for every state, stop — you've converged.
       Otherwise set policy = policy_new and go back to step 2.

Test yourself with:
    python compare_engines.py
which runs this against value_iteration_solution.py (a known-correct
reference) and reports policy agreement + average return.
"""

import numpy as np

from policy_engine import PolicyEngine, PolicyEngineResult


class PolicyIterationEngine(PolicyEngine):

    name = "Policy Iteration (yours)"

    def __init__(self, eval_tolerance=1e-6, max_eval_sweeps=1000):
        self.eval_tolerance = eval_tolerance
        self.max_eval_sweeps = max_eval_sweeps

    def solve(self, P, R, discount=0.95, **kwargs):
        A, S, _ = P.shape

        # Arbitrary starting policy — "always take action 0 (UP)".
        policy = np.zeros(S, dtype=int)
        value = np.zeros(S)

        iterations = 0
        while True:
            iterations += 1

            value = self._policy_evaluation(policy, P, R, discount)
            new_policy = self._policy_improvement(value, P, R, discount)

            if np.array_equal(new_policy, policy):
                policy = new_policy
                break

            policy = new_policy

        return PolicyEngineResult(policy, value, iterations, self.name)

    # ------------------------------------------------------------
    # TODO 1: POLICY EVALUATION
    # ------------------------------------------------------------
    def _policy_evaluation(self, policy, P, R, discount):
        """
        Return V, a numpy array of shape (S,): the value function for
        the given (fixed) policy.

        P has shape (A, S, S); R has shape (S, A); policy has shape (S,).

        Hint: for the FIXED policy, the transition probabilities you
        need at state s are P[policy[s], s, :], and the reward is
        R[s, policy[s]]. You can build these once per sweep as:

            P_pi = P[policy, np.arange(S), :]   # shape (S, S)
            R_pi = R[np.arange(S), policy]      # shape (S,)

        then iterate:
            V_new = R_pi + discount * P_pi.dot(V)
        until max(|V_new - V|) < self.eval_tolerance.
        """
        S = P.shape[1]
        value = np.zeros(S)

        # TODO: replace this with the iterative policy evaluation loop
        # described above.
        P_pi = P[policy, np.arange(S), :]   # (S, S)
        R_pi = R[np.arange(S), policy]      # (S,)
 
        for _ in range(self.max_eval_sweeps):
            new_value = R_pi + discount * P_pi.dot(value)
            if np.max(np.abs(new_value - value)) < self.eval_tolerance:
                value = new_value
                break
            value = new_value
 
        return value

    # ------------------------------------------------------------
    # TODO 2: POLICY IMPROVEMENT
    # ------------------------------------------------------------
    def _policy_improvement(self, value, P, R, discount):
        """
        Return new_policy, a numpy array of shape (S,): for every state
        s, the action that maximizes
            R(s,a) + discount * sum_s' P(a,s,s') * V(s')

        Hint: this is the same Q(s,a) computation used in
        value_iteration_solution.py — build a (S, A) array of Q-values,
        one column per action, then take argmax over the action axis.
        Try NOT to look at that file until after you've attempted this
        yourself.
        """
        A, S, _ = P.shape

        # TODO: replace this placeholder with real Q-value computation
        # and an argmax over actions.
        # Q[s, a] = R[s, a] + discount * sum_s' P[a, s, s'] * V[s']
        q_values = np.stack(
            [R[:, a] + discount * P[a].dot(value) for a in range(A)],
            axis=1,
        )  # shape (S, A)
 
        return q_values.argmax(axis=1)