"""
policy_engine.py — the common interface every MDP solver implements
=====================================================================

Both policy_iteration_lab.py (what you write) and
value_iteration_solution.py (the working reference you test against)
implement `PolicyEngine.solve(P, R, discount)`. Because they share this
interface, compare_engines.py and game_gui.py can use either one
interchangeably without knowing which algorithm is underneath.
"""

from abc import ABC, abstractmethod


class PolicyEngineResult:
    """
    policy           : np.ndarray, shape (S,), int — policy[s] is the
                        action index (0-3) to take in state s
    value_function   : np.ndarray, shape (S,), float — V(s)
    iterations       : int — outer loop count (policy iterations, or
                        value-iteration sweeps), reported for comparison
    engine_name      : str — for printing/logging
    """
    def __init__(self, policy, value_function, iterations, engine_name):
        self.policy = policy
        self.value_function = value_function
        self.iterations = iterations
        self.engine_name = engine_name

    def __repr__(self):
        return (f"PolicyEngineResult(engine={self.engine_name!r}, "
                f"iterations={self.iterations})")


class PolicyEngine(ABC):
    """
    Every solver takes MDP tensors P (A,S,S) and R (S,A) plus a discount
    factor, and returns a PolicyEngineResult. Nothing here knows what a
    "grid", a "wall", or a "treasure" is — that separation is the point.
    """

    name = "AbstractEngine"

    @abstractmethod
    def solve(self, P, R, discount=0.95, **kwargs) -> PolicyEngineResult:
        raise NotImplementedError
