"""
cave_mdp.py: adaptive pilot layer built on the Lab 3 MDP solvers.

The cave grid becomes an MDP (states = free cells, 4 actions with the Lab 3 80/10/10 slip model),
which PolicyIterationEngine / ValueIterationLabEngine solve through their shared
solve(P, R, discount) interface. If the Lab 3 files are missing, a built-in value iteration is used.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from cave_environment import CaveEnvironment

# Policy iteration keeps the current action unless another is better by more than this: above the
# lab's evaluation noise, far below any real reward difference.
TIE_TOLERANCE = 1e-4

try:
    from policy_iteration_lab import PolicyIterationEngine
    from value_iteration_lab import ValueIterationLabEngine

    class _TieStablePolicyIteration(PolicyIterationEngine):
        """Lab 3 policy iteration with the standard tie-break added.

        The lab's argmax improvement can flip forever between equally good actions (evaluation noise
        reorders exact ties), and its solve() loop has no iteration cap, so blocked cells or equal-length
        routes made it hang. Keeping the current action unless strictly better guarantees termination.
        The lab's own solve() loop and policy evaluation are used unchanged.
        """
        name = "Policy Iteration (Lab 3)"

        def _policy_evaluation(self, policy, P, R, discount):
            self._current_policy = policy
            return super()._policy_evaluation(policy, P, R, discount)

        def _policy_improvement(self, value, P, R, discount):
            greedy = super()._policy_improvement(value, P, R, discount)
            q_values = R + discount * np.einsum('ast,t->sa', P, value)
            rows = np.arange(len(greedy))
            keep = q_values[rows, self._current_policy] >= q_values[rows, greedy] - TIE_TOLERANCE
            return np.where(keep, self._current_policy, greedy)

    class _LabValueIteration(ValueIterationLabEngine):
        name = "Value Iteration (Lab 3)"

    LAB3_AVAILABLE = True
except ImportError:
    LAB3_AVAILABLE = False

Cell = Tuple[int, int]

# Same action indices as Lab 3 (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT), expressed as cave directions
ACTION_NAMES = ['NORTH', 'SOUTH', 'WEST', 'EAST']
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]            # (dx, dy); north is y-1
PERPENDICULAR = [[2, 3], [2, 3], [0, 1], [0, 1]]      # Slip directions for each action
INTENDED_PROB, SLIP_PROB = 0.8, 0.1

SURVIVOR_REWARD = 100.0
HIGH_WATER_PENALTY = -10.0
HIGH_WATER_THRESHOLD = 0.6
STEP_COST = -1.0
# Close to 1 on purpose: with discount g, "bump a wall forever" costs only -1/(1-g) in total (-20 at 0.95),
# which beats wading through -10 flooded cells, so the pilot refused most detours. At 0.999 it costs -1000,
# so a rescue route wins whenever one exists. Terminal survivor cells keep convergence fast.
DISCOUNT = 0.999

ALGORITHMS = ('policy_iteration', 'value_iteration')


class _BuiltinValueIteration:
    """From-scratch value iteration used when the Lab 3 engines can't be imported"""
    name = "Value Iteration (built-in fallback)"

    def __init__(self, tolerance=1e-8, max_sweeps=10000):
        self.tolerance = tolerance
        self.max_sweeps = max_sweeps

    def solve(self, P, R, discount=0.95):
        value = np.zeros(P.shape[1])
        sweep = 0
        for sweep in range(1, self.max_sweeps + 1):
            q_values = R + discount * np.einsum('ast,t->sa', P, value)
            new_value = q_values.max(axis=1)
            delta = np.max(np.abs(new_value - value))
            value = new_value
            if delta < self.tolerance:
                break
        q_values = R + discount * np.einsum('ast,t->sa', P, value)
        return _Result(q_values.argmax(axis=1), value, sweep, self.name)


@dataclass
class _Result:
    policy: np.ndarray
    value_function: np.ndarray
    iterations: int
    engine_name: str


@dataclass
class MDPSolution:
    policy: Dict[Cell, str]         # Best action name per free cell
    values: Dict[Cell, float]       # V(s) per free cell
    engine_name: str
    iterations: int
    targets: List[Cell]             # Cells rewarded with +100 (terminal)
    blocked: List[Cell]             # Free cells treated as walls for this solve


class CaveMDPPilot:
    """Plans survivor-seeking policies over the cave and replans when cells become impassable"""

    def __init__(self, cave_env: CaveEnvironment, discount: float = DISCOUNT):
        self.cave_env = cave_env
        self.discount = discount
        self.last_solution: Optional[MDPSolution] = None

    def compute_policy(self, algorithm: str = "policy_iteration", targets: Optional[Iterable[Cell]] = None,
                       blocked_cells: Iterable[Cell] = ()) -> Dict[Cell, str]:
        """Solves the cave MDP and returns {(x, y): best action name}; targets default to all survivors"""
        targets = list(targets) if targets is not None else list(self.cave_env.survivor_locations)
        blocked = sorted(set(blocked_cells) & set(self.cave_env.cave_map))
        states = [cell for cell in self.cave_env.cave_map if cell not in set(blocked)]
        index = {cell: i for i, cell in enumerate(states)}
        P, R = self._build_tensors(states, index, set(targets))

        result = self._engine(algorithm).solve(P, R, discount=self.discount)
        policy = {cell: ACTION_NAMES[int(result.policy[i])] for cell, i in index.items() if cell not in targets}
        values = {cell: float(result.value_function[i]) for cell, i in index.items()}
        self.last_solution = MDPSolution(policy, values, result.engine_name, int(result.iterations),
                                         [t for t in targets if t in index], blocked)
        return policy

    def replan(self, blocked_cells: Iterable[Cell], target: Optional[Cell] = None,
               algorithm: str = "policy_iteration") -> Dict[Cell, str]:
        """Recomputes the policy with blocked_cells as walls (flooding, collapse); target defaults to all survivors"""
        return self.compute_policy(algorithm, targets=None if target is None else [target], blocked_cells=blocked_cells)

    @staticmethod
    def get_action(position: Cell, policy: Dict[Cell, str]) -> Optional[str]:
        """Recommended move at position, or None at a target, a blocked cell, or rock"""
        return policy.get((position[0], position[1]))

    def trace_route(self, start: Cell, policy: Dict[Cell, str], max_steps: int = 500) -> List[Cell]:
        """Follows the policy's intended moves from start; stops at a target, a dead end or a loop"""
        if self.last_solution is None:
            raise RuntimeError("compute_policy() or replan() must run before trace_route()")
        open_cells = self.last_solution.values
        route: List[Cell] = [(start[0], start[1])]
        seen = {route[0]}
        while len(route) <= max_steps:
            action = policy.get(route[-1])
            if action is None:
                break
            dx, dy = MOVES[ACTION_NAMES.index(action)]
            nxt = (route[-1][0] + dx, route[-1][1] + dy)
            if nxt not in open_cells or nxt in seen:  # Wall, blocked cell or cycle
                break
            route.append(nxt)
            seen.add(nxt)
        return route

    # -----------------------------------------------------------------
    def _engine(self, algorithm: str):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
        if not LAB3_AVAILABLE:
            return _BuiltinValueIteration()
        return _TieStablePolicyIteration() if algorithm == 'policy_iteration' else _LabValueIteration()

    def _build_tensors(self, states: List[Cell], index: Dict[Cell, int], targets: set):
        """P[a, s, s'] with the Lab 3 slip model; R[s, a] = expected reward of the landing cell"""
        S, A = len(states), len(ACTION_NAMES)
        P = np.zeros((A, S, S))
        R = np.zeros((S, A))
        cave_map = self.cave_env.cave_map

        def landing_reward(cell: Cell) -> float:
            reward = STEP_COST
            if cave_map[cell].water_level > HIGH_WATER_THRESHOLD:
                reward += HIGH_WATER_PENALTY
            if cell in targets:
                reward += SURVIVOR_REWARD
            return reward

        for s, (x, y) in enumerate(states):
            if (x, y) in targets:  # Absorbing: the rescue reward is collected once
                P[:, s, s] = 1.0
                continue
            for a in range(A):
                for actual, prob in ((a, INTENDED_PROB), (PERPENDICULAR[a][0], SLIP_PROB),
                                     (PERPENDICULAR[a][1], SLIP_PROB)):
                    dx, dy = MOVES[actual]
                    landing = (x + dx, y + dy) if (x + dx, y + dy) in index else (x, y)  # Bump: stay put
                    P[a, s, index[landing]] += prob
                    R[s, a] += prob * landing_reward(landing)
        return P, R
