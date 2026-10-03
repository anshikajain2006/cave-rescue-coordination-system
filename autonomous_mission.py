"""Building blocks for the autonomous search loop: a Thompson-sampling bandit that picks which chamber to search
next, and a tabular Q-learning navigator that learns how to drive there.

The loop itself (bandit -> Q-learning route -> HMM belief update -> bandit reward) is orchestrated in app.py,
which owns the bot state, the safety verifier and the HMM localiser.
"""
import math
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from cave_environment import CaveEnvironment
from cave_hmm import HMM_BORDER, CaveHMMLocaliser

Cell = Tuple[int, int]

# Chance that a sonar sweep of a chamber detects a survivor who is there. Below 1, an empty reading is evidence
# rather than proof, so the bandit has a reason to weigh revisiting a chamber against exploring a new one.
DETECTION_PROB = 0.85

MOVES: List[Cell] = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W (same order as the HMM and MDP)

# Moves over which an uncertain bot's preference for a chamber falls by a factor of e (see HMMCoupling)
REACH_DISTANCE_SCALE = 10.0


# =====================================================================
# BANDIT: which chamber to search next
# =====================================================================
class ThompsonBandit:
    """Beta-Bernoulli Thompson sampling over chambers.

    Each arm's Beta(alpha, beta) posterior is the belief that a search there finds a survivor; sampling from it
    picks uncertain arms often enough to explore while favouring arms that look promising.
    """

    def __init__(self, arms: Iterable[int], rng: Optional[np.random.Generator] = None,
                 hmm: Optional['HMMCoupling'] = None):
        """hmm couples the bandit to the HMM localiser: selection is weighted by how reliably each chamber can be
        reached from the believed position, and updates are scaled by position confidence. None: plain Thompson."""
        self.arms = list(arms)
        self.hmm = hmm
        self.alpha: Dict[int, float] = {arm: 1.0 for arm in self.arms}  # Uniform Beta(1, 1) prior
        self.beta: Dict[int, float] = {arm: 1.0 for arm in self.arms}
        self.pulls: Dict[int, int] = {arm: 0 for arm in self.arms}
        self.retired: Dict[int, str] = {}  # Arm -> reason it can no longer be chosen
        self.rng = rng or np.random.default_rng()
        self.last_samples: Dict[int, float] = {}
        self.last_weights: Dict[int, float] = {}  # Reach weight per arm at the last select()
        self.last_confidence: float = 1.0         # HMM position confidence used by the last update()

    def active_arms(self, exclude: Iterable[int] = ()) -> List[int]:
        skip = set(exclude)
        return [arm for arm in self.arms if arm not in self.retired and arm not in skip]

    def select(self, exclude: Iterable[int] = ()) -> Optional[int]:
        """Draws one sample per active arm and returns the arm with the highest weighted draw (None if none left).

        With an HMM coupling, each draw is multiplied by the arm's reach weight, so a bot unsure of where it is
        prefers chambers that are close under its belief; a well-localised bot is barely biased.
        """
        arms = self.active_arms(exclude)
        self.last_samples = {arm: float(self.rng.beta(self.alpha[arm], self.beta[arm])) for arm in arms}
        self.last_weights = self.hmm.reach_weights(arms) if self.hmm else {arm: 1.0 for arm in arms}
        return max(arms, key=lambda arm: self.last_samples[arm] * self.last_weights[arm]) if arms else None

    def update(self, arm: int, reward: int):
        """reward 1 (survivor found) raises the arm's posterior, 0 (nothing found) lowers it.

        With an HMM coupling, the pseudo-count added is the position confidence (0-1) at the time of the sweep:
        a reading taken while unsure the bot is really in that chamber counts for less.
        """
        weight = self.hmm.position_confidence() if self.hmm else 1.0
        self.last_confidence = weight
        self.pulls[arm] += 1
        if reward:
            self.alpha[arm] += weight
        else:
            self.beta[arm] += weight

    def retire(self, arm: int, reason: str):
        self.retired[arm] = reason

    def mean(self, arm: int) -> float:
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def std(self, arm: int) -> float:
        a, b = self.alpha[arm], self.beta[arm]
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))

    def table(self) -> List[dict]:
        return [{'chamber': arm, 'searches': self.pulls[arm], 'posterior mean': round(self.mean(arm), 3),
                 'uncertainty (std)': round(self.std(arm), 3),
                 'last sample': round(self.last_samples[arm], 3) if arm in self.last_samples else None,
                 'reach weight': round(self.last_weights[arm], 3) if arm in self.last_weights else None,
                 'status': self.retired.get(arm, 'active')} for arm in self.arms]


# =====================================================================
# HMM COUPLING: position belief -> bandit selection weights and update strength
# =====================================================================
class HMMCoupling:
    """Reads the HMM localiser for the bandit.

    position_confidence() = 1 - H(belief) / log(N) over the N open cells: 1 for a single cell, 0 when uniform.
    reach_weights(): for each chamber, the expected number of moves to it under the belief (every possible position
    weighted by its probability, distances along the navigator's grid), turned into
        weight = exp(-(1 - confidence) * expected_moves / REACH_DISTANCE_SCALE)
    so distance only matters in proportion to how unsure the bot is of its position.
    """

    def __init__(self, localiser_fn: Callable[[], CaveHMMLocaliser], navigator: 'QLearningNavigator',
                 targets: Dict[int, Cell], distance_scale: float = REACH_DISTANCE_SCALE):
        self.localiser_fn = localiser_fn  # Called on every use, so the latest belief is always read
        self.navigator = navigator
        self.targets = targets
        self.distance_scale = distance_scale

    def belief_by_cell(self) -> Dict[Cell, float]:
        """The belief over open cave cells as {(x, y): probability}, renormalised (HMM states also cover rock)"""
        loc = self.localiser_fn()
        open_cells = self.navigator.index
        cells: Dict[Cell, float] = {}
        for state, p in enumerate(loc.belief):
            if p > 0:
                r, c = loc.hmm_env.state_to_coord(state)
                cell = (c - HMM_BORDER, r - HMM_BORDER)
                if cell in open_cells:
                    cells[cell] = float(p)
        total = sum(cells.values())
        return {cell: p / total for cell, p in cells.items()} if total > 0 else {}

    def position_confidence(self, belief: Optional[Dict[Cell, float]] = None) -> float:
        belief = belief if belief is not None else self.belief_by_cell()
        n = len(self.navigator.cells)
        if not belief or n <= 1:
            return 0.0 if not belief else 1.0
        entropy = -sum(p * math.log(p) for p in belief.values() if p > 0)
        return float(min(max(1.0 - entropy / math.log(n), 0.0), 1.0))

    def expected_moves(self, arm: int, belief: Optional[Dict[Cell, float]] = None) -> float:
        belief = belief if belief is not None else self.belief_by_cell()
        distances = self.navigator.distances_to(self.targets[arm])
        unreachable = len(self.navigator.cells)  # Longer than any real route
        total = sum(belief.values()) or 1.0
        return sum(p * distances.get(cell, unreachable) for cell, p in belief.items()) / total

    def reach_weights(self, arms: Iterable[int]) -> Dict[int, float]:
        belief = self.belief_by_cell()
        uncertainty = 1.0 - self.position_confidence(belief)
        return {arm: math.exp(-uncertainty * self.expected_moves(arm, belief) / self.distance_scale) for arm in arms}


# =====================================================================
# Q-LEARNING NAVIGATOR: how to get there
# =====================================================================
class QLearningNavigator:
    """Tabular Q-learning over the cave grid, one Q-table per target cell (trained on first use, then cached).

    blocked cells (e.g. flooded) can be left but never entered: moving into one, into rock or off the map
    leaves the bot in place with a bump penalty.
    """
    STEP_REWARD = -1.0
    BUMP_REWARD = -5.0
    GOAL_REWARD = 100.0

    def __init__(self, env: CaveEnvironment, blocked: Iterable[Cell] = (), alpha: float = 0.5, gamma: float = 0.95,
                 epsilon: float = 0.3, episodes: int = 500, max_steps: int = 200, seed: Optional[int] = 0):
        self.cells: List[Cell] = sorted(env.cave_map)
        self.index: Dict[Cell, int] = {cell: i for i, cell in enumerate(self.cells)}
        self.blocked: Set[Cell] = set(blocked)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.episodes, self.max_steps = episodes, max_steps
        self.rng = np.random.default_rng(seed)
        self.q_tables: Dict[Cell, np.ndarray] = {}
        self._distances: Dict[Cell, Dict[Cell, int]] = {}
        # next_state[s, a]: where action a leads from state s (bumps stay put)
        n = len(self.cells)
        self.next_state = np.empty((n, len(MOVES)), dtype=int)
        self.bumps = np.zeros((n, len(MOVES)), dtype=bool)
        for s, (x, y) in enumerate(self.cells):
            for a, (dx, dy) in enumerate(MOVES):
                nxt = (x + dx, y + dy)
                if nxt in self.index and nxt not in self.blocked:
                    self.next_state[s, a] = self.index[nxt]
                else:
                    self.next_state[s, a], self.bumps[s, a] = s, True

    def reachable(self, start: Cell, target: Cell) -> bool:
        """Whether any route from start reaches target without entering a blocked cell (breadth-first search)"""
        if start not in self.index or target not in self.index or target in self.blocked:
            return False
        goal, frontier, seen = self.index[target], deque([self.index[start]]), {self.index[start]}
        while frontier:
            s = frontier.popleft()
            if s == goal:
                return True
            for s2 in self.next_state[s]:
                if int(s2) not in seen:
                    seen.add(int(s2))
                    frontier.append(int(s2))
        return False

    def distances_to(self, target: Cell) -> Dict[Cell, int]:
        """Fewest moves from each cell to target under the same rules as navigation (blocked cells cannot be
        entered); cells that cannot reach target are absent. Breadth-first search backwards from target, cached."""
        if target not in self._distances:
            dist: Dict[Cell, int] = {}
            if target in self.index and target not in self.blocked:
                predecessors: Dict[int, List[int]] = {}
                for s in range(len(self.cells)):
                    for a in range(len(MOVES)):
                        if not self.bumps[s, a]:
                            predecessors.setdefault(int(self.next_state[s, a]), []).append(s)
                dist[target] = 0
                frontier = deque([self.index[target]])
                while frontier:
                    s = frontier.popleft()
                    for p in predecessors.get(s, []):
                        if self.cells[p] not in dist:
                            dist[self.cells[p]] = dist[self.cells[s]] + 1
                            frontier.append(p)
            self._distances[target] = dist
        return self._distances[target]

    def _train(self, target: Cell, starts: List[int], episodes: int):
        q = self.q_tables.setdefault(target, np.zeros((len(self.cells), len(MOVES))))
        goal = self.index[target]
        for episode in range(episodes):
            s = starts[episode % len(starts)] if starts else int(self.rng.integers(len(self.cells)))
            epsilon = self.epsilon * (1 - episode / max(episodes, 1)) + 0.02  # Decays towards mostly greedy
            for _ in range(self.max_steps):
                if s == goal:
                    break
                a = int(self.rng.integers(len(MOVES))) if self.rng.random() < epsilon else int(np.argmax(q[s]))
                s2 = int(self.next_state[s, a])
                reward = self.BUMP_REWARD if self.bumps[s, a] else self.STEP_REWARD
                if s2 == goal:
                    reward += self.GOAL_REWARD
                    target_value = reward  # Terminal: no future value
                else:
                    target_value = reward + self.gamma * q[s2].max()
                q[s, a] += self.alpha * (target_value - q[s, a])
                s = s2

    def greedy_route(self, start: Cell, target: Cell) -> List[Cell]:
        """Follows argmax Q from start; stops at the target, on a revisit (loop) or after max_steps"""
        q = self.q_tables[target]
        route, seen = [start], {start}
        s = self.index[start]
        while route[-1] != target and len(route) <= self.max_steps:
            nxt = self.cells[int(self.next_state[s, int(np.argmax(q[s]))])]
            if nxt in seen:
                break
            route.append(nxt)
            seen.add(nxt)
            s = self.index[nxt]
        return route

    def route(self, start: Cell, target: Cell) -> List[Cell]:
        """Learned route from start to target, or [] if the target is unreachable or the policy has not converged.

        The first request for a target trains from random start cells (exploring starts); if the greedy policy
        still fails from this particular start, extra episodes are run from it before giving up.
        """
        if start == target:
            return [start]
        if not self.reachable(start, target):  # Checked first: training on an impossible goal never converges
            return []
        if target not in self.q_tables:
            self._train(target, starts=[], episodes=self.episodes)
        for _ in range(3):
            route = self.greedy_route(start, target)
            if route[-1] == target:
                return route
            self._train(target, starts=[self.index[start]], episodes=self.episodes // 3)
        return []


def scan_detects_survivor(env: CaveEnvironment, cell: Cell, rng: np.random.Generator,
                          detection_prob: float = DETECTION_PROB) -> bool:
    """One sonar sweep at cell: a survivor there is detected with detection_prob; there are no false positives"""
    info = env.cave_map.get(cell)
    return bool(info and info.is_survivor_location and rng.random() < detection_prob)
