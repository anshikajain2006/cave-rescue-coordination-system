"""
env.py — Treasure Hunt MDP environment (engine layer)
======================================================

This module has NO pygame dependency. It is the "engine": the game's
state space, dynamics, and rewards. Anything that needs to reason about
the MDP mathematically (mdp_builder.py, the policy engines,
compare_engines.py) imports from here. The GUI (game_gui.py) also
imports from here, but only uses it to play episodes — it never touches
the MDP math.

Design choices that matter for the lab
---------------------------------------
1. `simulate_transition(state, actual_action)` is a PURE function: given
   an arbitrary (position, has_treasure) state and an actual action, it
   returns (next_state, reward) without touching `self.position` or any
   other mutable instance attribute. This is what makes it possible to
   enumerate the full transition/reward tensors for tensors P and R
   (mdp_builder.py) — the same code path used during live play in
   step() is reused for tensor construction, so the two can never drift
   apart.

2. `generate_random_layout(...)` builds a fresh grid (start / treasure /
   goal / walls / danger) each time it's called, subject to a
   solvability check (BFS reachability start -> treasure -> goal). This
   is what "the game has a more random layout each time" means in
   practice: nothing in the policy engines is allowed to hardcode grid
   geometry, since the geometry changes every episode.
"""

import random
from collections import deque


ACTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

MOVES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

# For a chosen action, the two actions it can "slip" into.
PERPENDICULAR_ACTIONS = {
    0: [2, 3],
    1: [2, 3],
    2: [0, 1],
    3: [0, 1],
}


class TreasureHuntEnvironment:

    def __init__(self, layout=None, rng=None):
        self.rng = rng or random.Random()

        if layout is None:
            layout = self.generate_random_layout(rng=self.rng)

        self.rows = layout["rows"]
        self.cols = layout["cols"]
        self.start = layout["start"]
        self.treasure = layout["treasure"]
        self.goal = layout["goal"]
        self.walls = set(layout["walls"])
        self.danger = set(layout["danger"])

        self.reset()

    # --------------------------------------------------------
    # Random layout generation
    # --------------------------------------------------------

    @staticmethod
    def generate_random_layout(rows=4, cols=4, num_walls=2, num_danger=1,
                                rng=None, max_attempts=500):
        """
        Build a random, solvable layout: a dict with keys
        rows, cols, start, treasure, goal, walls, danger.

        Solvable means: start -> treasure is reachable avoiding walls,
        AND treasure -> goal is reachable avoiding walls. Danger squares
        are allowed to sit on the only path (they cost reward, they
        don't block movement).
        """
        rng = rng or random.Random()
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]

        for _ in range(max_attempts):
            rng.shuffle(all_cells)
            start, treasure, goal = all_cells[0], all_cells[1], all_cells[2]
            remaining = all_cells[3:]

            walls = set(remaining[:num_walls])
            danger_pool = [c for c in remaining[num_walls:]
                           if c not in (start, treasure, goal)]
            danger = set(danger_pool[:num_danger])

            layout = dict(rows=rows, cols=cols, start=start,
                          treasure=treasure, goal=goal,
                          walls=walls, danger=danger)

            if TreasureHuntEnvironment._layout_is_solvable(layout):
                return layout

        raise RuntimeError(
            "Could not generate a solvable random layout — "
            "try fewer walls or a bigger grid."
        )

    @staticmethod
    def _layout_is_solvable(layout):
        def bfs_reachable(source, blocked, rows, cols):
            visited = {source}
            queue = deque([source])
            while queue:
                r, c = queue.popleft()
                for dr, dc in MOVES.values():
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < rows and 0 <= nxt[1] < cols
                            and nxt not in blocked and nxt not in visited):
                        visited.add(nxt)
                        queue.append(nxt)
            return visited

        rows, cols, walls = layout["rows"], layout["cols"], layout["walls"]

        reachable_from_start = bfs_reachable(layout["start"], walls, rows, cols)
        if layout["treasure"] not in reachable_from_start:
            return False

        reachable_from_treasure = bfs_reachable(layout["treasure"], walls, rows, cols)
        if layout["goal"] not in reachable_from_treasure:
            return False

        return True

    def randomize_layout(self, rows=None, cols=None, num_walls=2, num_danger=1):
        """Replace this environment's layout with a brand new random one."""
        layout = self.generate_random_layout(
            rows=rows or self.rows,
            cols=cols or self.cols,
            num_walls=num_walls,
            num_danger=num_danger,
            rng=self.rng,
        )
        self.rows = layout["rows"]
        self.cols = layout["cols"]
        self.start = layout["start"]
        self.treasure = layout["treasure"]
        self.goal = layout["goal"]
        self.walls = set(layout["walls"])
        self.danger = set(layout["danger"])
        self.reset()

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    def reset(self):
        self.position = self.start
        self.has_treasure = False
        self.total_reward = 0
        self.steps = 0
        self.done = False
        self.last_chosen_action = None
        self.last_actual_action = None
        self.last_reward = 0
        self.message = "Collect the treasure, then reach the goal."
        return self.get_state()

    def get_state(self):
        """The full MDP state: position AND whether treasure is held."""
        return (self.position, self.has_treasure)

    def all_states(self):
        """Every (position, has_treasure) state reachable on this grid
        (walls excluded). Order is deterministic given the layout, which
        is what mdp_builder.py relies on to build state_to_index."""
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                position = (row, col)
                if position in self.walls:
                    continue
                for has_treasure in (False, True):
                    states.append((position, has_treasure))
        return states

    def is_terminal(self, state):
        position, _ = state
        return position == self.goal

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    def is_valid_position(self, position):
        row, col = position
        inside_grid = 0 <= row < self.rows and 0 <= col < self.cols
        return inside_grid and position not in self.walls

    def deterministic_move(self, position, action):
        row, col = position
        dr, dc = MOVES[action]
        candidate = (row + dr, col + dc)
        return candidate if self.is_valid_position(candidate) else position

    def action_outcomes(self, chosen_action):
        """(actual_action, probability) triples for a chosen action:
        0.80 the intended direction, 0.10 / 0.10 the two perpendicular
        slips. Used both for live sampling and for tensor construction."""
        perp = PERPENDICULAR_ACTIONS[chosen_action]
        return [(chosen_action, 0.80), (perp[0], 0.10), (perp[1], 0.10)]

    def sample_actual_action(self, chosen_action):
        outcomes = self.action_outcomes(chosen_action)
        actions = [a for a, _ in outcomes]
        probs = [p for _, p in outcomes]
        return self.rng.choices(actions, weights=probs, k=1)[0]

    # --------------------------------------------------------
    # PURE transition function — the heart of the engine/GUI split.
    # --------------------------------------------------------

    def simulate_transition(self, state, actual_action):
        """
        Given an arbitrary state and an ALREADY-SAMPLED actual action,
        return (next_state, reward). Does not read or write
        self.position / self.has_treasure — this must work for any
        state in the state space, not just the environment's current
        one, since mdp_builder.py calls it once per (state, action,
        actual_action) triple to build P and R.
        """
        position, has_treasure = state
        next_position = self.deterministic_move(position, actual_action)
 
        self.reward = 0
        self.reward += -1
        next_has_treasure = has_treasure
 
        if next_position == self.treasure and not has_treasure:
            next_has_treasure = True
            self.reward += 15
 
        if next_position in self.danger:
            self.reward -= 25
 
        if next_position == self.goal:
            self.reward += 40 if next_has_treasure else -10
 
        return (next_position, next_has_treasure), self.reward

    # --------------------------------------------------------
    # Live-play transition (stochastic, mutates self)
    # --------------------------------------------------------

    def step(self, chosen_action):
        """
        Execute one stochastic transition from the environment's CURRENT
        state. Samples the actual action, then delegates the actual
        reward/next-state computation to simulate_transition() so the
        game and the MDP tensors can never disagree about the dynamics.
        """
        if self.done:
            return self.get_state(), 0, True, {"chosen_action": None, "actual_action": None}

        current_state = self.get_state()
        actual_action = self.sample_actual_action(chosen_action)
        next_state, reward = self.simulate_transition(current_state, actual_action)

        self.position, self.has_treasure = next_state

        event_messages = []
        if self.has_treasure and current_state[1] is False and next_state[1] is True:
            event_messages.append("Treasure collected! +15")
        if self.position in self.danger:
            event_messages.append("Danger! -25")
        if self.position == self.goal:
            self.done = True
            event_messages.append(
                "You escaped with the treasure! +40" if self.has_treasure
                else "You exited without the treasure. -10"
            )

        self.steps += 1
        self.total_reward += reward
        self.last_chosen_action = chosen_action
        self.last_actual_action = actual_action
        self.last_reward = reward
        self.message = " | ".join(event_messages) if event_messages else f"Move reward: {reward}"

        info = {"chosen_action": chosen_action, "actual_action": actual_action}
        return self.get_state(), reward, self.done, info
