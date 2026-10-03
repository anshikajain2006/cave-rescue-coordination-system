import random
from typing import Tuple

import numpy as np

from cave_environment import CaveEnvironment, get_wall_observation
from hmm_environment import WarehouseHMMEnvironment
from hmm_filter import HMMStateEstimator

CAVE_SENSOR_ACCURACY = 0.65  # Sonar is less reliable underground than in the warehouse

# WarehouseHMMEnvironment strips whitespace from each map line, which would delete free
# cells on the cave edges. Wrapping the map in a ring of '#' keeps every row intact.
# Off-grid already counts as a wall in both models, so wall counts and moves are unchanged;
# HMM coordinates are simply shifted by this offset.
HMM_BORDER = 1


def grid_to_hmm_string(cave_env: CaveEnvironment, pad_border: bool = True) -> str:
    """Converts the cave grid to WarehouseHMMEnvironment's map format: '#' wall, ' ' free, one row per line.

    With pad_border=True (required for WarehouseHMMEnvironment) the map is surrounded by a wall ring.
    """
    rows = [''.join('#' if cell == 1 else ' ' for cell in row) for row in cave_env.grid]
    if pad_border:
        wall_row = '#' * (cave_env.width + 2)
        rows = [wall_row] + ['#' + row + '#' for row in rows] + [wall_row]
    return '\n'.join(rows)


class CaveHMMLocaliser:
    """Tracks a rescue robot's position in the cave from noisy wall-count sonar readings.

    Actions: 0=North (y-1), 1=South (y+1), 2=East (x+1), 3=West (x-1).
    Observations: number of adjacent walls, 0-4.
    """

    def __init__(self, cave_env: CaveEnvironment, sensor_accuracy: float = CAVE_SENSOR_ACCURACY):
        self.cave_env = cave_env
        self.hmm_env = WarehouseHMMEnvironment(grid_to_hmm_string(cave_env), sensor_accuracy=sensor_accuracy)
        self.estimator = HMMStateEstimator(self.hmm_env.num_states, self.hmm_env.T, self.hmm_env.E)

    @property
    def belief(self) -> np.ndarray:
        """Current belief over HMM states (1D, sums to 1)"""
        return self.estimator.belief_state

    def step(self, action: int, observation: int) -> np.ndarray:
        """Runs one predict/update cycle and returns the full belief array over HMM states"""
        return self.estimator.bayesian_filter_step(action, observation)

    def sense(self, observation: int) -> np.ndarray:
        """Measurement update only, for a sonar reading taken without moving; returns the new belief"""
        self.estimator.belief_state = self.estimator.update(observation, self.estimator.belief_state)
        return self.estimator.belief_state

    def belief_entropy(self) -> float:
        """Shannon entropy of current belief distribution (nats). Higher = more uncertain."""
        b = self.belief
        b = b[b > 0]  # Avoid log(0)
        return float(-np.sum(b * np.log(b))) + 0.0  # + 0.0 turns -0.0 (a certain belief) into 0.0

    def most_likely_position(self) -> Tuple[int, int]:
        """Cave (x, y) of the highest-belief state"""
        r, c = self.hmm_env.state_to_coord(self.estimator.get_most_likely_state())
        return (c - HMM_BORDER, r - HMM_BORDER)


if __name__ == '__main__':
    rng = random.Random(0)
    env = CaveEnvironment()
    localiser = CaveHMMLocaliser(env)
    action_names = ['N', 'S', 'E', 'W']
    action_steps = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    true_pos = (0, 0)

    print(grid_to_hmm_string(env, pad_border=False))
    for t in range(1, 6):
        action = rng.randrange(4)
        dx, dy = action_steps[action]
        if env.is_valid(true_pos[0] + dx, true_pos[1] + dy):  # Bumping into rock leaves the bot in place
            true_pos = (true_pos[0] + dx, true_pos[1] + dy)
        observation = get_wall_observation(env, true_pos)
        belief = localiser.step(action, observation)
        x, y = localiser.most_likely_position()
        print(f"step {t}: action={action_names[action]} true={true_pos} walls_sensed={observation} "
              f"-> most likely ({x}, {y}) p={belief.max():.3f} entropy={localiser.belief_entropy():.3f}")
