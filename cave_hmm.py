import random
from typing import Tuple

import numpy as np

from cave_environment import CaveEnvironment
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

    def step(self, action: int, observation: int) -> np.ndarray:
        """Runs one predict/update cycle and returns the full belief array over HMM states"""
        return self.estimator.bayesian_filter_step(action, observation)

    def most_likely_position(self) -> Tuple[int, int]:
        """Cave (x, y) of the highest-belief state"""
        r, c = self.hmm_env.state_to_coord(self.estimator.get_most_likely_state())
        return (c - HMM_BORDER, r - HMM_BORDER)


if __name__ == '__main__':
    rng = random.Random(0)
    localiser = CaveHMMLocaliser(CaveEnvironment())
    action_names = ['N', 'S', 'E', 'W']

    print(grid_to_hmm_string(localiser.cave_env, pad_border=False))
    for t in range(1, 6):
        action, observation = rng.randrange(4), rng.randrange(5)
        belief = localiser.step(action, observation)
        x, y = localiser.most_likely_position()
        print(f"step {t}: action={action_names[action]} walls_sensed={observation} "
              f"-> most likely ({x}, {y}) p={belief.max():.3f} sum={belief.sum():.3f}")
