from typing import List, Sequence, Tuple

from cave_environment import CaveEnvironment

EMPTY_BOT_DIAMETER_M = 0.25
LOADED_BOT_DIAMETER_M = 0.6    # Diameter when carrying any payload
MAX_SAFE_WATER_LEVEL = 0.8
MOVES_PER_MINUTE = 10          # 0.1 minutes of battery per grid move
DEFAULT_BATTERY_MIN = 30.0


class SafetyVerifier:
    """Deterministic pre-mission safety checks against the cave map. Pure math, no LLM."""

    def __init__(self, cave_env: CaveEnvironment, battery_remaining_min: float = DEFAULT_BATTERY_MIN):
        self.cave_env = cave_env
        self.battery_remaining_min = battery_remaining_min  # Update between missions as the battery drains

    def check_mission(self, bot_payload_kg: float, target_cell: Tuple[int, int],
                      path: Sequence[Tuple[int, int]]) -> Tuple[bool, List[str]]:
        """Returns (is_safe, violations); the mission is safe only if no rule is violated.

        path is the full cell sequence including the start, as returned by the A* planner.
        """
        if bot_payload_kg < 0:
            raise ValueError(f"bot_payload_kg must be >= 0, got {bot_payload_kg}")

        violations: List[str] = []
        cells: List[Tuple[int, int]] = [(cell[0], cell[1]) for cell in path]  # Also accepts lists, e.g. from JSON
        cave_map = self.cave_env.cave_map

        # Rule 4: a path must exist
        if not cells:
            violations.append("NO PATH FOUND")

        # Path integrity: every cell must be open cave and each move a single grid step
        for i, (x, y) in enumerate(cells):
            if (x, y) not in cave_map:
                violations.append(f"INVALID CELL at ({x},{y}): rock or out of bounds")
            if i > 0:
                px, py = cells[i - 1]
                if abs(x - px) + abs(y - py) != 1:
                    violations.append(f"PATH DISCONTINUITY between ({px},{py}) and ({x},{y})")

        # Rule 1: every passage must be strictly wider than the bot
        bot_diameter = LOADED_BOT_DIAMETER_M if bot_payload_kg > 0 else EMPTY_BOT_DIAMETER_M
        reported = set()
        for cell in cells:
            if cell in cave_map and cell not in reported and cave_map[cell].passage_width <= bot_diameter:
                reported.add(cell)
                violations.append(f"PASSAGE TOO NARROW at ({cell[0]},{cell[1]})")

        # Rule 2: target must be below the flooding threshold
        target = (target_cell[0], target_cell[1])
        if target not in cave_map:
            violations.append(f"INVALID CELL at ({target[0]},{target[1]}): target is rock or out of bounds")
        elif cave_map[target].water_level >= MAX_SAFE_WATER_LEVEL:
            violations.append(f"FLOODING RISK: water level {cave_map[target].water_level}")

        # Rule 3: travel time must be strictly less than remaining battery
        moves = max(len(cells) - 1, 0)
        if moves / MOVES_PER_MINUTE >= self.battery_remaining_min:
            violations.append("BATTERY INSUFFICIENT")

        return (not violations, violations)
