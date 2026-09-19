import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from environment import SurgicalLabEnvironment

# =====================================================================
# CAVE GEOMETRY
# =====================================================================
CAVE_WIDTH, CAVE_HEIGHT = 20, 15

# Rock walls -- same layout as CUSTOM_OBSTACLES in config.py, renamed for the cave scenario
CAVE_WALLS = [
    # Western rock face separating the entrance chamber from the central tunnels
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
    # Collapsed ceiling ledge
    (7, 9), (8, 9), (9, 9), (10, 9), (11, 9),
    # Winding rock formation around the deep pocket chamber
    (12, 10), (13, 10), (13, 9), (13, 8), (13, 7), (14, 7), (14, 6), (14, 5),
    (13, 5), (13, 4), (13, 3), (12, 3), (11, 3), (11, 4), (10, 4), (10, 5),
    (9, 5), (9, 6), (9, 4), (9, 3), (10, 3), (10, 2), (10, 1), (11, 1),
    (11, 2), (12, 2), (13, 2),
    # Lower sump overhang
    (1, 12), (2, 12), (3, 12), (6, 13), (7, 13), (4, 12), (5, 12), (6, 12),
    # Entrance stalagmite column
    (1, 1), (1, 2), (1, 3),
]

# Named chambers as inclusive (x_min, y_min, x_max, y_max) bounding boxes
CHAMBERS: Dict[str, Tuple[int, int, int, int]] = {
    'entrance_chamber': (0, 0, 5, 11),
    'lower_sump': (0, 13, 5, 14),
    'deep_pocket': (11, 4, 13, 8),
    'east_gallery': (15, 0, 19, 14),
}

# Hardcoded survivor locations (all free cells reachable from the entrance)
SURVIVOR_LOCATIONS: List[Tuple[int, int]] = [
    (3, 14),   # lower_sump, beneath the overhang
    (12, 5),   # deep_pocket, inside the winding formation
    (17, 2),   # east_gallery, upper end
]

MIN_PASSAGE_WIDTH, MAX_PASSAGE_WIDTH = 0.3, 1.0  # metres


@dataclass
class CaveCell:
    """Attributes of a single traversable cave cell"""
    passage_width: float         # metres, 0.3-1.0
    water_level: float           # 0.0 (dry) to 1.0 (fully flooded)
    is_survivor_location: bool

    @property
    def is_tunnel(self) -> bool:
        """Narrow passages are tunnels; wider open areas are chambers"""
        return self.passage_width < 0.6


class CaveEnvironment(SurgicalLabEnvironment):
    """Cave rescue grid built on SurgicalLabEnvironment.

    Walls live in the inherited `grid`, so is_valid, get_neighbors and
    manhattan_distance behave exactly as in the parent and the A* planner
    works unchanged. Per-cell cave data is kept separately in `cave_map`.
    """

    def __init__(self, seed: int = 42):
        # 'random' at zero density yields an empty 20x15 grid without depending on config.py
        super().__init__(map_type='random', width=CAVE_WIDTH, height=CAVE_HEIGHT, obstacle_density=0.0)
        self.map_type = 'cave'
        self.survivor_locations = list(SURVIVOR_LOCATIONS)
        self._load_cave_walls()
        self.cave_map: Dict[Tuple[int, int], CaveCell] = self._build_cave_map(random.Random(seed))

    def _load_cave_walls(self):
        for x, y in CAVE_WALLS:
            if self.is_valid_coord(x, y):
                self.grid[y][x] = 1
        for x, y in self.survivor_locations:
            if not self.is_valid(x, y):
                raise ValueError(f"Survivor location ({x}, {y}) is out of bounds or inside rock")

    def _build_cave_map(self, rng: random.Random) -> Dict[Tuple[int, int], CaveCell]:
        """Assigns passage width, water level and survivor flag to every free cell"""
        survivors = set(self.survivor_locations)
        cave_map = {}
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1:
                    continue
                # Fewer open sides -> tighter squeeze
                open_sides = len(self.get_neighbors(x, y))
                width = MIN_PASSAGE_WIDTH + (MAX_PASSAGE_WIDTH - MIN_PASSAGE_WIDTH) * open_sides / 4
                width += rng.uniform(-0.1, 0.1)
                # Water pools toward the lower (higher y) levels of the cave
                water = 0.7 * y / (self.height - 1) + rng.uniform(0.0, 0.3)
                cave_map[(x, y)] = CaveCell(
                    passage_width=round(min(MAX_PASSAGE_WIDTH, max(MIN_PASSAGE_WIDTH, width)), 2),
                    water_level=round(min(1.0, max(0.0, water)), 2),
                    is_survivor_location=(x, y) in survivors,
                )
        return cave_map

    def get_cell(self, x: int, y: int) -> CaveCell:
        """Returns cave data for a free cell; raises KeyError for rock or out-of-bounds"""
        return self.cave_map[(x, y)]

    def chamber_at(self, x: int, y: int) -> str:
        """Name of the chamber containing (x, y), or 'tunnel' if it lies in no named chamber"""
        for name, (x0, y0, x1, y1) in CHAMBERS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return 'tunnel'


# =====================================================================
# RESCUE FLEET
# =====================================================================
DEFAULT_BATTERY_MIN = 30.0
FLEET_START_POSITIONS: Dict[str, Tuple[int, int]] = {
    'UNIT-01': (0, 0),     # C0 Entrance
    'UNIT-02': (17, 12),   # C5 South Gallery
}


@dataclass
class RescueBot:
    """Mutable state of one rescue bot"""
    bot_id: str
    position: Tuple[int, int]
    battery: float = DEFAULT_BATTERY_MIN    # Minutes of drive time left
    payload: Optional[str] = None           # Item carried on the active route, if any
    priority: str = 'medium'                # Priority of the active route, used when bidding for contested cells
    # Most recent route (start ... current position). Missions complete instantly in this simulation, so the
    # route is treated as still in progress, and reserved, until this bot's next mission.
    active_route: List[Tuple[int, int]] = field(default_factory=list)


def create_fleet(battery: float = DEFAULT_BATTERY_MIN) -> Dict[str, RescueBot]:
    return {bot_id: RescueBot(bot_id, position, battery) for bot_id, position in FLEET_START_POSITIONS.items()}
