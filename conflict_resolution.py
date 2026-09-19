"""
conflict_resolution.py: cell-bidding protocol for two rescue bots.

If two routes share any cell, each bot bids
    bid = priority_weight x battery_remaining / path_length   (path_length in moves)
The higher bid keeps its route. The loser replans with A* treating the contested cells as rock, repeating
if the detour touches other cells of the winner's route. If the loser has no detour (for example a
contested cell is its own start or goal), its final route is [] and it holds.
"""
import copy
from typing import List, Optional, Sequence, Tuple

from cave_environment import CaveEnvironment
from planner import ForkliftPlanner

Cell = Tuple[int, int]

# The spec defines high=2 and normal=1; the parser's "medium" is normal. Low is not in the spec: 0.5 here.
PRIORITY_WEIGHTS = {'high': 2.0, 'medium': 1.0, 'normal': 1.0, 'low': 0.5}

_default_env: Optional[CaveEnvironment] = None


def compute_bid(priority: str, battery: float, route: Sequence[Cell]) -> float:
    moves = max(len(route) - 1, 1)  # A zero-move route still bids as if it needs one move
    return PRIORITY_WEIGHTS.get(priority, 1.0) * battery / moves


def find_conflicts(route_a: Sequence[Cell], route_b: Sequence[Cell]) -> List[Cell]:
    """Cells on both routes, in route_a order"""
    other = {(c[0], c[1]) for c in route_b}
    return [(c[0], c[1]) for c in route_a if (c[0], c[1]) in other]


def replan_avoiding(env: CaveEnvironment, start: Cell, goal: Cell, avoid: Sequence[Cell]) -> List[Cell]:
    """A* from start to goal with the avoid cells treated as rock; [] if impossible"""
    if start in avoid or goal in avoid:
        return []
    blocked_env = copy.copy(env)
    blocked_env.grid = [row[:] for row in env.grid]
    for x, y in avoid:
        blocked_env.grid[y][x] = 1
    path, _, _ = ForkliftPlanner(blocked_env).a_star(start, goal)
    return [(c[0], c[1]) for c in path]


def resolve_conflicts(bot1_route, bot2_route, bot1_priority, bot1_battery, bot2_priority, bot2_battery,
                      env: Optional[CaveEnvironment] = None, bot1_name: str = 'UNIT-01', bot2_name: str = 'UNIT-02'):
    """Returns (final_route_1, final_route_2, resolution_log). Ties go to bot 1. A final route of [] means hold."""
    global _default_env
    if env is None:
        _default_env = _default_env or CaveEnvironment()
        env = _default_env
    route1: List[Cell] = [(c[0], c[1]) for c in bot1_route]
    route2: List[Cell] = [(c[0], c[1]) for c in bot2_route]
    log: List[str] = []

    contested = find_conflicts(route1, route2)
    if not route1 or not route2 or not contested:
        log.append("No shared cells; both routes stand.")
        return route1, route2, log

    bid1 = compute_bid(bot1_priority, bot1_battery, route1)
    bid2 = compute_bid(bot2_priority, bot2_battery, route2)
    log.append(f"Conflict on {len(contested)} cell(s): {', '.join(map(str, contested))}")
    log.append(f"{bot1_name} bid = {PRIORITY_WEIGHTS.get(bot1_priority, 1.0):g} x {bot1_battery:.1f} / "
               f"{max(len(route1) - 1, 1)} = {bid1:.3f}")
    log.append(f"{bot2_name} bid = {PRIORITY_WEIGHTS.get(bot2_priority, 1.0):g} x {bot2_battery:.1f} / "
               f"{max(len(route2) - 1, 1)} = {bid2:.3f}")

    bot1_wins = bid1 >= bid2
    winner_name, loser_name = (bot1_name, bot2_name) if bot1_wins else (bot2_name, bot1_name)
    winner_route, loser_route = (route1, route2) if bot1_wins else (route2, route1)
    log.append(f"{winner_name} wins the contested cells{' (tie goes to ' + bot1_name + ')' if bid1 == bid2 else ''}; "
               f"{loser_name} replans")

    avoid = list(contested)
    detour: List[Cell] = []
    # Each round that finds new conflicts avoids at least one more winner cell, so this always terminates
    for _ in range(len(winner_route) + 1):
        detour = replan_avoiding(env, loser_route[0], loser_route[-1], avoid)
        new_conflicts = [c for c in find_conflicts(detour, winner_route) if c not in avoid]
        if not detour or not new_conflicts:
            break
        avoid.extend(new_conflicts)  # Detour crossed the winner's route elsewhere: avoid those cells too
    else:
        detour = []

    if detour:
        log.append(f"{loser_name} detour: {len(detour) - 1} moves (was {len(loser_route) - 1}), "
                   f"avoiding {len(avoid)} cell(s)")
    else:
        log.append(f"{loser_name} has no detour avoiding the contested cells; it holds")
    return (winner_route, detour, log) if bot1_wins else (detour, winner_route, log)
