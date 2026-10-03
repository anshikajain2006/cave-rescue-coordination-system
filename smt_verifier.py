"""Z3 SMT verification of a planned route: every safety rule is a constraint, proved jointly over the whole path"""
from typing import Any, List, Sequence, Tuple

from z3 import BoolVal, If, IntVal, Q, RealVal, Solver, ToReal, sat, unsat

from cave_environment import CaveEnvironment
from models import CaveCommand
from safety_verifier import (DEFAULT_BATTERY_MIN, EMPTY_BOT_DIAMETER_M, LOADED_BOT_DIAMETER_M,
                             MAX_SAFE_WATER_LEVEL, MOVES_PER_MINUTE)

WATER_LIMIT = MAX_SAFE_WATER_LEVEL  # Cells with water_level >= this are unsafe
TRIVIAL_ACTIONS = ("scan", "status", "hold")  # The bot does not move, so there is no route to prove


def _real(value: float):
    """Exact Z3 rational for a Python float (e.g. 0.8 -> 4/5), so comparisons have no rounding error"""
    return RealVal(str(value))


def _abs(expr):
    return If(expr >= 0, expr, -expr)


def verify_path_smt(path: Sequence[Tuple[int, int]], env: CaveEnvironment, payload_kg: float = 0.0,
                    battery_remaining_min: float = DEFAULT_BATTERY_MIN) -> dict:
    """
    Use Z3 to formally verify a planned path satisfies all safety constraints.

    path is the full (x, y) cell sequence including the start, as returned by the A* planner.
    Rules (the same ones SafetyVerifier.check_mission applies, but water is enforced on every cell):
      - a path exists and every cell is open cave (not rock, not off the map)
      - each move is a single grid step
      - every passage is strictly wider than the bot (0.25 m empty, 0.6 m with any payload)
      - every cell's water level is below WATER_LIMIT
      - travel time (moves / MOVES_PER_MINUTE) is strictly less than the remaining battery
    Returns {"safe", "proof": "VERIFIED" | "VIOLATED", "violations", "checked_cells", "solver_result"}
    """
    if payload_kg < 0:
        raise ValueError(f"payload_kg must be >= 0, got {payload_kg}")
    cells: List[Tuple[int, int]] = [(int(cell[0]), int(cell[1])) for cell in path]
    cave_map = env.cave_map
    bot_diameter = _real(LOADED_BOT_DIAMETER_M if payload_kg > 0 else EMPTY_BOT_DIAMETER_M)
    water_limit = _real(WATER_LIMIT)

    # (label, constraint) pairs; the label is the violation reported if the constraint is false
    rules: List[Tuple[str, Any]] = []  # Any: z3's stubs type comparisons loosely
    if not cells:
        rules.append(("NO PATH FOUND", BoolVal(False)))

    for i, (x, y) in enumerate(cells):
        cell = cave_map.get((x, y))
        if cell is None:
            rules.append((f"INVALID CELL at ({x},{y}): rock or out of bounds", BoolVal(False)))
        else:
            rules.append((f"PASSAGE TOO NARROW at ({x},{y})", _real(cell.passage_width) > bot_diameter))
            rules.append((f"FLOODING RISK at ({x},{y}): water level {cell.water_level}",
                          _real(cell.water_level) < water_limit))
        if i > 0:
            px, py = cells[i - 1]
            step = _abs(IntVal(x) - IntVal(px)) + _abs(IntVal(y) - IntVal(py))
            rules.append((f"PATH DISCONTINUITY between ({px},{py}) and ({x},{y})", step == 1))

    moves = IntVal(max(len(cells) - 1, 0))
    rules.append(("BATTERY INSUFFICIENT", ToReal(moves) / Q(MOVES_PER_MINUTE, 1) < _real(battery_remaining_min)))

    # Prove the conjunction of every rule; each is tracked so a failure is attributable
    solver = Solver()
    for n, (_, constraint) in enumerate(rules):
        solver.assert_and_track(constraint, f"rule_{n}")
    result = solver.check()
    safe = result == sat

    # The unsat core may be a subset, so name every rule that cannot hold on its own
    violations = [] if safe else [label for label, constraint in rules if Solver().check(constraint) == unsat]
    if not safe and not violations:  # Only the combination of rules fails (or Z3 returned unknown)
        violations = [f"SMT CHECK FAILED: solver returned {result}"]
    reported, unique = set(), []
    for label in violations:  # A cell revisited on the route is only reported once per rule
        if label not in reported:
            reported.add(label)
            unique.append(label)

    return {
        "safe": safe,
        "proof": "VERIFIED" if safe else "VIOLATED",
        "violations": unique,
        "checked_cells": len(cells),
        "solver_result": str(result),
    }


def verify_command_smt(command: CaveCommand, path: Sequence[Tuple[int, int]], env: CaveEnvironment,
                       payload_kg: float = 0.0, battery_remaining_min: float = DEFAULT_BATTERY_MIN) -> dict:
    """Entry point: verify a full CaveCommand + planned path. Returns same dict as verify_path_smt."""
    if command.action in TRIVIAL_ACTIONS:
        return {"safe": True, "proof": "VERIFIED", "violations": [], "checked_cells": 0, "solver_result": "trivial"}
    return verify_path_smt(path, env, payload_kg, battery_remaining_min)
