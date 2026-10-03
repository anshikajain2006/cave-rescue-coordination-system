import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Cave environment loads
from cave_environment import CaveEnvironment
cave = CaveEnvironment()
print("PASS CaveEnvironment loaded")
print(f"  Free cells: {len(cave.cave_map)}")

# Test 2: A* finds paths to all survivors
from planner import ForkliftPlanner
from models import Node
planner = ForkliftPlanner(cave)
survivors = [(3,14), (12,5), (17,2)]
last_path = None
for sx, sy in survivors:
    path, expanded, t = planner.a_star((0,0), (sx,sy), "manhattan")
    status = "PASS" if path else "FAIL"
    print(f"{status} A* to ({sx},{sy}): {len(path)-1 if path else 'NO PATH'} moves, {expanded} nodes expanded")
    if path:
        last_path = (path, (sx, sy))

# Test 3: HMM loads
from cave_hmm import CaveHMMLocaliser
hmm = CaveHMMLocaliser(cave)
hmm.step(2, observation=2)
hmm.step(1, observation=1)
pos = hmm.most_likely_position()
print(f"PASS HMM localiser: belief updated, top position={pos}")

# Test 4: Safety verifier
from safety_verifier import SafetyVerifier
sv = SafetyVerifier(cave)
if last_path:
    path_coords, target = last_path
    safe, violations = sv.check_mission(0.0, target, path_coords)
    print(f"{'PASS' if safe else 'FAIL'} Safety check for {target}: {'OK' if safe else violations}")
else:
    print("SKIP Safety check (no path found)")

# Test 5: LLM parser fallback (offline)
from llm_parser import parse_command
result = parse_command("send bot to chamber 3", api_key=None)
print(f"PASS LLM fallback: action={result.action}, target={result.target_chamber}")

# Test 6: Z3 SMT verifier proves the A* route to (17,2)
from smt_verifier import verify_path_smt
path_17_2, _, _ = planner.a_star((0,0), (17,2), "manhattan")
smt_result = verify_path_smt(path_17_2, cave)
assert smt_result["proof"] == "VERIFIED", smt_result
print(f"PASS SMT verifier: {smt_result['checked_cells']} cells VERIFIED")

# Test 7: ReACT parser falls back to the keyword parser without an API key
from react_parser import react_parse
react_cmd, trace = react_parse("rush oxygen kit to chamber 5", api_key=None)
assert react_cmd.action == "deliver" and react_cmd.target_chamber == 5, react_cmd
assert trace[0]["type"] == "FALLBACK", trace
print(f"PASS ReACT parser (no-key fallback): action={react_cmd.action}, target={react_cmd.target_chamber}")

# Test 8: Autonomous mission building blocks — bandit updates and a Q-learning route
import numpy as np
from autonomous_mission import QLearningNavigator, ThompsonBandit
bandit = ThompsonBandit(range(1, 10), np.random.default_rng(0))
bandit.update(3, 1)
bandit.update(5, 0)
assert bandit.mean(3) > 0.5 > bandit.mean(5), bandit.table()
nav = QLearningNavigator(cave, seed=0)
q_route = nav.route((0,0), (17,2))
assert q_route and q_route[0] == (0,0) and q_route[-1] == (17,2), q_route
assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(q_route, q_route[1:])), "route not contiguous"
print(f"PASS Autonomous mission: bandit reward update OK, Q-learning route to (17,2) in {len(q_route)-1} moves")

# Test 9: HMM-coupled bandit — uncertainty favours nearby chambers and weakens reward updates
from autonomous_mission import HMMCoupling
from llm_parser import CHAMBER_MAP
coupled_loc = CaveHMMLocaliser(cave)
coupling = HMMCoupling(lambda: coupled_loc, nav, CHAMBER_MAP)
n_states = len(coupled_loc.belief)
state_cell = {s: (c - 1, r - 1) for s in range(n_states) for r, c in [coupled_loc.hmm_env.state_to_coord(s)]}
entrance = next(s for s, cell in state_cell.items() if cell == (0, 0))
spread = np.zeros(n_states)
spread[[s for s, cell in state_cell.items() if cell in cave.cave_map]] = 1.0
spread[entrance] = 246.0  # Half the belief at the entrance, half spread over the other open cells
coupled_loc.estimator.belief_state = spread / spread.sum()
weights = coupling.reach_weights([1, 3])
assert weights[1] > weights[3], weights  # C1 (9 moves from the entrance) beats C3 (21 moves) when unsure
coupled_bandit = ThompsonBandit(range(1, 10), np.random.default_rng(0), hmm=coupling)
coupled_bandit.update(3, 0)
assert 0 < coupled_bandit.last_confidence < 1 and coupled_bandit.beta[3] < 2, coupled_bandit.table()
certain = np.zeros(n_states)
certain[entrance] = 1.0
coupled_loc.estimator.belief_state = certain
assert coupling.position_confidence() == 1.0 and set(coupling.reach_weights([1, 3]).values()) == {1.0}
print(f"PASS HMM-coupled bandit: reach weight C1 {weights[1]:.2f} > C3 {weights[3]:.2f}, "
      f"uncertain update weight {coupled_bandit.last_confidence:.2f}")

# Test 10: HMM coupling steers the bandit — uncertain belief picks nearby chambers more often
spread[entrance] = 40.0  # ~14% of the belief at the entrance: poorly localised
coupled_loc.estimator.belief_state = spread / spread.sum()
uncertain_bandit = ThompsonBandit(range(1, 10), np.random.default_rng(1), hmm=coupling)
picks = [uncertain_bandit.select() for _ in range(1500)]
near, far = picks.count(1), picks.count(3)  # C1 is 9 moves from the entrance, C3 is 21
assert near > 2 * far, (near, far)
print(f"PASS HMM-coupled selection: uncertain bot picked nearby C1 {near}x vs distant C3 {far}x in 1500 draws")

# Tests 11-14: Autonomous mission loop exit conditions (headless engine, fixed seeds)
from mission_engine import (AutoConfig, EXIT_ALL_FOUND, EXIT_BATTERY_RESERVE, EXIT_CYCLE_CAP, EXIT_EARLY_STOP,
                            MissionSim, new_store)


def run_mission(config, use_smt=True, battery=None, seed=0, prepare=None):
    store = new_store(cave, use_smt=use_smt)
    sim = MissionSim(cave, store)
    if battery is not None:
        store['bots']['UNIT-01'].battery = battery
    if prepare:
        prepare(store)
    result = sim.run_autonomous_mission(rng=np.random.default_rng(seed), config=config, navigator=sim.navigator(seed=0))
    return store, result


# SMT off: chamber 2 (S1) is only reachable through flooded cells, so all three survivors need the rule-based check
_, result = run_mission(AutoConfig(arms=(2, 3, 4), detection_prob=1.0), use_smt=False)
assert result.exit_code == EXIT_ALL_FOUND and len(result.found) == 3, result.summary
print(f"PASS Autonomous exit — all survivors found: {len(result.found)}/3 in {result.cycles} cycles")

_, result = run_mission(AutoConfig(arms=(1, 6), exhausted_mean=0.0), battery=2.5)
assert result.exit_code == EXIT_BATTERY_RESERVE and result.battery_reserve_fired, result.summary
assert result.position == (0, 0) and result.returned_home and result.battery > 0, result.summary
print(f"PASS Autonomous exit — battery reserve: back at base with {result.battery:.1f} min after {result.cycles} cycles")

_, result = run_mission(AutoConfig(arms=(1,), exhausted_mean=0.45))
assert result.exit_code == EXIT_EARLY_STOP and result.early_stop_fired, result.summary
assert result.position == (0, 0) and result.returned_home, result.summary
print(f"PASS Autonomous exit — early stop: search exhausted after {result.cycles} cycle(s), returned to base")

_, result = run_mission(AutoConfig(arms=(1, 6), max_cycles=1, exhausted_mean=0.0))
assert result.exit_code == EXIT_CYCLE_CAP and result.position == (0, 0) and result.returned_home, result.summary
print("PASS Autonomous exit — cycle cap: returned to base")

# Test 15: Conflict resolution — UNIT-02 loses the bid and replans with A* around UNIT-01's route
def crossing_route(store):
    unit2 = store['bots']['UNIT-02']
    route, _, _ = planner.a_star(unit2.position, (16, 1))  # Crosses UNIT-01's route to chamber 4 at (16,3)
    unit2.active_route = [(c[0], c[1]) for c in route]
    unit2.position = unit2.active_route[-1]
    unit2.priority = 'low'


store, result = run_mission(AutoConfig(arms=(4,), detection_prob=1.0), prepare=crossing_route)
unit1_route, unit2 = store['bots']['UNIT-01'].active_route, store['bots']['UNIT-02']
reroute = store.get('last_reroute')
assert reroute and reroute['bot_id'] == 'UNIT-02' and reroute['new'] == unit2.active_route, reroute
assert unit2.active_route != reroute['old'] and set(reroute['old']) & set(unit1_route), "no conflict was staged"
assert not set(unit2.active_route) & set(unit1_route), "detour still shares cells with UNIT-01"
assert unit2.active_route[0] == reroute['old'][0] and unit2.active_route[-1] == reroute['old'][-1]
assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(unit2.active_route, unit2.active_route[1:]))
assert any('CONFLICT REROUTE: UNIT-02' in line for line in store['mission_log'])
print(f"PASS Conflict reroute: UNIT-02 detoured {len(reroute['old']) - 1} -> {len(unit2.active_route) - 1} moves, "
      f"no cells shared with UNIT-01")

# Test 16: Loaded bot blocked by narrow passages drives the MDP pilot's detour (full app, headless)
os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile
import cave_db
cave_db.MissionDatabase.__init__.__defaults__ = (os.path.join(tempfile.mkdtemp(), 'test_missions.db'),)
from streamlit.testing.v1 import AppTest
app = AppTest.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py'), default_timeout=300).run()
assert not app.exception, app.exception
next(t for t in app.text_input if t.label == "Rescue command").input("oxygen kit to chamber 4")
next(b for b in app.button if b.label == "Submit command").click()
app.run()
assert not app.exception, app.exception
kind, message = app.session_state.last_result
unit1 = app.session_state.bots['UNIT-01']
mission_log = app.session_state.mission_log
assert any('PASSAGE TOO NARROW' in line for line in mission_log), "A* route was not blocked by narrow passages"
assert kind == 'success' and unit1.position == (17, 2), (kind, message)
assert any('REROUTED [UNIT-01]' in line for line in mission_log) and 'MDP pilot detour' in message, message
print(f"PASS Narrow-passage reroute: oxygen kit delivered to (17,2) on the MDP detour "
      f"({len(unit1.active_route) - 1} moves)")

print("\nAll systems checked. Ready for demo.")
