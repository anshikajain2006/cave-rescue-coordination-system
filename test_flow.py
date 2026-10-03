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

print("\nAll systems checked. Ready for demo.")
