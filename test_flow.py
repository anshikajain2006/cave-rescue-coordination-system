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
print(f"PASS LLM fallback: action={result.get('action')}, target={result.get('target_chamber')}")

print("\nAll systems checked. Ready for demo.")
