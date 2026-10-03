"""
eval.py: runs many headless autonomous missions with different random seeds and summarises them.

    python eval.py                       # 100 missions, SMT verifier on (the app's default)
    python eval.py --smt off             # rule-based verifier: flooded chambers become reachable
    python eval.py --missions 20 --out quick.csv

Each mission starts from a fresh session (UNIT-01 at the entrance, 30 min battery, certain belief). The seed drives
the Thompson-sampling bandit and the 85% sonar detection draws. The Q-learning navigator is trained once and
shared by all missions (pass --fresh-navigator to retrain it per mission with that mission's seed; slower).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cave_environment import CaveEnvironment  # noqa: E402
from mission_engine import AutoConfig, MissionSim, new_store  # noqa: E402


def run_missions(missions: int, use_smt: bool, fresh_navigator: bool, first_seed: int = 0) -> pd.DataFrame:
    cave = CaveEnvironment()
    shared_navigator = None if fresh_navigator else MissionSim(cave, new_store(cave, use_smt)).navigator(seed=0)
    rows = []
    for i, seed in enumerate(range(first_seed, first_seed + missions), 1):
        store = new_store(cave, use_smt=use_smt)
        sim = MissionSim(cave, store)
        navigator = sim.navigator(seed=seed) if fresh_navigator else shared_navigator
        started = time.perf_counter()
        result = sim.run_autonomous_mission(rng=np.random.default_rng(seed), config=AutoConfig(), navigator=navigator)
        rows.append({
            'seed': seed,
            'survivors_found': len(result.found),
            'survivors_total': result.survivors_total,
            'cycles_used': result.cycles,
            'exit_reason': result.end_reason,
            'exit_code': result.exit_code,
            'battery_remaining_min': round(result.battery, 2),
            'early_stop_fired': result.early_stop_fired,
            'battery_reserve_fired': result.battery_reserve_fired,
            'cycle_cap_fired': result.cycle_cap_fired,
            'returned_home': result.returned_home,
            'final_position': f"({result.position[0]},{result.position[1]})",
            'seconds': round(time.perf_counter() - started, 2),
        })
        print(f"\r  mission {i}/{missions} (seed {seed}): {result.exit_code:<22} "
              f"{len(result.found)}/{result.survivors_total} found", end='', flush=True)
    print()
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, use_smt: bool) -> str:
    n = len(df)
    lines = [f"Autonomous mission evaluation — {n} missions, verifier: {'Z3 SMT' if use_smt else 'rule-based'}", ""]
    overview = pd.DataFrame({
        'metric': ['survivors found', 'cycles used', 'battery remaining (min)'],
        'mean': [df.survivors_found.mean(), df.cycles_used.mean(), df.battery_remaining_min.mean()],
        'median': [df.survivors_found.median(), df.cycles_used.median(), df.battery_remaining_min.median()],
        'min': [df.survivors_found.min(), df.cycles_used.min(), df.battery_remaining_min.min()],
        'max': [df.survivors_found.max(), df.cycles_used.max(), df.battery_remaining_min.max()],
    })
    lines.append(overview.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    lines.append("")
    found = df.survivors_found.value_counts().sort_index()
    lines.append("Survivors found per mission: "
                 + ", ".join(f"{k}/{df.survivors_total.iloc[0]}: {v} ({v / n:.0%})" for k, v in found.items()))
    lines.append("")
    exits = (df.groupby('exit_reason')
               .agg(missions=('seed', 'size'), avg_found=('survivors_found', 'mean'),
                    avg_cycles=('cycles_used', 'mean'), avg_battery=('battery_remaining_min', 'mean'))
               .sort_values('missions', ascending=False))
    exits['share'] = (exits.missions / n).map(lambda v: f"{v:.0%}")
    lines.append(exits.to_string(float_format=lambda v: f"{v:.2f}"))
    lines.append("")
    for column, label in (('early_stop_fired', 'Early stop fired'), ('battery_reserve_fired', 'Battery reserve fired'),
                          ('cycle_cap_fired', 'Cycle cap hit'), ('returned_home', 'Ended at base')):
        lines.append(f"{label:<22} {int(df[column].sum()):>4} / {n}  ({df[column].mean():.0%})")
    lines.append(f"{'Mean time per mission':<22} {df.seconds.mean():>6.2f} s")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--missions', type=int, default=100)
    parser.add_argument('--smt', choices=('on', 'off'), default='on')
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--fresh-navigator', action='store_true')
    parser.add_argument('--out', default='eval_results.csv')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    use_smt = args.smt == 'on'
    print(f"Running {args.missions} headless autonomous missions...")
    df = run_missions(args.missions, use_smt, args.fresh_navigator, args.first_seed)
    df.to_csv(args.out, index=False)
    print()
    print(summarise(df, use_smt))
    print(f"\nPer-mission results saved to {args.out}")


if __name__ == '__main__':
    main()
