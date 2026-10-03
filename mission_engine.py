"""
mission_engine.py: headless mission state and the autonomous search loop.

MissionSim works on a dict-like store holding the session state (bots, HMM localiser and belief, mission log,
verifier choice). The Streamlit app passes st.session_state, so the UI and the engine share one state; tests and
eval.py pass a plain dict from new_store() and run missions with no UI at all.

Autonomous loop (run_autonomous_mission), one cycle:
    SCANNING         localise with a sonar sweep if the HMM belief is too spread out, then the HMM-coupled
                     Thompson bandit picks a chamber
    NAVIGATING       Q-learning plans a route; it is bid against UNIT-02's route (the loser replans with A*
                     around the winner's cells) and proved by the active safety verifier before anything moves
    UPDATING BELIEF  one HMM predict/update step per move, with the real sonar wall count
    REWARD UPDATE    sonar sweep of the chamber; the bandit's posterior moves by the HMM position confidence
Ending conditions: all survivors found, cycle cap, early stop (search exhausted), battery reserve, battery
depleted, position still uncertain after a scan, or no reachable chambers. The cycle cap, early stop and battery
reserve endings drive the bot back to the entrance first.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

import numpy as np

from autonomous_mission import (DETECTION_PROB, HMMCoupling, QLearningNavigator, ThompsonBandit,
                                scan_detects_survivor)
from cave_environment import CaveEnvironment, RescueBot, create_fleet, get_wall_observation
from cave_hmm import HMM_BORDER, CaveHMMLocaliser
from confidence_gate import confidence_status, is_confident
from conflict_resolution import resolve_conflicts
from llm_parser import CHAMBER_MAP, CHAMBER_NAMES
from models import CaveCommand
from safety_verifier import SafetyVerifier
from smt_verifier import WATER_LIMIT as SMT_WATER_LIMIT, verify_command_smt

Cell = Tuple[int, int]

START_POSITION = CHAMBER_MAP[0]     # Bot is deployed at the cave entrance
INITIAL_BATTERY_MIN = 30.0
SCAN_SONAR_READINGS = 3             # Stationary sonar readings taken by a SCAN command
MINUTES_PER_MOVE = 0.1              # Battery drained per grid move
DEFAULT_PAYLOAD_KG = 1.0            # Assumed weight for items not listed below
PAYLOAD_WEIGHTS_KG = {
    'oxygen_kit': 3.0,
    'medical_kit': 2.0,
    'water': 1.5,
    'food': 1.0,
    'rope': 2.5,
    'radio': 0.8,
    'thermal_blanket': 0.5,
}

# Grid step (dx, dy) -> HMM action (0=N, 1=S, 2=E, 3=W)
STEP_TO_ACTION = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}
ACTION_NAMES = ['N', 'S', 'E', 'W']
HMM_BOT = 'UNIT-01'                 # The HMM localiser tracks UNIT-01; UNIT-02 reports its position directly

AUTO_ARMS = range(1, 10)            # Chambers the autonomous search can choose (C0 is the entrance)
AUTO_MAX_CYCLES = 30                # Hard stop, so a run of unlucky sonar misses cannot loop for ever
AUTO_EXHAUSTED_MEAN = 0.2           # Stop once every chamber still in play has a posterior mean below this
AUTO_HOME = CHAMBER_MAP[0]          # Base the bot must always keep enough battery to return to

# Exit codes reported in MissionResult.exit_code
EXIT_ALL_FOUND = 'all_found'
EXIT_CYCLE_CAP = 'cycle_cap'
EXIT_EARLY_STOP = 'early_stop'
EXIT_BATTERY_RESERVE = 'battery_reserve'
EXIT_BATTERY_DEPLETED = 'battery_depleted'
EXIT_UNCERTAIN = 'position_uncertain'
EXIT_NO_CHAMBERS = 'no_reachable_chambers'
EXIT_BLOCKED = 'blocked_by_unit02'

PhaseCallback = Callable[[str, int, str, ThompsonBandit], None]


@dataclass
class AutoConfig:
    max_cycles: int = AUTO_MAX_CYCLES
    exhausted_mean: float = AUTO_EXHAUSTED_MEAN
    detection_prob: float = DETECTION_PROB
    arms: Tuple[int, ...] = tuple(AUTO_ARMS)
    home: Cell = AUTO_HOME


@dataclass
class MissionResult:
    exit_code: str
    end_reason: str
    cycles: int
    found: List[Cell]
    survivors_total: int
    battery: float
    position: Cell
    returned_home: bool
    early_stop_fired: bool
    battery_reserve_fired: bool
    cycle_cap_fired: bool
    summary: str
    bandit_table: List[dict] = field(default_factory=list)


def mission_record(intent: CaveCommand) -> dict:
    """The parsed-intent columns MissionDatabase.log_mission stores for a mission"""
    return {'action': intent.action, 'target_chamber': intent.target_chamber, 'payload': intent.payload,
            'priority': intent.priority}


def payload_weight(payload: Optional[str]) -> float:
    return PAYLOAD_WEIGHTS_KG.get(payload, DEFAULT_PAYLOAD_KG) if payload else 0.0


def new_store(cave: CaveEnvironment, use_smt: bool = True) -> Dict[str, Any]:
    """A fresh headless session store"""
    store: Dict[str, Any] = {'use_smt': use_smt}
    MissionSim(cave, store).init_store()
    return store


class MissionSim:
    """Mission state operations over a store (st.session_state in the app, a dict headless)"""

    def __init__(self, cave: CaveEnvironment, store: MutableMapping[str, Any], db=None,
                 clock: Callable[[], datetime] = datetime.now):
        self.cave = cave
        self.store = store
        self.db = db
        self.clock = clock

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------
    def init_store(self):
        """Creates the localiser (belief certain at the entrance), the fleet and the log, if not there yet"""
        if 'localiser' in self.store:
            return
        self.store['localiser'] = CaveHMMLocaliser(self.cave)  # Per session: the estimator holds mutable belief
        self.reset_belief(START_POSITION)
        self.store['bots'] = create_fleet(INITIAL_BATTERY_MIN)
        self.store['mission_log'] = []
        for bot in self.store['bots'].values():
            self.log(f"System online. {bot.bot_id} deployed at {bot.position}, battery {bot.battery:.1f} min.")

    @property
    def bots(self) -> Dict[str, RescueBot]:
        return self.store['bots']

    @property
    def use_smt(self) -> bool:
        return bool(self.store.get('use_smt', True))

    def log(self, message: str):
        self.store['mission_log'].append(f"[{self.clock():%H:%M:%S}] {message}")

    def localiser(self) -> CaveHMMLocaliser:
        """The localiser with its estimator loaded from the stored belief"""
        loc = self.store['localiser']
        loc.estimator.belief_state = self.store['belief']
        return loc

    def map_estimate(self) -> Tuple[int, int, float]:
        """(x, y, probability) of the highest-belief cell"""
        x, y = self.localiser().most_likely_position()
        return x, y, float(self.store['belief'].max())

    def reset_belief(self, position: Cell):
        """UNIT-01 is known to be at position"""
        hmm_env = self.store['localiser'].hmm_env
        belief = np.zeros(hmm_env.num_states)
        belief[hmm_env.coord_to_state(position[1] + HMM_BORDER, position[0] + HMM_BORDER)] = 1.0
        self.store['belief'] = belief

    def navigator(self, seed: Optional[int] = None) -> QLearningNavigator:
        """Q-learning navigator, kept in the store (it caches a Q-table per chamber). With the SMT verifier on,
        flooded cells are walls to it, since SMT rejects deep water anywhere on a route."""
        if self.store.get('navigator_smt') != self.use_smt or 'navigator' not in self.store:
            flooded = ([cell for cell, info in self.cave.cave_map.items() if info.water_level >= SMT_WATER_LIMIT]
                       if self.use_smt else [])
            self.store['navigator'] = QLearningNavigator(self.cave, blocked=flooded, seed=seed)
            self.store['navigator_smt'] = self.use_smt
        return self.store['navigator']

    # -----------------------------------------------------------------
    # Verification and conflict resolution
    # -----------------------------------------------------------------
    def verify_route(self, intent: CaveCommand, path, payload_kg: float, bot: RescueBot) -> dict:
        """Checks a route with the chosen verifier (Z3 SMT or rule-based) and logs the verdict"""
        if self.use_smt:
            check = verify_command_smt(intent, path, self.cave, payload_kg, bot.battery)
        else:  # Rule-based check_mission, wrapped in the same result dict
            check = SafetyVerifier(self.cave, battery_remaining_min=bot.battery, use_smt=False).verify(
                intent, path, self.cave, payload_kg)
        self.log(f"{'SMT VERIFIER' if self.use_smt else 'SAFETY VERIFIER'}: {'PASS' if check['safe'] else 'FAIL'} "
                 f"[{bot.bot_id}] (payload {intent.payload or 'none'} = {payload_kg} kg, battery {bot.battery:.1f} min) "
                 f"{check['proof']}, {check['checked_cells']} cells, {check['solver_result']}"
                 + ("" if check['safe'] else f" -> {check['violations']}"))
        return check

    def deconflict(self, bot: RescueBot, other: RescueBot, path, priority: str):
        """Runs the bidding protocol against the other bot's route in progress; returns (own route, other's route).

        The loser replans with A* treating the winner's contested cells as rock, repeating until its detour shares
        no cell with the winner's route; [] means it has no detour and holds.
        """
        if not path or len(other.active_route) < 2:
            return path, other.active_route
        first, second = sorted((bot, other), key=lambda b: b.bot_id)  # UNIT-01 is always bot 1 (wins ties)
        routes = {bot.bot_id: path, other.bot_id: other.active_route}
        priorities = {bot.bot_id: priority, other.bot_id: other.priority}
        final_1, final_2, resolution_log = resolve_conflicts(
            routes[first.bot_id], routes[second.bot_id], priorities[first.bot_id], first.battery,
            priorities[second.bot_id], second.battery, env=self.cave, bot1_name=first.bot_id,
            bot2_name=second.bot_id)
        for line in resolution_log:
            self.log(f"CONFLICT: {line}")
        finals = {first.bot_id: final_1, second.bot_id: final_2}
        return finals[bot.bot_id], finals[other.bot_id]

    def yield_route(self, other: RescueBot, detour, winner: Optional[RescueBot] = None, winner_route=None):
        """Applies a lost bid to the other bot's route in progress: take the detour if safe, else hold at its start.

        Records the change in store['last_reroute'] so the map can draw the cancelled and the new route.
        """
        old_route = list(other.active_route)
        old_moves = len(old_route) - 1
        record = {'bot_id': other.bot_id, 'old': old_route, 'new': [],
                  'winner': winner.bot_id if winner else None, 'winner_route': list(winner_route or [])}
        if detour:
            # Same verifier as every other route; the battery includes the refund for the cancelled route
            battery = other.battery + old_moves * MINUTES_PER_MOVE
            if self.use_smt:
                intent = CaveCommand(action='move', priority=other.priority if other.priority in ('low', 'high') else
                                     'normal', raw_text=f"[CONFLICT] {other.bot_id} detour")
                check = verify_command_smt(intent, detour, self.cave, payload_weight(other.payload), battery)
                ok, problems = check['safe'], check['violations']
            else:
                verifier = SafetyVerifier(self.cave, battery_remaining_min=battery)
                ok, problems = verifier.check_mission(payload_weight(other.payload), detour[-1], detour)
            if ok:
                other.battery += (old_moves - (len(detour) - 1)) * MINUTES_PER_MOVE
                other.active_route = list(detour)
                record['new'] = list(detour)
                self.store['last_reroute'] = record
                self.log(f"CONFLICT REROUTE: {other.bot_id} replanned with A* around "
                         f"{record['winner'] or 'the winner'}'s route: {len(detour) - 1} moves (was {old_moves}); "
                         f"{' -> '.join(f'({x},{y})' for x, y in detour)}; battery {other.battery:.1f} min")
                return
            self.log(f"CONFLICT: {other.bot_id} detour fails the safety check ({'; '.join(problems)}); holding instead")
        other.battery += old_moves * MINUTES_PER_MOVE  # It never travelled the cancelled route
        other.position = old_route[0]
        other.active_route = []
        self.store['last_reroute'] = record
        if other.bot_id == HMM_BOT:
            self.reset_belief(other.position)
        self.log(f"CONFLICT: {other.bot_id} holds at {other.position}; route cancelled, "
                 f"{old_moves * MINUTES_PER_MOVE:.1f} min refunded")

    # -----------------------------------------------------------------
    # HMM
    # -----------------------------------------------------------------
    def simulate_hmm(self, path):
        """Runs a full predict/update filter step with the real sonar wall count at every cell along the route"""
        loc = self.localiser()
        moves = list(zip(path, path[1:]))
        if not moves:
            self.log("HMM: no movement, belief unchanged.")
            return
        for i, ((x0, y0), (x1, y1)) in enumerate(moves, 1):
            action = STEP_TO_ACTION[(x1 - x0, y1 - y0)]
            observation = get_wall_observation(self.cave, (x1, y1))  # Real wall count at the bot's true cell
            belief = loc.step(action, observation)
            x, y = loc.most_likely_position()
            self.log(f"HMM STEP {i}/{len(moves)}: action={ACTION_NAMES[action]} sonar_walls={observation} "
                     f"-> MAP ({x}, {y}) p={belief.max():.3f}")
        self.store['belief'] = loc.estimator.belief_state.copy()

    def scan_localise(self, position: Cell) -> dict:
        """Stationary sonar sweep: SCAN_SONAR_READINGS measurement updates at position; returns confidence_status"""
        loc = self.localiser()
        observation = get_wall_observation(self.cave, position)
        for i in range(1, SCAN_SONAR_READINGS + 1):
            belief = loc.sense(observation)
            x, y = loc.most_likely_position()
            self.log(f"HMM SCAN {i}/{SCAN_SONAR_READINGS}: sonar_walls={observation} -> MAP ({x}, {y}) "
                     f"p={belief.max():.3f}, entropy {loc.belief_entropy():.3f}")
        self.store['belief'] = loc.estimator.belief_state.copy()
        return confidence_status(loc)

    # -----------------------------------------------------------------
    # Autonomous mission
    # -----------------------------------------------------------------
    def run_autonomous_mission(self, rng: Optional[np.random.Generator] = None,
                               config: Optional[AutoConfig] = None,
                               navigator: Optional[QLearningNavigator] = None,
                               on_phase: Optional[PhaseCallback] = None) -> MissionResult:
        """Searches for survivors with UNIT-01 (the HMM-tracked unit) until an ending condition is met"""
        config = config or AutoConfig()
        rng = rng if rng is not None else np.random.default_rng()
        bot = self.bots[HMM_BOT]
        other = next(b for b in self.bots.values() if b.bot_id != HMM_BOT)
        navigator = navigator or self.navigator()
        bandit = ThompsonBandit(config.arms, rng, hmm=HMMCoupling(self.localiser, navigator, CHAMBER_MAP))
        survivors = set(self.cave.survivor_locations)
        found: set = set()
        flags = {'early_stop': False, 'reserve': False, 'cycle_cap': False, 'returned_home': False}
        self.log(f"AUTO: mission start — {HMM_BOT} at {bot.position}, battery {bot.battery:.1f} min, "
                 f"{len(survivors)} survivors to find, sonar detection p={config.detection_prob}")

        def show(phase: str, cycle: int, detail: str):
            if on_phase:
                on_phase(phase, cycle, detail, bandit)

        def drive(route, intent: CaveCommand, other_final, cycle: int, label: str, check: dict) -> float:
            """Executes a verified route and runs the HMM over it; returns the battery before the move"""
            moves = len(route) - 1
            show('NAVIGATING', cycle, f"Q-learning route to {label}: **{moves} moves**, verified "
                                      f"({check['proof']}, {check['checked_cells']} cells).")
            if other_final != other.active_route:
                self.yield_route(other, other_final, winner=bot, winner_route=route)
            battery_before = bot.battery
            self.log(f"Q-LEARNING PATH [{HMM_BOT}]: {' -> '.join(f'({x},{y})' for x, y in route)}")
            bot.position = route[-1]
            bot.battery -= moves * MINUTES_PER_MOVE
            bot.active_route = route
            bot.payload = None
            bot.priority = intent.priority

            # UPDATING BELIEF: one HMM filter step with the real wall count per move
            show('UPDATING BELIEF', cycle, f"HMM filter: {moves} predict/update steps along the route...")
            self.simulate_hmm(route)
            status = confidence_status(self.localiser())
            x, y, p = self.map_estimate()
            show('UPDATING BELIEF', cycle, f"Belief updated over {moves} moves: MAP ({x}, {y}) p={p:.2f}, "
                                           f"entropy {status['entropy']:.3f} — {status['message']}.")
            return battery_before

        def return_to_base(cycle: int):
            """Drives back to the entrance on the learned route; logs and stays put if it cannot be verified"""
            if bot.position == config.home:
                self.log(f"AUTO [{cycle}]: {HMM_BOT} is already at base {config.home}")
                flags['returned_home'] = True
                return
            intent = CaveCommand(action='move', target_chamber=0, raw_text="[AUTO] return to base")
            route = navigator.route(bot.position, config.home)
            route, other_final = (self.deconflict(bot, other, route, intent.priority) if route
                                  else ([], other.active_route))
            check = self.verify_route(intent, route, 0.0, bot) if route else None
            if not check or not check['safe']:
                problem = '; '.join(check['violations']) if check else 'no verified route home'
                self.log(f"AUTO [{cycle}]: cannot return to base ({problem}) — {HMM_BOT} holds at {bot.position}")
                return
            battery_before = drive(route, intent, other_final, cycle, "base (C0 Entrance)", check)
            flags['returned_home'] = True
            self.log(f"AUTO [{cycle}]: {HMM_BOT} back at base, battery {bot.battery:.1f} min")
            if self.db:
                self.db.log_mission("[AUTO] return to base", mission_record(intent), route, (True, []),
                                    battery_before, bot.battery, bot.position, "Q-learning", bot_id=HMM_BOT)

        cycle, end_reason, exit_code = 0, '', ''
        while not exit_code:
            if found == survivors:
                end_reason, exit_code = "all survivors found", EXIT_ALL_FOUND
                break
            if cycle >= config.max_cycles:
                flags['cycle_cap'] = True
                self.log(f"AUTO [{cycle}]: cycle limit ({config.max_cycles}) reached — returning to base")
                return_to_base(cycle)
                end_reason, exit_code = f"cycle limit ({config.max_cycles}) reached", EXIT_CYCLE_CAP
                break
            if bot.battery < MINUTES_PER_MOVE:  # Cannot make even one more move
                end_reason, exit_code = "battery depleted", EXIT_BATTERY_DEPLETED
                break
            active = bandit.active_arms()
            if active and all(bandit.mean(arm) < config.exhausted_mean for arm in active):
                flags['early_stop'] = True
                self.log(f"AUTO [{cycle}]: every remaining chamber's posterior mean is below "
                         f"{config.exhausted_mean} — returning to base")
                return_to_base(cycle)
                end_reason, exit_code = "search exhausted — no promising chambers remain", EXIT_EARLY_STOP
                break
            cycle += 1

            # SCANNING: localise if the belief is too spread out, then let the bandit pick a chamber to search
            if not is_confident(self.localiser()):
                show('SCANNING', cycle, f"Position uncertain — stationary sonar sweep at {bot.position} to localise.")
                self.scan_localise(bot.position)
                if not is_confident(self.localiser()):
                    end_reason, exit_code = "position still uncertain after a sonar sweep", EXIT_UNCERTAIN
                    break
            tried: List[int] = []
            route, arm, other_final, check, go_home = [], None, None, None, False
            target, intent = config.home, None
            while True:
                arm = bandit.select(exclude=tried)
                if arm is None:
                    break
                target = CHAMBER_MAP[arm]
                sample, mean, std = bandit.last_samples[arm], bandit.mean(arm), bandit.std(arm)
                reach = bandit.last_weights[arm]
                show('SCANNING', cycle, f"Thompson sampling picked **C{arm} {CHAMBER_NAMES[arm]}** — sampled "
                                        f"{sample:.2f} from its posterior (mean {mean:.2f} ± {std:.2f}) × "
                                        f"HMM reach weight {reach:.2f}.")
                self.log(f"AUTO [{cycle}]: bandit picked C{arm} (sample {sample:.2f} x reach {reach:.2f}, "
                         f"mean {mean:.2f}, std {std:.2f})")
                route = navigator.route(bot.position, target)
                if not route:
                    bandit.retire(arm, 'unreachable')
                    self.log(f"AUTO [{cycle}]: C{arm} unreachable without entering flooded cells — "
                             f"removed from the search")
                    continue
                # Battery reserve: going there must still leave enough to get back to the entrance
                home_route = navigator.route(target, config.home)
                if not home_route:
                    bandit.retire(arm, 'no safe return route')
                    self.log(f"AUTO [{cycle}]: C{arm} has no safe route back to base — removed from the search")
                    continue
                round_trip = (len(route) - 1) + (len(home_route) - 1)
                if round_trip * MINUTES_PER_MOVE >= bot.battery:
                    self.log(f"AUTO [{cycle}]: C{arm} and back to base needs {round_trip * MINUTES_PER_MOVE:.1f} "
                             f"min, battery {bot.battery:.1f} min — returning to base")
                    go_home = True
                    break
                intent = CaveCommand(action='move', target_chamber=arm, raw_text=f"[AUTO] search chamber {arm}")
                route, other_final = self.deconflict(bot, other, route, intent.priority)
                if not route:
                    tried.append(arm)
                    self.log(f"AUTO [{cycle}]: C{arm} route lost the bid against {other.bot_id}'s route — "
                             f"trying another chamber")
                    continue
                check = self.verify_route(intent, route, 0.0, bot)
                if not check['safe']:
                    reason = 'out of battery range' if 'BATTERY INSUFFICIENT' in check['violations'] else 'unsafe route'
                    bandit.retire(arm, reason)
                    self.log(f"AUTO [{cycle}]: C{arm} {reason} — removed from the search")
                    continue
                break
            if go_home:
                flags['reserve'] = True
                return_to_base(cycle)
                end_reason = "returning to base — insufficient battery for further search"
                exit_code = EXIT_BATTERY_RESERVE
                break
            if arm is None or intent is None or check is None:
                reasons = set(bandit.retired.values())
                if 'out of battery range' in reasons:
                    end_reason, exit_code = "battery depleted", EXIT_BATTERY_DEPLETED
                elif tried:
                    end_reason, exit_code = "remaining chambers are blocked by UNIT-02's route", EXIT_BLOCKED
                else:
                    end_reason, exit_code = "no reachable chambers left to search", EXIT_NO_CHAMBERS
                break

            # NAVIGATING + UPDATING BELIEF
            battery_before = drive(route, intent, other_final, cycle, f"C{arm}", check)

            # REWARD UPDATE: sonar sweep of the chamber, reward 1 if a survivor is detected
            detected = (scan_detects_survivor(self.cave, target, rng, config.detection_prob)
                        and target not in found)
            bandit.update(arm, int(detected))
            if detected:
                found.add(target)
                bandit.retire(arm, 'survivor rescued')
            outcome = (f"SURVIVOR FOUND at C{arm} ({len(found)}/{len(survivors)}) — reward 1, posterior raised"
                       if detected else f"C{arm} sweep found nobody — reward 0, posterior lowered")
            self.log(f"AUTO [{cycle}]: {outcome} (weight {bandit.last_confidence:.2f} from HMM position "
                     f"confidence); C{arm} now Beta({bandit.alpha[arm]:.2f}, {bandit.beta[arm]:.2f}); "
                     f"battery {bot.battery:.1f} min")
            show('REWARD UPDATE', cycle, f"{outcome}. Battery {bot.battery:.1f} min.")
            if self.db:
                self.db.log_mission(f"[AUTO] search chamber {arm}", mission_record(intent), route, (True, []),
                                    battery_before, bot.battery, bot.position, "Q-learning", bot_id=HMM_BOT)

        missed = sorted(survivors - found)
        summary = (f"{end_reason.capitalize()} after {cycle} cycles: {len(found)} of {len(survivors)} survivors found"
                   + (f" (not found: {', '.join(f'({x},{y})' for x, y in missed)})" if missed else "")
                   + f". {HMM_BOT} at {bot.position}, battery {bot.battery:.1f} min.")
        self.log(f"AUTO: mission complete — {summary}")
        show('COMPLETE', cycle, summary)
        return MissionResult(exit_code=exit_code, end_reason=end_reason, cycles=cycle, found=sorted(found),
                             survivors_total=len(survivors), battery=bot.battery, position=bot.position,
                             returned_home=flags['returned_home'], early_stop_fired=flags['early_stop'],
                             battery_reserve_fired=flags['reserve'], cycle_cap_fired=flags['cycle_cap'],
                             summary=summary, bandit_table=bandit.table())
