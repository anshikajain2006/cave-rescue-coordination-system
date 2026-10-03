# Safety Notes

> **This is a simulation.** The cave, its water levels, passage widths, sonar model (65% accurate wall counts,
> 85% survivor detection) and battery drain (0.1 min per move) are made-up parameters, not measurements. The
> safety guarantees below hold for this model only. A real deployment would need calibrated sensors (sonar range
> and error, water depth, passage measurements), a validated battery and drive model, live map updates as water
> rises, and testing on hardware. Without those, a proof against this model says nothing about a real cave.

## 1. Properties the SMT verifier proves

`smt_verifier.verify_path_smt()` turns each rule into a Z3 constraint over the planned route (the full cell sequence,
start included) and asks Z3 to satisfy all of them together. The route is `VERIFIED` only if the conjunction is
satisfiable. Each constraint is tracked, so a `VIOLATED` result names every rule that fails. All numbers are exact
rationals (0.8 is 4/5), so there are no floating-point edge cases.

For a route `p₀ … pₙ`, bot payload `w` and remaining battery `B` minutes:

| # | Property | Constraint |
|---|---|---|
| P1 | A route exists | `n ≥ 0` (an empty route is `NO PATH FOUND`) |
| P2 | Every cell is open cave | `pᵢ ∈ cave_map` for all i (not rock, not off the map) |
| P3 | Moves are single grid steps | `|xᵢ − xᵢ₋₁| + |yᵢ − yᵢ₋₁| = 1` for all i ≥ 1 |
| P4 | Every passage is wide enough | `width(pᵢ) > d`, with `d = 0.25 m` empty or `0.6 m` when `w > 0` |
| P5 | No deep water anywhere on the route | `water(pᵢ) < 0.8` for all i |
| P6 | Enough battery | `n / 10 < B` (0.1 min per move, strictly less than what is left) |

Commands where the bot does not move (`scan`, `status`, `hold`) have no route and are trivially `VERIFIED`.

The rule-based `SafetyVerifier.check_mission()`, used when the SMT toggle is off, checks P1–P4 and P6, but checks
**water only at the destination**. That is why it can reach flooded chambers the SMT verifier refuses.

Other safety checks outside the verifier:

- **Localisation gate.** UNIT-01 may only `move`, `deliver` or `replan` while its HMM belief entropy is below 3.5
  nats. Otherwise the command is refused with "run SCAN first".
- **Battery reserve (autonomous mode).** The trip to a chamber plus the learned route from it back to C0 must take
  less than the remaining battery.
- **Validated commands.** Parsed commands are Pydantic `CaveCommand`s: unknown actions, chambers outside 0–9 or
  malformed LLM output are rejected before planning.

## 2. What happens when a constraint fires

| Situation | Response |
|---|---|
| Operator route fails P4 only, and the bot carries a payload | **Reroute.** The MDP pilot replans with the too-narrow cells (and, with SMT on, the flooded cells) as walls. The detour is bid against the other bot and re-verified; if `VERIFIED`, the bot drives it ("Rerouted around passages too narrow for the load"). If not, the mission is suspended. |
| Operator route fails any other rule (P1, P2, P3, P5, P6, or P4 on an empty bot) | **Suspend.** The bot holds position; the violations are shown and logged (`SMT_VIOLATION` in the database), and the MDP pilot's alternative is shown as advice only. |
| HMM belief too uncertain | **Suspend.** `BLOCKED_UNCERTAINTY`; the operator must scan first. In autonomous mode the bot scans by itself, and ends the mission if it is still uncertain. |
| Two routes share cells, and this bot loses the bid | **Reroute** with A* around the winner's cells, re-verified with the active verifier. With no safe detour the loser **holds**, and the battery for its cancelled route is refunded. |
| Autonomous: chamber unreachable without flooded cells, no safe route back, or its route fails verification | The chamber is **removed from the search**; the bandit picks another. |
| Autonomous: next trip plus the way home no longer fits the battery | **Return to base** on a verified route, then end ("returning to base — insufficient battery"). |
| Autonomous: every remaining chamber's posterior mean < 0.2 | **Return to base**, then end ("search exhausted"). |
| Autonomous: 30 cycles used | **Return to base**, then end. |
| Autonomous: battery cannot make a single move | End where it is (the reserve check is meant to prevent this). |

Nothing moves on a `VIOLATED` result. That covers operator routes, autonomous routes, returns to base, MDP detours
and conflict detours.

## 3. Known failure modes

- **S1 is unreachable with SMT on.** Every route into the Lower Sump (C2), where survivor S1 is, crosses water ≥ 0.8,
  as does every route into C8. With SMT on, the autonomous search never finds S1 (0 of 100 evaluation missions).
  This is the verifier working as designed, but it means a survivor is left behind. A human has to decide whether
  to accept the flooding risk, by switching to the rule-based verifier, which only checks the destination's water.
- **No safe route home.** If the verified route back to C0 fails, for example because the other bot's route
  holds a cell on the only way back or the battery is already too low, the bot logs "cannot return to base" and
  **stays where it is**. The battery reserve makes the second case unlikely but not impossible: the reserve uses the
  Q-learning route home planned before the trip, and the route actually planned on the way back can be longer or be
  blocked by the other bot.
- **Ending away from base.** When every survivor is found, or no reachable chamber is left, the mission ends at the
  last chamber without driving home.
- **Planning uses the true position.** Routes are planned and verified from the bot's true cell. The HMM belief only
  gates navigation and weights the bandit. A real robot would plan from its estimated position, and a wrong but
  confident estimate (sonar wall counts are ambiguous in symmetric passages) would make a "verified" route wrong.
- **Static water.** Water levels never change during a mission. A route verified at the start stays verified even
  if a real cave were flooding.
- **Rule-based verifier is weaker.** With SMT off, a route can pass through deep water to reach a dry destination.
- **Instant missions.** Moves happen in one step, and conflict bidding compares whole routes, not timed positions.
  Two bots are never checked for being in the same cell at the same moment, because time is not modelled.
- **Parser mistakes.** A command can parse into a valid but wrong chamber, for example from an ambiguous nickname.
  The safety proof then covers the wrong route. The parse trace and mission log show what was understood.
