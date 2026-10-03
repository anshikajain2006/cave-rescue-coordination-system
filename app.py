import base64
import html
import io
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import llm_parser
import react_parser
from autonomous_mission import (DETECTION_PROB, HMMCoupling, QLearningNavigator, ThompsonBandit,
                                scan_detects_survivor)
from cave_db import MissionDatabase
from cave_environment import CaveEnvironment, RescueBot, create_fleet, get_wall_observation
from cave_mdp import CaveMDPPilot
from llm_parser import CHAMBER_MAP, CHAMBER_NAMES
from models import CaveCommand
from react_parser import AwaitingClarification, keyword_parse, react_parse, resume_react
from planner import ForkliftPlanner
from safety_verifier import EMPTY_BOT_DIAMETER_M, LOADED_BOT_DIAMETER_M, SafetyVerifier
from smt_verifier import WATER_LIMIT as SMT_WATER_LIMIT, verify_command_smt
from conflict_resolution import resolve_conflicts

st.set_page_config(layout="wide", page_icon="🔦", page_title="CAVE RESCUE OPS")
st.markdown('<style>section.main > div {max-width: 100%; padding-left: 1rem; padding-right: 1rem;} .block-container {padding-top: 1rem; padding-bottom: 1rem;}</style>', unsafe_allow_html=True)

try:
    from cave_hmm import HMM_BORDER, CaveHMMLocaliser
    from confidence_gate import confidence_status, is_confident
except SyntaxError as exc:  # hmm_environment.py currently has mixed tab/space indentation
    st.error(f"Could not load the HMM module: {type(exc).__name__} in {exc.filename}, line {exc.lineno}. "
             "Fix the indentation in _build_transition_matrix in hmm_environment.py and reload.")
    st.stop()

START_POSITION = CHAMBER_MAP[0]     # Bot is deployed at the cave entrance
INITIAL_BATTERY_MIN = 30.0
SCAN_SONAR_READINGS = 3             # Stationary sonar readings taken by a SCAN command
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
BOT_IDS = ('UNIT-01', 'UNIT-02')
HMM_BOT = 'UNIT-01'                 # The HMM localiser tracks UNIT-01; UNIT-02 reports its position directly

# =====================================================================
# THEME: geological survey
# =====================================================================
PAPER = '#f5f0e8'
SIDEBAR_BG = '#ede8dc'
CARD_BG = '#ffffff'
TEXT = '#1a1a1a'
TEXT_SECONDARY = '#5c5c5c'
BORDER = '#c8bfa8'
ROCK = '#4a3728'
DRY_CELL = '#e8e0d0'
WATER_LOW = '#d4e8f0'
WATER_HIGH = '#1a6b9a'
SURVEY_RED = '#cc2200'
SURVEY_BLUE = '#005580'
ACCENT = '#2c5f8a'
SUCCESS = '#2d6a2d'
WARNING = '#8a5a00'
DANGER = '#8a1a00'
PURPLE = '#5a2d82'
MDP_COLOR = '#1f6f6f'
UNIT2_COLOR = '#7b3fa0'
CONFLICT_COLOR = '#a0522d'
LOW_BATTERY_FRACTION = 0.3
MINUTES_PER_MOVE = 0.1          # Same drain rate run_command applies to the battery
AUTO_ARMS = range(1, 10)        # Chambers the autonomous search can choose (C0 is the entrance)
AUTO_MAX_CYCLES = 30            # Hard stop, so a run of unlucky sonar misses cannot loop for ever
AUTO_STEP_DELAY_S = 0.8         # Pause after each phase so the run can be followed on screen
AUTO_EXHAUSTED_MEAN = 0.2       # Stop once every chamber still in play has a posterior mean below this
AUTO_HOME = CHAMBER_MAP[0]      # Base the bot must always keep enough battery to return to
AUTO_PHASE_COLORS = {'SCANNING': '#005580', 'NAVIGATING': '#b35c00', 'UPDATING BELIEF': '#6a3d9a',
                     'REWARD UPDATE': '#2e7d32', 'COMPLETE': '#5c5346'}
METRES_PER_CELL = 2
FIG_SERIF = ['Source Serif 4', 'Georgia', 'Cambria', 'Times New Roman', 'DejaVu Serif']
FIG_SANS = ['Source Sans 3', 'Segoe UI', 'Arial', 'DejaVu Sans']


# Dry cells blend into the paper; water deepens from pale to survey blue
WATER_CMAP = LinearSegmentedColormap.from_list('hydrology', [DRY_CELL, WATER_LOW, WATER_HIGH])
WATER_CMAP.set_bad('none')
BELIEF_CMAP = LinearSegmentedColormap.from_list('posterior', [DRY_CELL, '#b8d4e8', SURVEY_BLUE])
BELIEF_CMAP.set_bad('none')
# Tops out at a mid green so the policy arrows stay legible on high-value cells
VALUE_CMAP = LinearSegmentedColormap.from_list('mdp_value', [DRY_CELL, '#c6d8b0', '#7fa86b'])
VALUE_CMAP.set_bad('none')

APP_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

:root {{
  --paper: {PAPER}; --sidebar: {SIDEBAR_BG}; --card: {CARD_BG}; --text: {TEXT}; --text2: {TEXT_SECONDARY};
  --border: {BORDER}; --accent: {ACCENT}; --success: {SUCCESS}; --warning: {WARNING}; --danger: {DANGER};
  --serif: 'Source Serif 4', Georgia, Cambria, 'Times New Roman', serif;
  --sans: 'Source Sans 3', 'Segoe UI', Arial, sans-serif;
  --mono: 'Courier Prime', 'Courier New', Consolas, ui-monospace, monospace;
  --shadow: 0 1px 2px rgba(74, 55, 40, 0.08), 0 2px 6px rgba(74, 55, 40, 0.06);
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--paper); color: var(--text); }}
.stApp *:not([data-testid="stIconMaterial"]) {{ font-family: var(--sans); }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
[data-testid="stElementContainer"]:has(style) {{ display: none; }}  /* The CSS block itself takes no space */

[data-testid="stMainBlockContainer"] {{ padding: 0 1.5rem 1.5rem; max-width: 100%; }}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}

/* Sidebar: graph-paper diagonal texture */
[data-testid="stSidebar"] {{
  background-color: var(--sidebar);
  background-image: repeating-linear-gradient(45deg, transparent 0 19px, #d4cdb8 19px 20px);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebarContent"] {{ background: transparent; }}
[data-testid="stSidebar"][aria-expanded="true"] {{ min-width: 310px !important; max-width: 310px !important; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.4rem; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

/* Inputs & buttons */
[data-testid="stWidgetLabel"] p {{ font-size: 0.8rem; font-weight: 600; color: var(--text2); }}
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
  background: #fff !important; border-color: var(--border) !important; border-radius: 2px;
}}
[data-testid="stTextInput"] input {{ background: #fff !important; color: var(--text) !important; font-size: 0.9rem; }}
[data-testid="stTextInput"] input::placeholder {{ color: #9a9384 !important; }}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{ border-color: var(--accent) !important; }}
[data-testid="InputInstructions"] {{ display: none; }}
[data-testid="stRadio"] label p, [data-testid="stRadio"] label div {{ color: var(--text) !important; font-weight: 600; }}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
  background: #fff !important; border: 1px solid var(--border) !important; border-radius: 2px;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] * {{ color: var(--text) !important; }}
[data-testid="stSelectbox"] svg {{ fill: var(--text2) !important; }}
[data-baseweb="popover"] li {{ background: #fff; color: var(--text); }}
[data-testid="stTooltipIcon"] svg {{ stroke: var(--text2); }}
[data-testid="stTextInput"] button svg {{ fill: var(--text2); }}
[data-testid="stForm"] {{ border: 1px solid var(--border); border-radius: 4px; background: #fff; padding: 12px;
  box-shadow: var(--shadow); }}
.stButton button, [data-testid="stFormSubmitButton"] button {{
  background: var(--accent); color: #fff; border: none; border-radius: 0; min-height: 2.25rem;
}}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: #234d71; color: #fff; }}
.stButton button:focus:not(:active), [data-testid="stFormSubmitButton"] button:focus:not(:active) {{ color: #fff; }}
.stButton button p, [data-testid="stFormSubmitButton"] button p {{ font-weight: 600; font-size: 0.88rem; color: #fff; }}
[data-testid="stSpinner"] p {{ color: var(--text2); }}

/* Tabs: plain underline */
[data-baseweb="tab-list"] {{ gap: 1.8rem; background: transparent; border-bottom: 1px solid var(--border); }}
[data-baseweb="tab"] {{ background: transparent !important; padding: 0.5rem 0 !important; }}
[data-baseweb="tab"] p {{ color: var(--text2); font-size: 0.95rem; }}
[data-baseweb="tab"][aria-selected="true"] p {{ color: var(--text); font-weight: 700; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--text) !important; height: 2px !important; }}
[data-baseweb="tab-border"] {{ display: none; }}

/* Building blocks */
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 4px; box-shadow: var(--shadow); padding: 14px 16px; }}
.section-head {{
  font-family: var(--serif) !important; font-variant: small-caps; text-transform: lowercase; letter-spacing: 0.14em;
  font-size: 13px; font-weight: 600; color: var(--text2); margin: 0.5rem 0 0.35rem;
}}
.rule {{ border: none; border-top: 1px solid var(--border); border-bottom: 1px solid #f7f3ea; margin: 12px 0 6px; }}
.mono {{ font-family: var(--mono) !important; }}

/* Sidebar blocks */
.brand-title {{ font-family: var(--serif) !important; font-size: 20px; font-weight: 700; color: {ROCK}; line-height: 1.2; }}
.brand-sub {{ font-size: 11px; color: var(--text2); margin-top: 3px; letter-spacing: 0.02em; }}
.form-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.form-table td {{ border-bottom: 1px solid #e6dfcf; padding: 6px 2px; vertical-align: middle; }}
.form-table tr:last-child td {{ border-bottom: none; }}
.form-table td.k {{ color: var(--text2); width: 38%; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; }}
.form-table td.v {{ color: var(--text); font-weight: 600; }}
.form-table td.v.coord {{ font-family: var(--mono) !important; font-weight: 400; color: {SURVEY_BLUE}; }}
.battery-track {{ height: 5px; background: #eee8da; border: 1px solid #ddd4c0; margin-top: 4px; }}
.battery-fill {{ height: 100%; }}
.status-dot {{ display: inline-block; width: 8px; height: 8px; margin-right: 6px; vertical-align: 0; }}
.legend-list {{ font-size: 12.5px; line-height: 1.95; color: var(--text); }}
.legend-list .sq {{ display: inline-block; width: 10px; height: 10px; border: 1px solid #8f8570; margin-right: 8px; vertical-align: -1px; }}
.legend-list .id {{ font-weight: 700; margin-right: 4px; }}
.legend-list .xy {{ color: var(--text2); font-family: var(--mono) !important; font-size: 11.5px; }}
.legend-list .svr {{ color: {SURVEY_RED}; font-size: 10px; margin-left: 4px; }}
.legend-note {{ font-size: 10.5px; color: var(--text2); margin-top: 4px; font-style: italic; }}

/* Header */
.survey-header {{
  display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 0.5rem 1rem;
  background: #fff; border-bottom: 1px solid var(--border); margin: 0 -1.5rem; padding: 12px 1.5rem;
  box-shadow: 0 1px 3px rgba(74, 55, 40, 0.06);
}}
.survey-title {{ font-family: var(--serif) !important; font-size: 16px; font-weight: 700; color: var(--text); }}
.survey-meta {{ font-size: 11px; color: var(--text2); text-align: center; white-space: nowrap; }}
.survey-badge-wrap {{ text-align: right; }}
.survey-badge {{ display: inline-block; font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em; padding: 2px 8px;
  border: 1px solid currentColor; }}

/* Alerts */
.alert {{ background: #fff; border: 1px solid var(--border); border-radius: 4px; box-shadow: var(--shadow); padding: 14px 18px; }}
.alert .title {{ font-family: var(--serif) !important; font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
.alert .body {{ font-size: 14px; color: var(--text); line-height: 1.55; }}
.alert ul {{ margin: 0.2rem 0 0.3rem; padding-left: 1.2rem; }}
.alert li {{ font-size: 14px; color: var(--text); font-family: var(--mono) !important; line-height: 1.7; }}
.alert .note {{ font-size: 12.5px; color: var(--text2); font-style: italic; }}
.alert-danger {{ border-left: 4px solid var(--danger); }} .alert-danger .title {{ color: var(--danger); }}
.alert-success {{ border-left: 4px solid var(--success); }} .alert-success .title {{ color: var(--success); }}
.alert-error {{ border-left: 4px solid var(--danger); }} .alert-error .title {{ color: var(--danger); }}
.alert-info {{ border-left: 4px solid var(--accent); }} .alert-info .title {{ color: var(--accent); }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 0.3rem 2.2rem; margin: 2px 0 6px; }}
.metric .k {{ font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text2); }}
.metric .v {{ font-family: var(--serif) !important; font-size: 17px; font-weight: 600; color: var(--text); }}

/* Figure plates */
.plate {{ padding: 10px 12px 8px; margin-top: 0.3rem; }}
.plate img {{ width: 100%; height: auto; display: block; }}
/* Compact plate: capped at 420 px tall and never wider than its card, keeping the figure's aspect ratio */
.plate.plate-compact {{ max-width: 100%; min-width: 0; box-sizing: border-box; }}
.plate.plate-compact img {{ width: auto; height: auto; max-width: 100%; max-height: 420px; margin: 0 auto; }}
.plate-legend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1.5rem; font-size: 12px; color: var(--text2); padding: 2px 2px 8px;
  border-bottom: 1px solid #eee8da; margin-bottom: 8px; }}
.plate-legend .ref {{ font-family: var(--serif) !important; font-style: italic; color: var(--text); margin-right: auto; }}
.sw {{ display: inline-block; vertical-align: middle; margin-right: 6px; }}
.sw.block {{ width: 14px; height: 10px; border: 1px solid #8f8570; }}
.sw.tri {{ width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent;
  border-bottom: 10px solid {SURVEY_RED}; }}
.sw.dot {{ width: 9px; height: 9px; border-radius: 50%; background: {SURVEY_BLUE}; }}
.sw.dash {{ width: 22px; border-top: 2px dashed {SURVEY_RED}; }}

/* Mission log: typed field sheet with ruled lines */
.log-sheet {{
  height: 450px; overflow-y: auto; padding: 10px 16px; margin-top: 0.3rem;
  background-color: #fff; background-image: repeating-linear-gradient(to bottom, transparent 0 19px, #efe9dc 19px 20px);
  background-attachment: local; display: flex; flex-direction: column-reverse;
}}
.log-line {{ font-family: var(--mono) !important; font-size: 12px; line-height: 20px; color: var(--text);
  white-space: pre-wrap; word-break: break-word; }}
.log-line span {{ font-family: var(--mono) !important; }}
.log-time {{ color: #8a8577; }}
.log-tag {{ font-weight: 700; display: inline-block; min-width: 9ch; }}
</style>
"""

# Mission log message prefix -> (tag, tag colour); first match wins
LOG_TAGS = [
    ('SAFETY VERIFIER: FAIL', 'ERR', DANGER),
    ('MISSION ABORTED', 'ERR', DANGER),
    ('REJECTED', 'ERR', DANGER),
    ('SAFETY VERIFIER', 'VERIFY', WARNING),
    ('COMMAND', 'LLM', ACCENT),
    ('PARSED INTENT', 'LLM', ACCENT),
    ('LLM PARSER', 'LLM', ACCENT),
    ('A* ', 'A*', SUCCESS),
    ('HMM', 'HMM', PURPLE),
    ('MDP PILOT', 'MDP', MDP_COLOR),
    ('CONFLICT', 'CONFLICT', CONFLICT_COLOR),
]
LOG_LINE_PATTERN = re.compile(r'^\[(\d{2}:\d{2}:\d{2})\] (.*)$', re.DOTALL)


# =====================================================================
# SHARED RESOURCES & SESSION STATE
# =====================================================================
@st.cache_resource
def load_cave():
    cave = CaveEnvironment()
    return cave, ForkliftPlanner(cave)


cave, planner = load_cave()


@st.cache_resource
def get_db():
    return MissionDatabase()


db = get_db()


def init_session():
    ss = st.session_state
    if 'localiser' in ss:
        return
    ss.localiser = CaveHMMLocaliser(cave)  # Per session: the estimator holds mutable belief
    hmm_env = ss.localiser.hmm_env
    belief = np.zeros(hmm_env.num_states)
    belief[hmm_env.coord_to_state(START_POSITION[1] + HMM_BORDER, START_POSITION[0] + HMM_BORDER)] = 1.0
    ss.belief = belief
    ss.bots = create_fleet(INITIAL_BATTERY_MIN)
    ss.last_bot = HMM_BOT
    ss.last_result = None
    ss.mission_log = []
    for bot in ss.bots.values():
        log(f"System online. {bot.bot_id} deployed at {bot.position}, battery {bot.battery:.1f} min.")


def log(message: str):
    st.session_state.mission_log.append(f"[{datetime.now():%H:%M:%S}] {message}")


def synced_localiser() -> CaveHMMLocaliser:
    """The session localiser with its estimator loaded from the stored belief"""
    loc = st.session_state.localiser
    loc.estimator.belief_state = st.session_state.belief
    return loc


def map_estimate():
    """(x, y, probability) of the highest-belief cell"""
    x, y = synced_localiser().most_likely_position()
    return x, y, float(st.session_state.belief.max())


@contextmanager
def capture_parser_warnings():
    """Routes the parsers' fallback warnings into the mission log"""
    class Handler(logging.Handler):
        def emit(self, record):
            log(f"LLM PARSER: {record.getMessage()}")
    handler = Handler(level=logging.WARNING)
    loggers = (llm_parser.logger, react_parser.logger)
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeHandler(handler)


# =====================================================================
# MISSION PIPELINE
# =====================================================================
def run_command(text: str, api_key: str, bot_id: str = HMM_BOT):
    ss = st.session_state
    ss.pop('pending_clarification', None)  # A new command abandons any unanswered question
    ss.last_bot = bot_id
    log(f"COMMAND [{bot_id}]: {text!r}")

    if ss.get('use_react', True):
        intent = parse_intent(lambda: react_parse(text, api_key, clarify_callback=defer_clarification), bot_id)
    else:
        intent = parse_intent(lambda: keyword_only(text), bot_id)
    if intent:
        execute_intent(text, intent, bot_id)


def keyword_only(text: str):
    """The keyword parser alone, with a one-step trace, for when the ReACT parser is switched off"""
    intent = keyword_parse(text)
    return intent, [{"step": 1, "type": "KEYWORD", "reason": "ReACT parser off — keyword parser used",
                     "result": intent.action}]


def answer_clarification(answer: str, api_key: str):
    """Resumes the paused ReACT loop with the user's answer, then runs the command if it now parses"""
    ss = st.session_state
    state = ss.pop('pending_clarification')
    bot_id = state.extra['bot_id']
    ss.last_bot = bot_id
    log(f"ANSWER [{bot_id}]: {answer!r}")
    intent = parse_intent(lambda: resume_react(state, answer, api_key, clarify_callback=defer_clarification), bot_id)
    if intent:
        execute_intent(state.user_text, intent, bot_id)


def defer_clarification(question: str) -> str:
    """ReACT clarify_callback for Streamlit: the answer comes on a later rerun, so pause the loop"""
    raise AwaitingClarification(question)


def parse_intent(parse, bot_id: str) -> Optional[CaveCommand]:
    """Runs a parse step; returns the command, or None after posting a clarifying question or rejection"""
    ss = st.session_state
    try:
        with capture_parser_warnings():
            intent, trace = parse()
    except AwaitingClarification as pause:
        state = pause.state
        state.extra['bot_id'] = bot_id
        ss.pending_clarification = state
        ss.parse_trace = state.trace
        log(f"REACT CLARIFY [{bot_id}]: {state.question}")
        ss.last_result = ('clarify', state.question)
        return None
    except ValueError as exc:
        ss.parse_trace = []
        log(f"REJECTED: {exc}")
        ss.last_result = ('error', str(exc))
        return None
    ss.parse_trace = trace
    for step in trace:
        log(f"REACT STEP {step['step']} {step['type']}: "
            + ', '.join(f"{k}={v}" for k, v in step.items() if k not in ('step', 'type')))
    log(f"PARSED INTENT: {intent.model_dump_json()}")
    return intent


def execute_intent(text: str, intent: CaveCommand, bot_id: str):
    """Gates, plans, verifies and executes a parsed command for bot_id"""
    ss = st.session_state
    bot = ss.bots[bot_id]
    other = next(b for b in ss.bots.values() if b.bot_id != bot_id)

    if intent.action == 'hold':
        msg = f"{bot_id} holding position at {bot.position}, battery {bot.battery:.1f} min."
        log(f"HOLD: {msg}")
        ss.last_result = ('info', msg)
        return

    if intent.action == 'status':
        if bot_id == HMM_BOT:
            x, y, p = map_estimate()
            msg = (f"{bot_id} at {bot.position}, MAP estimate ({x}, {y}) with p={p:.2f}, "
                   f"battery {bot.battery:.1f} min.")
        else:
            msg = f"{bot_id} at {bot.position}, battery {bot.battery:.1f} min, payload {bot.payload or 'none'}."
        log(f"STATUS: {msg}")
        ss.last_result = ('info', msg)
        return

    chamber = intent.target_chamber
    if chamber is None and intent.action in ('move', 'deliver', 'replan'):
        msg = f"No valid target chamber (0-9) in the command; {bot_id} holds position."
        log(f"REJECTED: {msg}")
        ss.last_result = ('error', msg)
        return
    target = CHAMBER_MAP[chamber] if chamber is not None else bot.position  # Scan without target: scan here

    # Confidence gate: UNIT-01 only navigates when its HMM belief is concentrated enough
    if bot_id == HMM_BOT and intent.action in ('move', 'deliver', 'replan') and not is_confident(synced_localiser()):
        status = confidence_status(synced_localiser())
        msg = f"⚠️ {status['message']}. Issue SCAN command to localise first."
        log(f"BLOCKED_UNCERTAINTY [{bot_id}]: entropy {status['entropy']} >= threshold {status['threshold']}; "
            f"{bot_id} holds position.")
        ss.last_result = ('error', msg)
        ss.mdp_advice = None
        db.log_mission(text, mission_record(intent), [], (False, [f"BLOCKED_UNCERTAINTY: {status['message']}"]),
                       bot.battery, bot.battery, bot.position, "HMM gate", bot_id=bot_id)
        return

    payload = intent.payload
    payload_kg = PAYLOAD_WEIGHTS_KG.get(payload, DEFAULT_PAYLOAD_KG) if payload else 0.0

    # Plan a candidate route, then verify it before anything moves
    path, expanded, _ = planner.a_star(bot.position, target)
    log(f"A* CANDIDATE [{bot_id}]: {bot.position} -> {target} (chamber {chamber}): "
        f"{max(len(path) - 1, 0)} moves, {expanded} nodes expanded" if path else
        f"A* CANDIDATE [{bot_id}]: no path from {bot.position} to {target}")

    # Bid for any cells shared with the other bot's route in progress
    candidate = path
    path, other_final = deconflict(bot, other, path, intent.priority)
    if candidate and not path:
        violations = [f"CONFLICT HOLD: {bot_id} lost the bid for cells on {other.bot_id}'s route and has no detour"]
        ss.last_result = ('violations', violations)
        ss.mdp_advice = None
        log(f"MISSION ABORTED: {bot_id} holds position.")
        db.log_mission(text, mission_record(intent), candidate, (False, violations), bot.battery, bot.battery, bot.position, "A*",
                       bot_id=bot_id)
        return

    use_smt = ss.get('use_smt', True)
    check = verify_route(intent, path, payload_kg, bot)
    is_safe, violations = check['safe'], check['violations']
    route_source = "A*"
    if not is_safe:
        # Too narrow only because of the load: drive the MDP pilot's detour instead, if it is fully verified
        narrow_only = payload_kg > 0 and all(v.startswith("PASSAGE TOO NARROW") for v in violations)
        reroute = consult_mdp_pilot(intent, target, payload_kg, bot, other if narrow_only else None)
        if narrow_only and reroute:
            blocked_moves = len(path) - 1
            path, other_final, check = reroute
            is_safe, violations = check['safe'], check['violations']
            route_source = "MDP"
            log(f"REROUTED [{bot_id}]: {blocked_moves} -> {len(path) - 1} moves on the MDP pilot's detour around "
                f"passages too narrow for {payload}")
        else:
            ss.last_result = ('violations', violations)
            log(f"MISSION ABORTED: {bot_id} holds position.")
            verdict = (False, ["SMT_VIOLATION"] + violations) if use_smt else (False, violations)
            db.log_mission(text, mission_record(intent), path, verdict, bot.battery, bot.battery, bot.position, "A*",
                           bot_id=bot_id)
            return
    if use_smt:
        detour_note = " (MDP detour)" if route_source == "MDP" else ""
        st.success(f"✓ SMT VERIFIED — {check['checked_cells']} cells checked{detour_note}")
    if other_final != other.active_route:
        yield_route(other, other_final)
    battery_before = bot.battery

    # Execute
    log(f"{route_source} PATH [{bot_id}]: {' -> '.join(f'({x},{y})' for x, y in path)}")
    scan_status = None
    if bot_id == HMM_BOT:
        if intent.action == 'scan':
            scan_status = scan_localise(bot.position)
        else:
            simulate_hmm(path)
    moves = len(path) - 1
    bot.position = target
    bot.battery -= moves / 10
    bot.active_route = path
    bot.payload = payload
    bot.priority = intent.priority

    msg = (f"{bot_id} {intent.action.upper()} complete: at {target} after {moves} moves "
           f"(priority {intent.priority}).")
    if intent.action == 'scan':
        found = cave.cave_map[target].is_survivor_location
        msg += " SURVIVOR DETECTED." if found else " No survivor at this location."
    if scan_status:
        x, y, p = map_estimate()
        msg += (f" Localisation: entropy {scan_status['entropy']:.3f} (threshold {scan_status['threshold']}), "
                f"MAP ({x}, {y}) p={p:.2f} — {scan_status['message']}.")
    if payload:
        msg += f" Delivered {payload}."
    if route_source == "MDP":
        msg += " Rerouted around passages too narrow for the load (MDP pilot detour)."
    estimate = ''
    if bot_id == HMM_BOT:
        x, y, p = map_estimate()
        estimate = f" MAP estimate ({x}, {y}) p={p:.2f};"
    log(f"{msg}{estimate} battery {bot.battery:.1f} min.")
    ss.last_result = ('success', msg)
    db.log_mission(text, mission_record(intent), path, (is_safe, violations), battery_before, bot.battery, bot.position,
                   route_source, bot_id=bot_id)


def verify_route(intent: CaveCommand, path, payload_kg: float, bot: RescueBot) -> dict:
    """Checks a route with the verifier chosen in the sidebar (Z3 SMT or rule-based) and logs the verdict"""
    use_smt = st.session_state.get('use_smt', True)
    if use_smt:
        check = verify_command_smt(intent, path, cave, payload_kg, bot.battery)
    else:  # Rule-based check_mission, wrapped in the same result dict
        check = SafetyVerifier(cave, battery_remaining_min=bot.battery, use_smt=False).verify(intent, path, cave,
                                                                                            payload_kg)
    log(f"{'SMT VERIFIER' if use_smt else 'SAFETY VERIFIER'}: {'PASS' if check['safe'] else 'FAIL'} [{bot.bot_id}] "
        f"(payload {intent.payload or 'none'} = {payload_kg} kg, battery {bot.battery:.1f} min) "
        f"{check['proof']}, {check['checked_cells']} cells, {check['solver_result']}"
        + ("" if check['safe'] else f" -> {check['violations']}"))
    return check


def mission_record(intent: CaveCommand) -> dict:
    """The parsed-intent columns MissionDatabase.log_mission stores for a mission"""
    return {'action': intent.action, 'target_chamber': intent.target_chamber, 'payload': intent.payload,
            'priority': intent.priority}


def deconflict(bot: RescueBot, other: RescueBot, path, priority: str):
    """Runs the bidding protocol against the other bot's route in progress; returns (own route, other's route)"""
    if not path or len(other.active_route) < 2:
        return path, other.active_route
    first, second = sorted((bot, other), key=lambda b: b.bot_id)  # UNIT-01 is always bot 1 (wins ties)
    routes = {bot.bot_id: path, other.bot_id: other.active_route}
    priorities = {bot.bot_id: priority, other.bot_id: other.priority}
    final_1, final_2, resolution_log = resolve_conflicts(
        routes[first.bot_id], routes[second.bot_id], priorities[first.bot_id], first.battery,
        priorities[second.bot_id], second.battery, env=cave, bot1_name=first.bot_id, bot2_name=second.bot_id)
    for line in resolution_log:
        log(f"CONFLICT: {line}")
    finals = {first.bot_id: final_1, second.bot_id: final_2}
    return finals[bot.bot_id], finals[other.bot_id]


def yield_route(other: RescueBot, detour):
    """Applies a lost bid to the other bot's route in progress: take the detour if safe, else hold at its start"""
    old_route = other.active_route
    old_moves = len(old_route) - 1
    if detour:
        payload_kg = PAYLOAD_WEIGHTS_KG.get(other.payload, DEFAULT_PAYLOAD_KG) if other.payload else 0.0
        verifier = SafetyVerifier(cave, battery_remaining_min=other.battery + old_moves / 10)
        ok, problems = verifier.check_mission(payload_kg, detour[-1], detour)
        if ok:
            other.battery += (old_moves - (len(detour) - 1)) / 10
            other.active_route = detour
            log(f"CONFLICT: {other.bot_id} rerouted onto its detour ({len(detour) - 1} moves, was {old_moves}); "
                f"battery {other.battery:.1f} min")
            return
        log(f"CONFLICT: {other.bot_id} detour fails the safety check ({'; '.join(problems)}); holding instead")
    other.battery += old_moves / 10  # It never travelled the cancelled route
    other.position = old_route[0]
    other.active_route = []
    if other.bot_id == HMM_BOT:
        reset_belief(other.position)
    log(f"CONFLICT: {other.bot_id} holds at {other.position}; route cancelled, {old_moves / 10:.1f} min refunded")


def reset_belief(position):
    """UNIT-01 is known to be back at position (its route was cancelled)"""
    ss = st.session_state
    hmm_env = ss.localiser.hmm_env
    belief = np.zeros(hmm_env.num_states)
    belief[hmm_env.coord_to_state(position[1] + HMM_BORDER, position[0] + HMM_BORDER)] = 1.0
    ss.belief = belief


def simulate_hmm(path):
    """Runs a full predict/update filter step with the real sonar wall count at every cell along the route"""
    ss = st.session_state
    loc = ss.localiser
    loc.estimator.belief_state = ss.belief
    moves = list(zip(path, path[1:]))
    if not moves:
        log("HMM: no movement, belief unchanged.")
        return

    for i, ((x0, y0), (x1, y1)) in enumerate(moves, 1):
        action = STEP_TO_ACTION[(x1 - x0, y1 - y0)]
        observation = get_wall_observation(cave, (x1, y1))  # Real wall count at the bot's true cell
        belief = loc.step(action, observation)
        x, y = loc.most_likely_position()
        log(f"HMM STEP {i}/{len(moves)}: action={ACTION_NAMES[action]} sonar_walls={observation} "
            f"-> MAP ({x}, {y}) p={belief.max():.3f}")
    ss.belief = loc.estimator.belief_state.copy()


def scan_localise(position) -> dict:
    """Stationary sonar sweep: SCAN_SONAR_READINGS measurement updates at position; returns confidence_status"""
    ss = st.session_state
    loc = synced_localiser()
    observation = get_wall_observation(cave, position)
    for i in range(1, SCAN_SONAR_READINGS + 1):
        belief = loc.sense(observation)
        x, y = loc.most_likely_position()
        log(f"HMM SCAN {i}/{SCAN_SONAR_READINGS}: sonar_walls={observation} -> MAP ({x}, {y}) "
            f"p={belief.max():.3f}, entropy {loc.belief_entropy():.3f}")
    ss.belief = loc.estimator.belief_state.copy()
    return confidence_status(loc)


# =====================================================================
# AUTONOMOUS MISSION: bandit picks a chamber -> Q-learning drives there -> HMM updates -> bandit reward
# =====================================================================
def get_navigator() -> QLearningNavigator:
    """Per-session Q-learning navigator (it caches a Q-table per chamber). With the SMT verifier on, flooded
    cells are walls to it, since SMT rejects deep water anywhere on a route."""
    ss = st.session_state
    use_smt = ss.get('use_smt', True)
    if ss.get('navigator_smt') != use_smt:
        flooded = [cell for cell, info in cave.cave_map.items() if info.water_level >= SMT_WATER_LIMIT] if use_smt else []
        ss.navigator = QLearningNavigator(cave, blocked=flooded, seed=None)
        ss.navigator_smt = use_smt
    return ss.navigator


def phase_badge_html(phase: str, cycle: int) -> str:
    color = AUTO_PHASE_COLORS[phase]
    return (f'<div style="display:flex;align-items:center;gap:12px;margin:2px 0 4px">'
            f'<span style="background:{color};color:#fff;font-weight:700;letter-spacing:0.1em;font-size:12px;'
            f'padding:4px 12px;border-radius:3px">{phase}</span>'
            f'<span style="font-size:12px;color:var(--text2)">Cycle {cycle} of at most {AUTO_MAX_CYCLES}</span></div>')


def run_autonomous_mission(phase_slot, detail_slot, table_slot):
    """Searches for survivors with UNIT-01 (the HMM-tracked unit) until all are found or nothing reachable is left"""
    ss = st.session_state
    bot = ss.bots[HMM_BOT]
    other = next(b for b in ss.bots.values() if b.bot_id != HMM_BOT)
    ss.pop('pending_clarification', None)
    ss.last_bot = HMM_BOT
    ss.parse_trace = []
    rng = np.random.default_rng()
    navigator = get_navigator()
    bandit = ThompsonBandit(AUTO_ARMS, rng, hmm=HMMCoupling(synced_localiser, navigator, CHAMBER_MAP))
    survivors = set(cave.survivor_locations)
    found = set()
    log(f"AUTO: mission start — {HMM_BOT} at {bot.position}, battery {bot.battery:.1f} min, "
        f"{len(survivors)} survivors to find, sonar detection p={DETECTION_PROB}")

    def show(phase: str, cycle: int, detail: str):
        phase_slot.html(phase_badge_html(phase, cycle))
        detail_slot.markdown(detail)
        table_slot.dataframe(pd.DataFrame(bandit.table()), hide_index=True, width='stretch')
        ss.auto_phase = phase
        if phase != 'COMPLETE':
            time.sleep(AUTO_STEP_DELAY_S)

    def drive(route, intent: CaveCommand, other_final, cycle: int, label: str, check: dict) -> float:
        """Executes a verified route and runs the HMM over it; returns the battery before the move"""
        moves = len(route) - 1
        show('NAVIGATING', cycle, f"Q-learning route to {label}: **{moves} moves**, verified "
                                  f"({check['proof']}, {check['checked_cells']} cells).")
        if other_final != other.active_route:
            yield_route(other, other_final)
        battery_before = bot.battery
        log(f"Q-LEARNING PATH [{HMM_BOT}]: {' -> '.join(f'({x},{y})' for x, y in route)}")
        bot.position = route[-1]
        bot.battery -= moves * MINUTES_PER_MOVE
        bot.active_route = route
        bot.payload = None
        bot.priority = intent.priority

        # UPDATING BELIEF: one HMM filter step with the real wall count per move
        show('UPDATING BELIEF', cycle, f"HMM filter: {moves} predict/update steps along the route...")
        simulate_hmm(route)
        status = confidence_status(synced_localiser())
        x, y, p = map_estimate()
        show('UPDATING BELIEF', cycle, f"Belief updated over {moves} moves: MAP ({x}, {y}) p={p:.2f}, "
                                       f"entropy {status['entropy']:.3f} — {status['message']}.")
        return battery_before

    def return_to_base(cycle: int):
        """Drives back to the entrance on the learned route; logs and stays put if that route cannot be verified"""
        if bot.position == AUTO_HOME:
            log(f"AUTO [{cycle}]: {HMM_BOT} is already at base {AUTO_HOME}")
            return
        intent = CaveCommand(action='move', target_chamber=0, raw_text="[AUTO] return to base")
        route = navigator.route(bot.position, AUTO_HOME)
        route, other_final = deconflict(bot, other, route, intent.priority) if route else ([], other.active_route)
        check = verify_route(intent, route, 0.0, bot) if route else None
        if not check or not check['safe']:
            problem = '; '.join(check['violations']) if check else 'no verified route home'
            log(f"AUTO [{cycle}]: cannot return to base ({problem}) — {HMM_BOT} holds at {bot.position}")
            return
        battery_before = drive(route, intent, other_final, cycle, "base (C0 Entrance)", check)
        log(f"AUTO [{cycle}]: {HMM_BOT} back at base, battery {bot.battery:.1f} min")
        db.log_mission("[AUTO] return to base", mission_record(intent), route, (True, []), battery_before,
                       bot.battery, bot.position, "Q-learning", bot_id=HMM_BOT)

    cycle, end_reason = 0, None
    while end_reason is None:
        if found == survivors:
            end_reason = "all survivors found"
            break
        if cycle >= AUTO_MAX_CYCLES:
            log(f"AUTO [{cycle}]: cycle limit ({AUTO_MAX_CYCLES}) reached — returning to base")
            return_to_base(cycle)
            end_reason = f"cycle limit ({AUTO_MAX_CYCLES}) reached"
            break
        if bot.battery < MINUTES_PER_MOVE:  # Cannot make even one more move
            end_reason = "battery depleted"
            break
        active = bandit.active_arms()
        if active and all(bandit.mean(arm) < AUTO_EXHAUSTED_MEAN for arm in active):
            log(f"AUTO [{cycle}]: every remaining chamber's posterior mean is below {AUTO_EXHAUSTED_MEAN} — "
                f"returning to base")
            return_to_base(cycle)
            end_reason = "search exhausted — no promising chambers remain"
            break
        cycle += 1

        # SCANNING: localise if the belief is too spread out, then let the bandit pick a chamber to search
        if not is_confident(synced_localiser()):
            show('SCANNING', cycle, f"Position uncertain — stationary sonar sweep at {bot.position} to localise.")
            scan_localise(bot.position)
            if not is_confident(synced_localiser()):
                end_reason = "position still uncertain after a sonar sweep"
                break
        tried, route, arm, other_final, check, go_home = [], [], None, None, None, False
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
            log(f"AUTO [{cycle}]: bandit picked C{arm} (sample {sample:.2f} x reach {reach:.2f}, "
                f"mean {mean:.2f}, std {std:.2f})")
            route = navigator.route(bot.position, target)
            if not route:
                bandit.retire(arm, 'unreachable')
                log(f"AUTO [{cycle}]: C{arm} unreachable without entering flooded cells — removed from the search")
                continue
            # Battery reserve: going there must still leave enough to get back to the entrance
            home_route = navigator.route(target, AUTO_HOME)
            if not home_route:
                bandit.retire(arm, 'no safe return route')
                log(f"AUTO [{cycle}]: C{arm} has no safe route back to base — removed from the search")
                continue
            round_trip = (len(route) - 1) + (len(home_route) - 1)
            if round_trip * MINUTES_PER_MOVE >= bot.battery:
                log(f"AUTO [{cycle}]: C{arm} and back to base needs {round_trip * MINUTES_PER_MOVE:.1f} min, "
                    f"battery {bot.battery:.1f} min — returning to base")
                go_home = True
                break
            intent = CaveCommand(action='move', target_chamber=arm, raw_text=f"[AUTO] search chamber {arm}")
            route, other_final = deconflict(bot, other, route, intent.priority)
            if not route:
                tried.append(arm)
                log(f"AUTO [{cycle}]: C{arm} route lost the bid against {other.bot_id}'s route — trying another chamber")
                continue
            check = verify_route(intent, route, 0.0, bot)
            if not check['safe']:
                reason = 'out of battery range' if 'BATTERY INSUFFICIENT' in check['violations'] else 'unsafe route'
                bandit.retire(arm, reason)
                log(f"AUTO [{cycle}]: C{arm} {reason} — removed from the search")
                continue
            break
        if go_home:
            return_to_base(cycle)
            end_reason = "returning to base — insufficient battery for further search"
            break
        if arm is None:
            reasons = set(bandit.retired.values())
            end_reason = ("battery depleted" if 'out of battery range' in reasons else
                          "remaining chambers are blocked by UNIT-02's route" if tried else
                          "no reachable chambers left to search")
            break

        # NAVIGATING + UPDATING BELIEF
        battery_before = drive(route, intent, other_final, cycle, f"C{arm}", check)

        # REWARD UPDATE: sonar sweep of the chamber, reward 1 if a survivor is detected
        detected = scan_detects_survivor(cave, target, rng) and target not in found
        bandit.update(arm, int(detected))
        if detected:
            found.add(target)
            bandit.retire(arm, 'survivor rescued')
        outcome = (f"SURVIVOR FOUND at C{arm} ({len(found)}/{len(survivors)}) — reward 1, posterior raised"
                   if detected else f"C{arm} sweep found nobody — reward 0, posterior lowered")
        log(f"AUTO [{cycle}]: {outcome} (weight {bandit.last_confidence:.2f} from HMM position confidence); "
            f"C{arm} now Beta({bandit.alpha[arm]:.2f}, {bandit.beta[arm]:.2f}); "
            f"battery {bot.battery:.1f} min")
        show('REWARD UPDATE', cycle, f"{outcome}. Battery {bot.battery:.1f} min.")
        db.log_mission(f"[AUTO] search chamber {arm}", mission_record(intent), route, (True, []), battery_before,
                       bot.battery, bot.position, "Q-learning", bot_id=HMM_BOT)

    missed = sorted(survivors - found)
    summary = (f"{end_reason.capitalize()} after {cycle} cycles: {len(found)} of {len(survivors)} survivors found"
               + (f" (not found: {', '.join(f'({x},{y})' for x, y in missed)})" if missed else "")
               + f". {HMM_BOT} at {bot.position}, battery {bot.battery:.1f} min.")
    log(f"AUTO: mission complete — {summary}")
    ss.last_result = ('auto', summary)
    ss.auto_report = {'summary': summary, 'table': bandit.table()}
    show('COMPLETE', cycle, summary)


# =====================================================================
# MDP PILOT
# =====================================================================
MDP_ALGORITHMS = {'Policy Iteration': 'policy_iteration', 'Value Iteration': 'value_iteration'}


def get_mdp_pilot() -> CaveMDPPilot:
    """Per-session pilot, since it keeps the last solution"""
    if 'mdp_pilot' not in st.session_state:
        st.session_state.mdp_pilot = CaveMDPPilot(cave)
    return st.session_state.mdp_pilot


def store_mdp_view(label: str, algorithm: str, kind: str, policy, bot: RescueBot, **extra):
    """Snapshot of the latest MDP solve for the MDP Pilot tab"""
    ss = st.session_state
    pilot = get_mdp_pilot()
    solution = pilot.last_solution
    origin = bot.position
    route = pilot.trace_route(origin, policy)
    if origin in solution.targets:
        status = 'at survivor'
    elif len(route) > 1 and route[-1] in solution.targets:
        status = 'route'
    else:
        status = 'no route'  # Boxed in or every target blocked: any "move" would just bump a wall
    ss.mdp_view = {'label': label, 'algorithm': algorithm, 'kind': kind, 'solution': solution, 'bot_id': bot.bot_id,
                   'origin': origin, 'status': status, 'route': route if status == 'route' else [origin],
                   'action': pilot.get_action(origin, policy) if status == 'route' else None,
                   'flooded': [], 'before': None, 'stamp': time.time_ns(), **extra}
    return ss.mdp_view


def consult_mdp_pilot(intent: CaveCommand, target, payload_kg: float, bot: RescueBot,
                      other: Optional[RescueBot] = None):
    """After a safety block: replan around passages too narrow for this load and log the pilot's advice.

    With other given, the alternative is also bid against other's route in progress, so it can be executed.
    Returns (route, other's final route, verifier result) when the alternative passes the active verifier, else None.
    """
    ss = st.session_state
    log("MDP PILOT: Safety blocked — consulting MDP pilot for alternative route...")
    diameter = LOADED_BOT_DIAMETER_M if payload_kg > 0 else EMPTY_BOT_DIAMETER_M
    blocked = [cell for cell, info in cave.cave_map.items()
               if info.passage_width <= diameter and cell != bot.position]
    narrow_count = len(blocked)
    if ss.get('use_smt', True):  # The SMT verifier also rejects deep water anywhere on the route: route around it
        blocked += [cell for cell, info in cave.cave_map.items()
                    if info.water_level >= SMT_WATER_LIMIT and cell != bot.position and cell not in blocked]
    flooded_count = len(blocked) - narrow_count
    pilot = get_mdp_pilot()
    policy = pilot.replan(blocked, target)
    view = store_mdp_view('Policy Iteration', 'policy_iteration', 'safety', policy, bot)
    action, route = view['action'], view['route']
    engine = f"({pilot.last_solution.engine_name}, {pilot.last_solution.iterations} iterations)"

    if view['status'] == 'at survivor':
        ss.mdp_advice = f"{bot.bot_id} is already at {target}"
    elif view['status'] == 'no route':
        flooded = f" and {flooded_count} flooded cells" if flooded_count else ""
        ss.mdp_advice = f"HOLD — no route to {target} avoids the {narrow_count} passages too narrow for this load{flooded}"
    else:
        other_final = other.active_route if other else None
        if other is not None:
            route, other_final = deconflict(bot, other, route, intent.priority)
        if not route:
            ss.mdp_advice = (f"HOLD — the alternative route loses the bid for cells on {other.bot_id}'s route"
                             if other else "HOLD — no alternative route")
        else:
            check = verify_route(intent, route, payload_kg, bot)
            avoiding = (f" avoiding {narrow_count} narrow passages" if narrow_count else "") + (
                f" and {flooded_count} flooded cells" if flooded_count else "")
            ss.mdp_advice = (f"first move {action}; alternative route of {len(route) - 1} moves{avoiding} — "
                             f"safety verifier {'PASS' if check['safe'] else 'FAIL: ' + '; '.join(check['violations'])}")
            log(f"MDP PILOT: recommended {ss.mdp_advice} {engine}")
            return (route, other_final, check) if check['safe'] else None
    log(f"MDP PILOT: recommended {ss.mdp_advice} {engine}")
    return None


def run_mdp_compute(label: str, bot_id: str):
    ss = st.session_state
    bot = ss.bots[bot_id]
    algorithm = MDP_ALGORITHMS[label]
    policy = get_mdp_pilot().compute_policy(algorithm)
    view = store_mdp_view(label, algorithm, 'computed', policy, bot)
    sol = view['solution']
    log(f"MDP PILOT: computed policy with {sol.engine_name} ({sol.iterations} iterations); "
        f"recommended action for {bot_id} at {bot.position}: {view['action'] or 'HOLD (' + view['status'] + ')'}")


def run_mdp_replan(label: str, bot_id: str):
    """Pretends the next cell on the bot's recommended route has flooded, then replans around it"""
    ss = st.session_state
    bot = ss.bots[bot_id]
    algorithm = MDP_ALGORITHMS[label]
    view = ss.get('mdp_view')
    if (not view or view['kind'] == 'safety' or view['algorithm'] != algorithm or view['bot_id'] != bot_id
            or view['origin'] != bot.position):
        run_mdp_compute(label, bot_id)  # Start from a fresh policy for this bot, position and algorithm
        view = ss.mdp_view
    if view['status'] != 'route':
        ss.mdp_notice = (f"{bot_id} is at a survivor location, so there is no route ahead to flood."
                         if view['status'] == 'at survivor' else
                         "No route to any survivor remains, so there is nothing further to flood. "
                         "Press Compute Policy to start over.")
        return
    flooded = view['flooded'] + [view['route'][1]]
    pilot = get_mdp_pilot()
    policy = pilot.replan(flooded, algorithm=algorithm)
    new_view = store_mdp_view(label, algorithm, 'replan', policy, bot, flooded=flooded, before=view['action'])
    log(f"MDP PILOT: simulated flooding at {flooded[-1]}; replanned with {len(flooded)} blocked cell(s); "
        f"action for {bot_id} at {bot.position}: {view['action']} -> {new_view['action'] or 'HOLD (no route)'}")


# =====================================================================
# PLOTS
# =====================================================================
FIG_RC = {'font.family': 'sans-serif', 'font.serif': FIG_SERIF, 'font.sans-serif': FIG_SANS, 'svg.fonttype': 'path'}
# The browser shows SVG at 96 px per inch: this keeps the survey map near its 420 px display cap, so its text stays legible
SURVEY_MAP_FIGSIZE = (6.8, 4.4)
PAPER_BOX = {'facecolor': PAPER, 'edgecolor': BORDER, 'linewidth': 0.6, 'boxstyle': 'square,pad=0.3', 'alpha': 0.92}


def survey_axes(title, figsize=(16, 10)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PAPER)
    ax.set_facecolor(DRY_CELL)
    ax.set_title(title, color=TEXT, fontsize=11, loc='left', family='serif', fontweight='bold', pad=10)
    ax.set_xticks(range(cave.width))
    ax.set_yticks(range(cave.height))
    ax.set_xticks(np.arange(-0.5, cave.width), minor=True)
    ax.set_yticks(np.arange(-0.5, cave.height), minor=True)
    ax.grid(which='minor', color=BORDER, linewidth=0.4)  # Graph-paper ruling
    ax.grid(which='major', visible=False)
    ax.tick_params(which='both', length=0, colors=TEXT_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#8f8570')
        spine.set_linewidth(0.8)
    return fig, ax


def draw_rock(ax):
    for y in range(cave.height):
        for x in range(cave.width):
            if cave.grid[y][x] == 1:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=ROCK, edgecolor='none', zorder=1))


def draw_map_furniture(ax):
    """North arrow (top right) and scale note (bottom left), as on a printed survey sheet"""
    ax.add_patch(Rectangle((0.938, 0.842), 0.052, 0.148, transform=ax.transAxes, facecolor=PAPER,
                           edgecolor=BORDER, linewidth=0.6, alpha=0.92, zorder=8))
    ax.annotate('', xy=(0.964, 0.925), xytext=(0.964, 0.855), xycoords='axes fraction',
                arrowprops={'arrowstyle': '-|>,head_width=0.35,head_length=0.8', 'color': TEXT, 'linewidth': 1.4},
                zorder=9)
    ax.text(0.964, 0.935, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=11,
            fontweight='bold', family='serif', color=TEXT, zorder=9)
    ax.text(0.008, 0.012, f'1 cell = {METRES_PER_CELL} m', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, family='serif', color=TEXT, bbox=PAPER_BOX, zorder=9)


def survey_colorbar(fig, ax, mappable, label, ticks, tick_labels=None):
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.022, pad=0.012, aspect=38, ticks=ticks)
    cbar.set_label(label, color=TEXT, fontsize=9, family='serif')
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=2, width=0.6)
    if tick_labels:
        cbar.set_ticklabels(tick_labels)
    cbar.outline.set_edgecolor('#8f8570')
    cbar.outline.set_linewidth(0.6)


def draw_cave_map():
    ss = st.session_state
    water = np.full((cave.height, cave.width), np.nan)
    for (x, y), cell in cave.cave_map.items():
        water[y, x] = cell.water_level

    fig, ax = survey_axes("Hydrological Survey — Water Intrusion Overlay", figsize=SURVEY_MAP_FIGSIZE)
    im = ax.imshow(np.ma.masked_invalid(water), cmap=WATER_CMAP, vmin=0, vmax=1, origin='upper',
                   interpolation='nearest', zorder=0)
    draw_rock(ax)
    survey_colorbar(fig, ax, im, 'Water Depth (relative)', ticks=[0.0, 0.5, 1.0])

    unit1, unit2 = ss.bots['UNIT-01'], ss.bots['UNIT-02']
    for bot, color in ((unit1, SURVEY_RED), (unit2, UNIT2_COLOR)):
        if len(bot.active_route) > 1:
            xs, ys = zip(*bot.active_route)
            ax.plot(xs, ys, color=color, linewidth=1.2, dashes=(4, 3), zorder=4)
    for i, (x, y) in enumerate(cave.survivor_locations, 1):
        ax.scatter([x], [y], s=100, color=SURVEY_RED, marker='^', edgecolors='white', linewidths=0.6, zorder=5)
        ax.text(x + 0.3, y, f"S{i}", ha='left', va='center', fontsize=10, fontweight='bold', family='serif',
                color=SURVEY_RED, bbox=PAPER_BOX, zorder=6)  # Paper backing keeps labels legible over rock
    mx, my, _ = map_estimate()
    ax.scatter([mx], [my], s=80, color=SURVEY_BLUE, marker='o', edgecolors='white', linewidths=0.8, zorder=7)
    label_right = mx < 2  # Stay inside the axes near the left edge
    ax.text(mx + 0.28 if label_right else mx - 0.28, my, "UNIT-01", ha='left' if label_right else 'right',
            va='center', fontsize=8.5, fontweight='bold', color=SURVEY_BLUE, bbox=PAPER_BOX, zorder=8)
    ux, uy = unit2.position
    ax.scatter([ux], [uy], s=75, color=UNIT2_COLOR, marker='s', edgecolors='white', linewidths=0.8, zorder=7)
    ax.text(ux, uy + 0.36, "UNIT-02", ha='center', va='top', fontsize=8.5, fontweight='bold', color=UNIT2_COLOR,
            bbox=PAPER_BOX, zorder=8)  # Below the marker, so it never collides with UNIT-01's side label
    draw_map_furniture(ax)
    fig.tight_layout()
    return fig


def draw_belief_map():
    ss = st.session_state
    h, w = ss.localiser.hmm_env.height, ss.localiser.hmm_env.width
    belief = ss.belief.reshape(h, w)[HMM_BORDER:h - HMM_BORDER, HMM_BORDER:w - HMM_BORDER].astype(float)
    belief[np.array(cave.grid) == 1] = np.nan

    mx, my, p = map_estimate()
    fig, ax = survey_axes("Posterior Belief Distribution — P(position | sensor readings)")
    vmax = max(p, 1e-9)
    im = ax.imshow(np.ma.masked_invalid(belief), cmap=BELIEF_CMAP, vmin=0, vmax=vmax, origin='upper',
                   interpolation='nearest', zorder=0)
    draw_rock(ax)
    ticks = [0.0, vmax / 2, vmax]
    survey_colorbar(fig, ax, im, 'P(position)', ticks=ticks, tick_labels=[f'{t:.0%}' for t in ticks])

    ax.scatter([mx], [my], s=120, color=SURVEY_RED, marker='^', edgecolors='white', linewidths=0.8, zorder=7)
    label_left = mx > cave.width - 6
    ax.text(mx - 0.35 if label_left else mx + 0.35, my, "ESTIMATED POSITION", ha='right' if label_left else 'left',
            va='center', fontsize=8, fontweight='bold', color=TEXT, bbox=PAPER_BOX, zorder=8)
    draw_map_furniture(ax)
    fig.tight_layout()
    return fig


def draw_mdp_map():
    ss = st.session_state
    view = ss.mdp_view
    sol = view['solution']
    values = np.full((cave.height, cave.width), np.nan)
    for (x, y), v in sol.values.items():
        values[y, x] = v

    fig, ax = survey_axes(f"MDP Pilot — Value Function V(s) · {sol.engine_name}")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=VALUE_CMAP, origin='upper', interpolation='nearest', zorder=0)
    draw_rock(ax)
    survey_colorbar(fig, ax, im, 'V(s)  (expected return)', ticks=None)

    for x, y in sol.blocked:
        flooded = (x, y) in view['flooded']
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='#f3dcd4' if flooded else '#efe7d6',
                               edgecolor=DANGER if flooded else '#9a8f78', hatch='xxx' if flooded else '///',
                               linewidth=0.8, zorder=2))

    # Policy arrows, centred on each cell (north is up because the y axis is inverted)
    arrow_moves = {'NORTH': (0, -1), 'SOUTH': (0, 1), 'WEST': (-1, 0), 'EAST': (1, 0)}
    cells = list(sol.policy)
    u = np.array([arrow_moves[sol.policy[c]][0] for c in cells]) * 0.5
    v = np.array([arrow_moves[sol.policy[c]][1] for c in cells]) * 0.5
    xs = np.array([c[0] for c in cells]) - u / 2
    ys = np.array([c[1] for c in cells]) - v / 2
    ax.quiver(xs, ys, u, v, angles='xy', scale_units='xy', scale=1, width=0.0016, headwidth=4.5,
              headlength=4.5, color='#3d372d', zorder=3)

    if len(view['route']) > 1:
        rx, ry = zip(*view['route'])
        ax.plot(rx, ry, color=SURVEY_BLUE, linewidth=1.6, dashes=(1.5, 1.5), zorder=4)
    for i, (x, y) in enumerate(cave.survivor_locations, 1):
        ax.scatter([x], [y], s=100, color=SURVEY_RED, marker='^', edgecolors='white', linewidths=0.6, zorder=5)
        ax.text(x + 0.3, y, f"S{i}", ha='left', va='center', fontsize=10, fontweight='bold', family='serif',
                color=SURVEY_RED, bbox=PAPER_BOX, zorder=6)
    bx, by = view['origin']
    ax.scatter([bx], [by], s=80, color=SURVEY_BLUE, marker='o', edgecolors='white', linewidths=0.8, zorder=7)
    label_right = bx < 2
    ax.text(bx + 0.28 if label_right else bx - 0.28, by, view['bot_id'], ha='left' if label_right else 'right',
            va='center', fontsize=8.5, fontweight='bold', color=SURVEY_BLUE, bbox=PAPER_BOX, zorder=7)
    draw_map_furniture(ax)
    fig.tight_layout()
    return fig


def figure_img(draw_fn, alt: str) -> str:
    """Draws a figure and returns it as a crisp, resolution-independent SVG <img>"""
    with plt.rc_context(FIG_RC):
        fig = draw_fn()
        buf = io.StringIO()
        fig.savefig(buf, format='svg', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.12, dpi=144)
    plt.close(fig)
    svg = buf.getvalue()
    encoded = base64.b64encode(svg[svg.index('<svg'):].encode('utf-8')).decode('ascii')
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="{html.escape(alt)}">'


def plate(draw_fn, legend_html: str, alt: str, cache_key=None, css_class: str = ''):
    """Renders a figure card; with cache_key, reuses the last SVG while the key is unchanged (~1 s per render)"""
    cache = st.session_state.setdefault('plate_cache', {})
    cached = cache.get(draw_fn.__name__)
    if cache_key is not None and cached and cached[0] == cache_key:
        img = cached[1]
    else:
        img = figure_img(draw_fn, alt)
        cache[draw_fn.__name__] = (cache_key, img)
    st.html(f'<div class="card plate {css_class}"><div class="plate-legend">{legend_html}</div>{img}</div>')


# =====================================================================
# UI COMPONENTS
# =====================================================================
def blend(color_a: str, color_b: str, t: float) -> str:
    """Linear blend between two hex colours, t=0 -> a, t=1 -> b"""
    a = [int(color_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(color_b[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(ca + (cb - ca) * t):02x}' for ca, cb in zip(a, b))


def battery_color(fraction: float) -> str:
    """Earthy green at full, amber at half, red when empty"""
    if fraction >= 0.5:
        return blend(WARNING, SUCCESS, (fraction - 0.5) / 0.5)
    return blend(DANGER, WARNING, fraction / 0.5)


def water_hex(level: float) -> str:
    r, g, b, _ = WATER_CMAP(level)
    return f'#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}'


def bot_status(bot_id: str):
    ss = st.session_state
    if ss.last_result and ss.last_result[0] == 'violations' and ss.last_bot == bot_id:
        return 'HOLDING', DANGER
    if ss.bots[bot_id].battery / INITIAL_BATTERY_MIN < LOW_BATTERY_FRACTION:
        return 'LOW POWER', WARNING
    return 'OPERATIONAL', SUCCESS


def unit_card_html(bot: RescueBot) -> str:
    status, status_color = bot_status(bot.bot_id)
    fraction = min(max(bot.battery / INITIAL_BATTERY_MIN, 0.0), 1.0)
    x, y = bot.position
    if bot.bot_id == HMM_BOT:
        mx, my, p = map_estimate()
        estimate_row = f'<tr><td class="k">HMM Est.</td><td class="v coord">({mx}, {my}) · p {p:.2f}</td></tr>'
    else:
        estimate_row = '<tr><td class="k">Localisation</td><td class="v">Direct report</td></tr>'
    color = SURVEY_BLUE if bot.bot_id == HMM_BOT else UNIT2_COLOR
    return f"""
    <div class="card" style="padding:6px 12px;border-left:3px solid {color}">
      <table class="form-table">
        <tr><td class="k">Bot ID</td><td class="v" style="color:{color}">{bot.bot_id}</td></tr>
        <tr><td class="k">Position</td><td class="v coord">({x}, {y})</td></tr>
        {estimate_row}
        <tr><td class="k">Payload</td><td class="v">{bot.payload or 'none'}</td></tr>
        <tr><td class="k">Battery</td><td class="v">{fraction * 100:.0f}%
          <span style="color:{TEXT_SECONDARY};font-weight:400"> · {bot.battery:.1f} min</span>
          <div class="battery-track"><div class="battery-fill"
            style="width:{fraction * 100:.1f}%;background:{battery_color(fraction)}"></div></div></td></tr>
        <tr><td class="k">Status</td><td class="v" style="color:{status_color}">
          <span class="status-dot" style="background:{status_color}"></span>{status}</td></tr>
      </table>
    </div>
    """


def sidebar_status_html() -> str:
    cards = '<div style="height:8px"></div>'.join(unit_card_html(bot) for bot in st.session_state.bots.values())
    return f"""
    <div class="brand-title">Cave Rescue</div>
    <div class="brand-sub">Field Operations — Sector Alpha</div>
    <hr class="rule">
    <div class="section-head">Unit Status</div>
    {cards}
    """


def location_index_html() -> str:
    survivors = set(cave.survivor_locations)
    rows = ''.join(
        f'<div><span class="sq" style="background:{water_hex(cave.cave_map[(x, y)].water_level)}"></span>'
        f'<span class="id">C{n}</span>{CHAMBER_NAMES[n]} <span class="xy">({x}, {y})</span>'
        f'{"<span class=svr>▲ survivor</span>" if (x, y) in survivors else ""}</div>'
        for n, (x, y) in CHAMBER_MAP.items())
    return (f'<hr class="rule"><div class="section-head">Location Index</div>'
            f'<div class="legend-list">{rows}</div>'
            f'<div class="legend-note">Square shade = water depth at the chamber.</div>')


def header_html() -> str:
    flagged = [(bot_id, *bot_status(bot_id)) for bot_id in BOT_IDS if bot_status(bot_id)[0] != 'OPERATIONAL']
    badge, color = (f"{flagged[0][0]} {flagged[0][1]}", flagged[0][2]) if flagged else ('LIVE', SUCCESS)
    return f"""
    <div class="survey-header">
      <div class="survey-title">Subsurface Survey — Cave System Alpha</div>
      <div class="survey-meta">Grid {cave.width}×{cave.height} · {len(cave.survivor_locations)} Survivors Located ·
        Updated {datetime.now():%H:%M:%S}</div>
      <div class="survey-badge-wrap"><span class="survey-badge" style="color:{color}">{badge}</span></div>
    </div>
    """


def result_html() -> str:
    ss = st.session_state
    kind, content = ss.last_result
    if kind == 'violations':
        items = ''.join(f'<li>{html.escape(v)}</li>' for v in content)
        advice = ss.get('mdp_advice')
        advice_html = (f'<div class="body" style="margin-top:6px"><b>MDP pilot:</b> {html.escape(advice)}. '
                       f'See the MDP Pilot tab.</div>' if advice else '')
        return (f'<div class="alert alert-danger"><div class="title">Mission Suspended — Safety Constraint Violated</div>'
                f'<ul>{items}</ul><div class="note">Unit held position; no movement executed.</div>{advice_html}</div>')
    if kind == 'success':
        bot = ss.bots[ss.last_bot]
        moves = max(len(bot.active_route) - 1, 0)
        metrics = ''.join(f'<div class="metric"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in (
            ('Route length', f'{moves} cells · {moves * METRES_PER_CELL} m'),
            ('Est. transit', f'{moves * MINUTES_PER_MOVE:.1f} min'),
            ('Unit', ss.last_bot),
            ('Battery remaining', f'{bot.battery:.1f} min')))
        return (f'<div class="alert alert-success"><div class="title">Route Confirmed — Mission Authorized</div>'
                f'<div class="metrics">{metrics}</div><div class="body">{html.escape(content)}</div></div>')
    css_class, title = {'error': ('alert-error', 'Command Rejected'), 'info': ('alert-info', 'Status Report'),
                        'clarify': ('alert-info', 'Clarification Needed'),
                        'auto': ('alert-info', 'Autonomous Mission')}[kind]
    return (f'<div class="alert {css_class}"><div class="title">{title}</div>'
            f'<div class="body">{html.escape(content)}</div></div>')


def mission_log_html() -> str:
    lines = []
    for entry in st.session_state.mission_log:
        match = LOG_LINE_PATTERN.match(entry)
        timestamp, message = match.groups() if match else ('--:--:--', entry)
        tag, color = next(((t, c) for prefix, t, c in LOG_TAGS if message.startswith(prefix)), ('SYS', TEXT_SECONDARY))
        lines.append(f'<div class="log-line"><span class="log-time">{timestamp}</span>  '
                     f'<span class="log-tag" style="color:{color}">[{tag}]</span> {html.escape(message)}</div>')
    # Single line with no blank lines, so st.markdown passes it through as one raw HTML block
    return f'<div class="card log-sheet"><div>{"".join(lines)}</div></div>'


MAP_LEGEND = (
    '<span class="ref">Plate 1 — Hydrological survey</span>'
    f'<span><span class="sw block" style="background:{ROCK}"></span>Rock</span>'
    f'<span><span class="sw block" style="background:linear-gradient(90deg,{DRY_CELL},{WATER_LOW},{WATER_HIGH})">'
    '</span>Water depth</span>'
    '<span><span class="sw tri"></span>Survivor (S1–S3)</span>'
    '<span><span class="sw dot"></span>UNIT-01 (HMM estimate)</span>'
    '<span><span class="sw dash"></span>UNIT-01 route</span>'
    f'<span><span class="sw block" style="background:{UNIT2_COLOR};border-color:{UNIT2_COLOR}"></span>UNIT-02</span>'
    f'<span><span class="sw dash" style="border-top-color:{UNIT2_COLOR}"></span>UNIT-02 route</span>'
)
BELIEF_LEGEND = (
    '<span class="ref">Plate 2 — Posterior belief</span>'
    f'<span><span class="sw block" style="background:{ROCK}"></span>Rock</span>'
    f'<span><span class="sw block" style="background:linear-gradient(90deg,{DRY_CELL},#b8d4e8,{SURVEY_BLUE})">'
    '</span>P(position)</span>'
    '<span><span class="sw tri"></span>Estimated position</span>'
)
MDP_LEGEND = (
    '<span class="ref">Plate 3 — MDP value function &amp; policy</span>'
    f'<span><span class="sw block" style="background:{ROCK}"></span>Rock</span>'
    f'<span><span class="sw block" style="background:linear-gradient(90deg,{DRY_CELL},#c6d8b0,#7fa86b)">'
    '</span>V(s)</span>'
    '<span>→ Policy</span>'
    f'<span><span class="sw block" style="background:#f3dcd4;border-color:{DANGER}"></span>Blocked / flooded</span>'
    '<span><span class="sw tri"></span>Survivor</span>'
    f'<span><span class="sw dash" style="border-top-color:{SURVEY_BLUE};border-top-style:dotted"></span>'
    'Recommended route</span>'
)


def mdp_table_html() -> str:
    view = st.session_state.mdp_view
    sol = view['solution']
    origin = view['origin']
    route = view['route']
    source = {'computed': 'Computed on request', 'replan': 'Adaptive replan (simulated flooding)',
              'safety': 'Safety-block fallback (narrow passages blocked)'}[view['kind']]
    rows = [
        ('Solver', f'{sol.engine_name} · {sol.iterations} iterations'),
        ('Source', source),
        (f"{view['bot_id']} position", f'<span class="mono">{origin}</span>'),
        ('V(bot)', f'{sol.values[origin]:.1f}' if origin in sol.values else '—'),
        ('Blocked cells', ', '.join(str(c) for c in view['flooded']) if view['flooded'] else f'{len(sol.blocked)}'),
        ('Route to survivor', f'{len(route) - 1} moves → {route[-1]}' if view['status'] == 'route' else 'none'),
    ]
    if view['before'] is not None:
        rows.append(('Action before replan', view['before']))
    body = ''.join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in rows)
    action = view['action'] or {'at survivor': 'HOLD — at survivor location',
                                'no route': 'HOLD — no route to any survivor'}[view['status']]
    return (f'<div class="card" style="padding:10px 16px">'
            f'<div class="alert-title-line" style="font-family:var(--serif);font-size:15px;font-weight:700;'
            f'margin-bottom:6px">Recommended action at current bot position: '
            f'<span style="color:{MDP_COLOR}">{action}</span></div>'
            f'<table class="form-table">{body}</table></div>')


# =====================================================================
# LAYOUT
# =====================================================================
st.markdown(APP_CSS, unsafe_allow_html=True)
init_session()
ss = st.session_state

with st.sidebar:
    status_slot = st.empty()  # Filled after command handling so it shows post-mission state
    st.html('<div class="section-head">Mission Parameters</div>')
    selected_bot = st.radio("Select Bot", BOT_IDS, horizontal=True, key='selected_bot',
                            help="Rescue commands and the MDP Pilot tab apply to this unit.")
    api_key = st.text_input("OpenAI API key", type="password",
                            help="Used only for this session. Without it, commands use the keyword parser.")
    st.checkbox("Use SMT Verifier", value=True, key='use_smt',
                help="Prove each route with the Z3 SMT solver. Unchecked: the rule-based SafetyVerifier.")
    st.toggle("Use ReACT Parser", value=True, key='use_react',
              help="Parse with an LLM Reason-Act-Observe loop that can ask a clarifying question. "
                   "Off: keyword parser only.")
    with st.form("command_form", clear_on_submit=True):
        command = st.text_input("Rescue command", placeholder="e.g. Oxygen kit to chamber 3")
        submitted = st.form_submit_button("Submit command", use_container_width=True)

    if submitted and command.strip():
        with st.spinner("Parsing, verifying and planning..."):
            run_command(command.strip(), api_key, selected_bot)

    if ss.get('pending_clarification'):
        with st.form("clarify_form", clear_on_submit=True):
            st.info(f"🤔 {ss.pending_clarification.question}")
            answer = st.text_input("Your answer", key='clarify_answer')
            answered = st.form_submit_button("Send answer", use_container_width=True)
        if answered and answer.strip():
            with st.spinner("Resuming parse..."):
                answer_clarification(answer.strip(), api_key)
            if ss.get('pending_clarification'):  # A follow-up question: redraw the form with it
                st.rerun()

    st.html('<div class="section-head">Autonomous Mission</div>')
    run_auto = st.button("Run Autonomous Mission", use_container_width=True, type='primary',
                         help="UNIT-01 searches on its own: a Thompson-sampling bandit picks chambers, Q-learning "
                              "drives there, the HMM tracks its position and each sonar sweep updates the bandit.")

    if st.button("Reset mission", use_container_width=True):
        for key in list(ss.keys()):
            del ss[key]
        st.rerun()
    st.html(location_index_html())

    with status_slot.container():
        st.html(sidebar_status_html())

st.html(header_html())
if ss.last_result:
    st.html(result_html())
if ss.get('parse_trace'):
    with st.expander("🔍 Parse trace"):
        for step in ss.parse_trace:
            st.markdown(f"**Step {step['step']} — {step['type']}**")
            details = '\n'.join(f"{key}: {value}" for key, value in step.items() if key not in ('step', 'type'))
            if details:
                st.code(details, language=None, wrap_lines=True)

if run_auto:
    with st.container(border=True):
        st.html('<div class="section-head">Autonomous Mission — live</div>')
        auto_phase, auto_detail, auto_table = st.empty(), st.empty(), st.empty()
        run_autonomous_mission(auto_phase, auto_detail, auto_table)
    st.rerun()  # Redraw the map, status cards and log with the mission's final state
if ss.get('auto_report'):
    with st.expander("🤖 Last autonomous mission — bandit posteriors"):
        st.markdown(ss.auto_report['summary'])
        st.dataframe(pd.DataFrame(ss.auto_report['table']), hide_index=True, width='stretch')

tab_map, tab_loc, tab_log, tab_mdp = st.tabs(["Cave Map", "Bot Localisation", "Mission Log", "MDP Pilot"])
with tab_map:
    plate(draw_cave_map, MAP_LEGEND, "Survey map with water depth, rock, survivors, A* route and estimated unit position",
          css_class='plate-compact',
          cache_key=tuple((tuple(b.active_route), b.position) for b in ss.bots.values()) + (map_estimate()[:2],))
with tab_loc:
    st.html('<div class="section-head">Localisation — UNIT-01</div>')
    gate = confidence_status(synced_localiser())
    map_x, map_y, map_p = map_estimate()
    col_entropy, col_threshold, col_map, col_badge = st.columns(4, vertical_alignment='center')
    col_entropy.metric("Belief entropy", f"{gate['entropy']:.3f}", help="Shannon entropy of the HMM belief (nats)")
    col_threshold.metric("Threshold", f"{gate['threshold']}")
    col_map.metric("MAP estimate", f"({map_x}, {map_y})", f"p = {map_p:.2f}", delta_color='off')
    with col_badge:
        if gate['confident']:
            st.success("CONFIDENT")
        else:
            st.error("UNCERTAIN")
    plate(draw_belief_map, BELIEF_LEGEND, "Posterior belief map of the unit's position",
          cache_key=hash(ss.belief.tobytes()))
with tab_mdp:
    col_algo, col_compute, col_replan = st.columns([2, 1, 1], vertical_alignment='bottom')
    mdp_label = col_algo.selectbox("Algorithm", list(MDP_ALGORITHMS), key='mdp_algorithm')
    if col_compute.button("Compute Policy", use_container_width=True):
        ss.pop('mdp_notice', None)
        with st.spinner(f"Running {mdp_label}..."):
            run_mdp_compute(mdp_label, selected_bot)
    if col_replan.button("Simulate adaptive replan", use_container_width=True):
        ss.pop('mdp_notice', None)
        with st.spinner("Flooding the next cell on the route and replanning..."):
            run_mdp_replan(mdp_label, selected_bot)
    st.caption(f"Planning for {selected_bot} (change it with Select Bot in the sidebar).")
    if ss.get('mdp_notice'):
        st.info(ss.mdp_notice)
    if ss.get('mdp_view'):
        plate(draw_mdp_map, MDP_LEGEND, "MDP value function heatmap with policy arrows and recommended route",
              cache_key=ss.mdp_view['stamp'])
        st.html(mdp_table_html())
    else:
        st.html('<div class="card" style="color:var(--text2)">Choose an algorithm and press <b>Compute Policy</b> '
                'to solve the cave MDP (+100 per survivor, −10 per cell with water above 0.6, −1 per step).</div>')
with tab_log:  # Rendered last so it includes entries logged by the MDP buttons above
    st.markdown(mission_log_html(), unsafe_allow_html=True)

    st.divider()
    st.subheader("Mission Database")
    stats = db.get_stats()
    m_total, m_rate, m_path, m_chamber = st.columns(4)
    m_total.metric("Total missions", stats['total'])
    m_rate.metric("Success rate", '—' if stats['success_rate'] is None else f"{stats['success_rate']:.0%}")
    m_path.metric("Avg path", '—' if stats['avg_path_length'] is None else f"{stats['avg_path_length']:.1f} moves",
                  help="Mean route length of missions that passed the safety check and were executed")
    m_chamber.metric("Most visited", '—' if stats['most_visited_chamber'] is None else
                     f"C{stats['most_visited_chamber']} · {CHAMBER_NAMES[stats['most_visited_chamber']]}")

    f_bot, f_chamber, f_outcome = st.columns(3)
    bot_filter = f_bot.selectbox("Bot", ["All", *BOT_IDS], key='db_bot')
    chamber_filter = f_chamber.selectbox("Chamber", ["All"] + [f"C{n}" for n in CHAMBER_MAP], key='db_chamber')
    outcome_filter = f_outcome.selectbox("Outcome", ["All", "PASS", "FAIL"], key='db_outcome')
    missions = db.query_missions(chamber=None if chamber_filter == "All" else int(chamber_filter[1:]),
                                 safety_outcome=None if outcome_filter == "All" else outcome_filter,
                                 bot_id=None if bot_filter == "All" else bot_filter)
    display_cols = ['timestamp', 'bot_id', 'command_text', 'action', 'target_chamber', 'payload', 'priority', 'path_length',
                    'safety_verdict', 'safety_reason', 'battery_before', 'battery_after', 'bot_x', 'bot_y',
                    'algorithm_used']
    if missions:
        st.dataframe(pd.DataFrame(missions)[display_cols], hide_index=True, width="stretch")
    else:
        st.info("No missions match these filters yet." if stats['total'] else
                "No missions logged yet. Send a rescue command to create the first record.")
