import json
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from openai.types.shared_params import ResponseFormatJSONSchema

logger = logging.getLogger(__name__)

MODEL = 'gpt-4o-mini'
API_TIMEOUT_SECONDS = 15

# Chamber number -> (x, y) free cell in the CaveEnvironment grid
CHAMBER_MAP: Dict[int, Tuple[int, int]] = {
    0: (0, 0),     # Entrance
    1: (3, 6),     # Entry Hall
    2: (3, 14),    # Lower Sump (survivor)
    3: (12, 5),    # Deep Pocket (survivor)
    4: (17, 2),    # North Gallery (survivor)
    5: (17, 12),   # South Gallery
    6: (8, 4),     # Central Tunnel
    7: (10, 11),   # Mid Cavern
    8: (8, 14),    # Lower Gallery
    9: (15, 9),    # East Chamber
}

# Canonical chamber names (also shown in the app) and the nicknames the LLM should map to each number
CHAMBER_NAMES: Dict[int, str] = {
    0: 'Entrance',
    1: 'Entry Hall',
    2: 'Lower Sump',
    3: 'Deep Pocket',
    4: 'North Gallery',
    5: 'South Gallery',
    6: 'Central Tunnel',
    7: 'Mid Cavern',
    8: 'Lower Gallery',
    9: 'East Chamber',
}
CHAMBER_ALIASES: Dict[int, List[str]] = {
    0: ['entrance', 'start', 'base'],
    1: ['entry hall', 'staging area'],
    2: ['lower sump', 'sump', 'flooded section', 'bottom'],
    3: ['deep pocket', 'pocket', 'central chamber'],
    4: ['north gallery', 'north', 'east gallery', 'upper gallery'],
    5: ['south gallery', 'south'],
    6: ['central tunnel', 'tunnel', 'middle'],
    7: ['mid cavern', 'cavern', 'mid section'],
    8: ['lower gallery', 'lower section'],
    9: ['east chamber', 'east'],
}

ACTIONS = ('navigate', 'scan', 'status')
PRIORITIES = ('low', 'medium', 'high')

CHAMBER_LINES = '\n'.join(
    f"- C{n}: {CHAMBER_NAMES[n]} ({x},{y}) — also called " + ', '.join(f'"{a}"' for a in CHAMBER_ALIASES[n])
    for n, (x, y) in CHAMBER_MAP.items())

SYSTEM_PROMPT = f"""You are a cave rescue command parser. Extract structured mission intent from natural language rescue commands.

The cave has these chambers (use the NUMBER as target_chamber):
{CHAMBER_LINES}
Chambers 2, 3 and 4 hold known survivors.

Return ONLY valid JSON with these fields:
- action: one of "navigate" (move to a chamber, including rescue and delivery runs), "scan" (search or sonar sweep for survivors), "status" (report position, battery or mission state)
- target_chamber: integer 0-9, or null if no chamber is given or it is outside 0-9
- payload: short snake_case item to deliver, e.g. "oxygen_kit", "medical_kit", "water", "food", "thermal_blanket", "rope", "radio"; or null if none
- priority: one of "high" (urgent, emergency or life-threatening wording), "medium" (default), "low" (explicitly non-urgent)

Only use information in the command; never invent a chamber or payload.

Example: "Rush an oxygen kit to the lower sump" -> {{"action": "navigate", "target_chamber": 2, "payload": "oxygen_kit", "priority": "high"}}"""

RESPONSE_SCHEMA: 'ResponseFormatJSONSchema' = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'rescue_command',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': list(ACTIONS)},
                'target_chamber': {'type': ['integer', 'null']},
                'payload': {'type': ['string', 'null']},
                'priority': {'type': 'string', 'enum': list(PRIORITIES)},
            },
            'required': ['action', 'target_chamber', 'payload', 'priority'],
            'additionalProperties': False,
        },
    },
}


def parse_command(user_text: str, api_key: str) -> dict:
    """Parses a rescue command with gpt-4o-mini, falling back to keyword rules if the call fails"""
    try:
        return _parse_with_llm(user_text, api_key)
    except Exception as exc:
        logger.warning("LLM parse failed (%s: %s); using keyword fallback", type(exc).__name__, exc)
        return _parse_with_keywords(user_text)


def _parse_with_llm(user_text: str, api_key: str) -> dict:
    from openai import OpenAI  # Imported here so the fallback still works if openai is not installed

    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format=RESPONSE_SCHEMA,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text},
        ],
    )
    content = response.choices[0].message.content
    if content is None:  # Refusal or empty completion
        raise ValueError("model returned no content")
    return _validate(json.loads(content))


def _validate(result: dict) -> dict:
    """Enforces the output contract; raises ValueError so bad model output triggers the fallback"""
    action, priority = result.get('action'), result.get('priority')
    chamber, payload = result.get('target_chamber'), result.get('payload')
    if action not in ACTIONS:
        raise ValueError(f"invalid action {action!r}")
    if priority not in PRIORITIES:
        raise ValueError(f"invalid priority {priority!r}")
    if chamber is not None and (isinstance(chamber, bool) or not isinstance(chamber, int)):
        raise ValueError(f"invalid target_chamber {chamber!r}")
    if payload is not None and not isinstance(payload, str):
        raise ValueError(f"invalid payload {payload!r}")
    return {
        'action': action,
        'target_chamber': chamber if chamber in CHAMBER_MAP else None,
        'payload': (payload.strip() or None) if payload else None,
        'priority': priority,
    }


# =====================================================================
# KEYWORD FALLBACK
# =====================================================================
NUMBER_WORDS = {w: i for i, w in enumerate(
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])}
# Chamber names recognised by the keyword fallback; checked in order, first match wins
CHAMBER_NAME_KEYWORDS: Dict[str, int] = {
    'lower sump': 2, 'sump': 2,
    'deep pocket': 3, 'pocket': 3,
    'north gallery': 4, 'east gallery': 4,
    'south gallery': 5,
    'central tunnel': 6, 'tunnel': 6,
    'mid cavern': 7, 'cavern': 7,
    'lower gallery': 8,
    'east chamber': 9,
    'entrance': 0, 'entry hall': 1,
}
# "chamber 3" / "chamber three" / "chamber #3" (group 1), or the short id "c3" / "C-3" (group 2; text is lowercased)
CHAMBER_PATTERN = re.compile(r'\b(?:chamber\s*(?:#|no\.?|number)?\s*(\d+|' + '|'.join(NUMBER_WORDS) + r')|c-?(\d+))\b')

# Checked in order; first match wins
PAYLOAD_KEYWORDS = [
    ('oxygen', 'oxygen_kit'),
    ('medical', 'medical_kit'),
    ('first aid', 'medical_kit'),
    ('medkit', 'medical_kit'),
    ('blanket', 'thermal_blanket'),
    ('water', 'water'),
    ('food', 'food'),
    ('rope', 'rope'),
    ('radio', 'radio'),
]
SCAN_WORDS = ('scan', 'search', 'sonar', 'sweep', 'look for', 'check for', 'locate')
STATUS_WORDS = ('status', 'report', 'battery', 'where are you', 'position', 'progress')
HIGH_PRIORITY_WORDS = ('urgent', 'emergency', 'immediately', 'critical', 'asap', 'now', 'hurry', 'dying')
LOW_PRIORITY_WORDS = ('low priority', 'no rush', 'when possible', 'whenever', 'not urgent')


def _parse_with_keywords(user_text: str) -> dict:
    text = user_text.lower()

    chamber: Optional[int] = next(
        (number for name, number in CHAMBER_NAME_KEYWORDS.items() if re.search(rf'\b{re.escape(name)}\b', text)), None)
    match = CHAMBER_PATTERN.search(text)  # An explicit "chamber N" / "cN" overrides a name
    if match:
        token = match.group(1) or match.group(2)
        number = NUMBER_WORDS[token] if token in NUMBER_WORDS else int(token)
        chamber = number if number in CHAMBER_MAP else None

    payload = next((name for word, name in PAYLOAD_KEYWORDS if word in text), None)

    if any(word in text for word in SCAN_WORDS):
        action = 'scan'
    elif any(word in text for word in STATUS_WORDS) and chamber is None and payload is None:
        action = 'status'
    else:
        action = 'navigate'

    if any(word in text for word in LOW_PRIORITY_WORDS):
        priority = 'low'
    elif any(re.search(rf'\b{re.escape(word)}\b', text) for word in HIGH_PRIORITY_WORDS):
        priority = 'high'
    else:
        priority = 'medium'

    return {'action': action, 'target_chamber': chamber, 'payload': payload, 'priority': priority}
