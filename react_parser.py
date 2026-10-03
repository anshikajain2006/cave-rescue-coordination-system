"""
ReACT parser: iterative Reason → Act → Observe loop for command parsing.
Max 3 iterations before falling back to keyword parser.

Clarifying questions: clarify_callback(question) -> answer is called when the model asks one. A UI that cannot
block (Streamlit) raises AwaitingClarification from the callback instead; the exception comes back out of
react_parse carrying a ReactState, and resume_react(state, answer, ...) continues the loop from that point.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

from llm_parser import API_TIMEOUT_SECONDS, CHAMBER_LINES, MODEL
from llm_parser import _parse_with_keywords as keyword_parse  # The keyword fallback
from models import CaveCommand

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

REACT_SYSTEM_PROMPT = f"""You are the command parser for a cave rescue robot system.
You receive a natural language command and must extract a structured CaveCommand.

You operate in a Reason-Act-Observe loop:
- REASON: think about what the command means and what is ambiguous
- ACT: either output a parsed CaveCommand as JSON, or ask ONE clarifying question
- OBSERVE: you will receive the user's answer or a validation error, then loop again

The cave has these chambers (use the NUMBER as target_chamber):
{CHAMBER_LINES}
Chambers 2, 3 and 4 hold known survivors.

CaveCommand schema:
{{ "action": "move"|"deliver"|"scan"|"status"|"replan"|"hold",
  "target_chamber": 0-9 or null,
  "payload": string or null,
  "priority": "low"|"normal"|"high",
  "bot_id": string or null,
  "raw_text": string (always the original input) }}

Use "deliver" when a payload is carried to a chamber, "move" when nothing is. "scan" and "status" act at the
bot's current position, so their target_chamber is null. payload is short snake_case, e.g. "oxygen_kit",
"medical_kit", "water", "food", "thermal_blanket", "rope", "radio". bot_id looks like "UNIT-01".
Priority is "high" for urgent or life-threatening wording, "low" only when explicitly non-urgent, else "normal".

When confident, output exactly:
PARSED: {{"action": ..., "target_chamber": ..., ...}}

When ambiguous, output exactly:
CLARIFY: <one short question>

Never output anything else."""

Trace = List[dict]
ClarifyCallback = Callable[[str], str]


@dataclass
class ReactState:
    """Everything needed to continue a paused ReACT loop"""
    user_text: str
    messages: List['ChatCompletionMessageParam']
    trace: Trace
    iteration: int                       # Iteration that asked the pending question
    question: str = ''
    extra: dict = field(default_factory=dict)  # Free for the caller, e.g. which bot the command was for


class AwaitingClarification(Exception):
    """Raised by a non-blocking clarify_callback; react_parse re-raises it with .state filled in"""
    def __init__(self, question: str = ''):
        super().__init__(question)
        self.state: Optional[ReactState] = None


def react_parse(user_text: str, api_key: Optional[str],
                clarify_callback: Optional[ClarifyCallback] = None) -> Tuple[CaveCommand, Trace]:
    """
    Run the ReACT loop to parse a command.

    clarify_callback: callable(question: str) -> str  — gets the user's answer to a clarification question.
      If None (non-interactive mode), an ambiguous command falls back to the keyword parser.

    Returns (CaveCommand, trace) where trace is the list of ReACT steps for display.
    Raises ValueError if even the keyword parser cannot understand the command, or AwaitingClarification
    (with .state) when the callback defers the answer.
    """
    if not api_key:
        cmd = keyword_parse(user_text)
        return cmd, [{"step": 1, "type": "FALLBACK", "reason": "No API key — keyword parser used", "result": cmd.action}]

    messages: List['ChatCompletionMessageParam'] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Command to parse: {user_text}"},
    ]
    return _run_loop(ReactState(user_text, messages, [], iteration=0), api_key, clarify_callback)


def resume_react(state: ReactState, answer: str, api_key: Optional[str],
                 clarify_callback: Optional[ClarifyCallback] = None) -> Tuple[CaveCommand, Trace]:
    """Continues a loop paused on a clarifying question, with the user's answer"""
    state.trace[-1]["answer"] = answer
    state.messages.append({"role": "user", "content": f"User answered: {answer}"})
    if not api_key:
        return _fallback(state, "No API key to resume with — keyword parser used")
    return _run_loop(state, api_key, clarify_callback)


def _run_loop(state: ReactState, api_key: str, clarify_callback: Optional[ClarifyCallback]
              ) -> Tuple[CaveCommand, Trace]:
    try:
        from openai import OpenAI
    except ImportError:
        return _fallback(state, "openai not installed")
    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT_SECONDS)
    messages, trace = state.messages, state.trace

    for iteration in range(state.iteration + 1, MAX_ITERATIONS + 1):
        try:
            response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=200, temperature=0)
        except Exception as exc:  # Network, auth, rate limit: the command still gets parsed
            logger.warning("ReACT LLM call failed (%s: %s)", type(exc).__name__, exc)
            return _fallback(state, f"LLM call failed: {type(exc).__name__}")
        content = (response.choices[0].message.content or '').strip()
        messages.append({"role": "assistant", "content": content})

        if content.startswith("PARSED:"):
            json_str = content[len("PARSED:"):].strip()
            try:
                data = json.loads(json_str)
                if not isinstance(data, dict):
                    raise ValueError(f"expected a JSON object, got {type(data).__name__}")
                data["raw_text"] = state.user_text
                cmd = CaveCommand(**data)
                trace.append({"step": iteration, "type": "PARSED", "content": content, "result": cmd.action})
                return cmd, trace
            except ValueError as e:  # json.JSONDecodeError and pydantic's ValidationError are both ValueErrors
                trace.append({"step": iteration, "type": "PARSE_ERROR", "content": content, "error": str(e)})
                messages.append({"role": "user", "content": f"Validation error: {e}. Try again."})

        elif content.startswith("CLARIFY:"):
            question = content[len("CLARIFY:"):].strip()
            trace.append({"step": iteration, "type": "CLARIFY", "question": question})
            if clarify_callback is None:
                # Non-interactive: treat ambiguous as best-effort keyword parse
                trace.append({"step": iteration, "type": "CLARIFY_SKIPPED", "reason": "non-interactive mode"})
                return _fallback(state, "clarification unavailable")
            try:
                answer = clarify_callback(question)
            except AwaitingClarification as pause:
                state.iteration, state.question = iteration, question
                pause.state = state
                raise
            trace[-1]["answer"] = answer
            messages.append({"role": "user", "content": f"User answered: {answer}"})

        else:
            trace.append({"step": iteration, "type": "UNEXPECTED", "content": content})
            messages.append({"role": "user", "content": "Output must start with PARSED: or CLARIFY:"})

    return _fallback(state, "max iterations reached")


def _fallback(state: ReactState, reason: str) -> Tuple[CaveCommand, Trace]:
    cmd = keyword_parse(state.user_text)
    step = max((t["step"] for t in state.trace), default=0) + 1
    state.trace.append({"step": step, "type": "FALLBACK", "reason": reason, "result": cmd.action})
    return cmd, state.trace
