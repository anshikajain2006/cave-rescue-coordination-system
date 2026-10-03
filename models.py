from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

@dataclass
class Node:
    """Represents a grid cell for forklift navigation"""
    x: int
    y: int
    g: float = float('inf')  # Cost from start
    h: float = float('inf')  # Heuristic to goal
    f: float = float('inf')  # Total cost
    parent: 'Node' = None
    
    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class CaveCommand(BaseModel):
    """Structured rescue command produced by llm_parser from a natural-language instruction"""
    action: Literal["move", "deliver", "scan", "status", "replan", "hold"]
    target_chamber: Optional[int] = None    # Chamber number 0-9
    payload: Optional[str] = None           # e.g. "oxygen_kit", "medical_kit"
    priority: Literal["low", "normal", "high"] = "normal"
    bot_id: Optional[str] = None            # e.g. "UNIT-01"
    raw_text: str                           # The original user input, always preserved

    @field_validator('target_chamber')
    @classmethod
    def check_chamber_range(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not 0 <= value <= 9:
            raise ValueError(f"target_chamber must be between 0 and 9, got {value}")
        return value

    @model_validator(mode='after')
    def drop_target_for_local_actions(self) -> 'CaveCommand':
        if self.action in ('scan', 'status'):  # Scans and status reports act at the bot's current position
            self.target_chamber = None
        return self
