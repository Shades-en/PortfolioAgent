from __future__ import annotations

from pydantic import BaseModel
from typing import Any

class State(BaseModel):
    step: int = 1
    turns_after_last_summary: int = 0 # Number of turns after last summary
    total_token_after_last_summary: int = 0 # Total token count of all turns since last summary
    active_summary: Any = None # Summary | None - using Any to avoid circular import with Pydantic
    user_defined_state: dict = {}
    turn_number: int = 1
    new_chat: bool = True
    new_user: bool = True