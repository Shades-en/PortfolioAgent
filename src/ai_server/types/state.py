from pydantic import BaseModel
from ai_server.schemas import Summary

class State(BaseModel):
    step: int = 1
    turns_after_last_summary: int = 0 # Number of turns after last summary
    total_token_after_last_summary: int = 0 # Total token count of all turns since last summary
    active_summary: Summary | None = None # Summary of active conversation
    user_defined_state: dict = {},
    turn_number: int = 1
    new_chat: bool = True
    new_user: bool = True