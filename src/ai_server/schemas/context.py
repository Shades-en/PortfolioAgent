from pydantic import BaseModel
from typing import List
from ai_server.schemas.message import Message

class Context(BaseModel):
    summary: str

    # Subset  of Previous conversation
    previous_conversation: List[Message]

    # Total token count including summary and previous conversation (mandatory turns and additional turns)
    context_token_count: int 

    # Number of turns in the user conversation after last summary
    turns_after_last_summary: int

    # Total token count since last summary - only previous conversation
    total_token_count_since_last_summary: int