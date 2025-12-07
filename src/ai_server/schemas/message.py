from pydantic import BaseModel, model_validator, Field
from enum import Enum
from typing import Self

from uuid import uuid4
from datetime import datetime, timezone

from ai_server.utils.general import get_token_count

class Role(Enum):
    HUMAN = 'human'
    SYSTEM = 'system'
    AI = 'ai'
    TOOL = 'tool'

class FunctionCallRequest(BaseModel):
    name: str
    arguments: dict

class Message(BaseModel):
    role: Role
    # tool_call_id is given "null" when not exists because redis tag field does not accept None
    tool_call_id: str 
    metadata: dict
    content: str | None
    function_call: FunctionCallRequest | None
    token_count: int = 0
    turn_id: str
    session_id: str
    user_id: str
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))
    
    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from content if not explicitly provided."""
        if self.token_count == 0 and self.content:
            self.token_count = get_token_count(self.content)
        return self    



