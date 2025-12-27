from pydantic import BaseModel, model_validator, Field
from enum import Enum
from typing import Self
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

class MessageDTO(BaseModel):
    """
    Data Transfer Object for Message.
    Used for in-memory message representation in conversation flows.
    Does not include DB-specific fields like session Link.
    """
    role: Role
    tool_call_id: str = "null"
    metadata: dict = Field(default_factory=dict)
    content: str | None = None
    function_call: FunctionCallRequest | None = None
    token_count: int = 0
    error: bool = False
    order: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from content if not explicitly provided."""
        if self.token_count == 0 and self.content:
            self.token_count = get_token_count(self.content)
        return self



