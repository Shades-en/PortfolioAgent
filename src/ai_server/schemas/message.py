from pydantic import BaseModel, model_validator
from pydantic import Field
from enum import Enum
from typing import Self, List

from uuid import uuid4
from datetime import datetime, timezone

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
    turn_id: str
    session_id: str
    user_id: str
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    embedding: List[float] | None = None
    created_at: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))    


