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

    @model_validator(mode='after')
    def validate_hierarchical_ids(self) -> Self:
        """Validate session_id format and generate hierarchical message_id."""
        # Validate that session_id follows x-y format (user_id_session_id)
        expected_session_prefix = f"{self.user_id}_"
        if not self.session_id.startswith(expected_session_prefix):
            raise ValueError(
                f"session_id must start with '{expected_session_prefix}'. "
                f"Expected format: 'userid_sessionid', got: '{self.session_id}'"
            )
        
        if self.session_id.count('_') != 1:
            raise ValueError(
                f"session_id must follow 'userid_sessionid' format with exactly one underscore. "
                f"Got: '{self.session_id}'"
            )
        
        # Extract the session part (session_id) from session_id
        session_part = self.session_id.split('_', 1)[1]
        
        if not session_part:
            raise ValueError(
                f"session_id must have a non-empty sessionid part after 'userid_'. "
                f"Got: '{self.session_id}'"
            )
        
        # Validate that turn_id follows userid_sessionid_turnid format
        expected_turn_prefix = f"{self.user_id}_{session_part}_"
        if not self.turn_id.startswith(expected_turn_prefix):
            raise ValueError(
                f"turn_id must start with '{expected_turn_prefix}'. "
                f"Expected format: 'userid_sessionid_turnid', got: '{self.turn_id}'"
            )
        
        if self.turn_id.count('_') != 2:
            raise ValueError(
                f"turn_id must follow 'userid_sessionid_turnid' format with exactly two underscores. "
                f"Got: '{self.turn_id}'"
            )
        
        # Extract the turn part (turnid) from turn_id
        turn_part = self.turn_id.split('_', 2)[2]
        
        if not turn_part:
            raise ValueError(
                f"turn_id must have a non-empty turnid part after 'userid_sessionid_'. "
                f"Got: '{self.turn_id}'"
            )
        
        return self


