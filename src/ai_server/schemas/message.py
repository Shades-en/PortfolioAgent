from pydantic import BaseModel, model_validator
from enum import Enum
from uuid import uuid4
from pydantic import Field
from typing import Self
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
    tool_call_id: str | None
    metadata: dict
    content: str | None
    function_call: FunctionCallRequest | None
    session_id: str
    user_id: str
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))

    @model_validator(mode='after')
    def generate_hierarchical_ids(self) -> Self:
        """Generate hierarchical IDs with x-y format for session_id and x-y-z format for message_id."""
        # Ensure session_id follows x-y format (user_id-session_uuid)
        if not self.session_id.startswith(f"{self.user_id}-") or self.session_id.count('-') != 1:
            session_uuid = uuid4().hex
            self.session_id = f"{self.user_id}-{session_uuid}"
        
        # Extract the session part (y) from session_id
        session_part = self.session_id.split('-', 1)[1]
        
        # Ensure message_id follows x-y-z format (user_id-session_part-message_uuid)
        expected_prefix = f"{self.user_id}-{session_part}-"
        if not self.message_id.startswith(expected_prefix) or self.message_id.count('-') != 2:
            message_uuid = uuid4().hex
            self.message_id = f"{self.user_id}-{session_part}-{message_uuid}"
        
        return self

