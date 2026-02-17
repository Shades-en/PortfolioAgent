from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from omniagent.utils.general import generate_id
from omniagent.types import MessageQuery

class ChatRequest(BaseModel):
    query_message: MessageQuery
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default_factory=lambda: generate_id(24, "mongodb"), description="The session id of the chat")
    provider_options: Optional[Dict[str, Any]] = Field(default=None, description="Provider-specific options (e.g., {'api_type': 'chat_completion'} for OpenAI)")


class CancelChatRequest(BaseModel):
    """Request to cancel an in-progress chat stream."""
    session_id: str = Field(..., description="The session id of the chat to cancel")