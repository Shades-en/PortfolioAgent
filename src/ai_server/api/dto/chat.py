from pydantic import BaseModel, ConfigDict, Field

from omniagent.utils.general import generate_id
from omniagent.types import MessageQuery

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_message: MessageQuery
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default_factory=lambda: generate_id(24, "mongodb"), description="The session id of the chat")


class CancelChatRequest(BaseModel):
    """Request to cancel an in-progress chat stream."""
    session_id: str = Field(..., description="The session id of the chat to cancel")
