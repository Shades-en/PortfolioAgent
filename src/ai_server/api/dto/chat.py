from pydantic import BaseModel, Field
from typing import Literal

from omniagent.utils.general import generate_id
from omniagent.types import MessageQuery

class ChatRequestOptions(BaseModel):
    api_type: Literal["responses", "chat_completion"] = Field(default="responses", description="OpenAI API type to use: 'responses' (default) or 'chat_completion'")

class ChatRequest(BaseModel):
    query_message: MessageQuery
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default_factory=lambda: generate_id(24, "mongodb"), description="The session id of the chat")
    options: ChatRequestOptions = Field(default_factory=ChatRequestOptions, description="Request options including API type selection")


class CancelChatRequest(BaseModel):
    """Request to cancel an in-progress chat stream."""
    session_id: str = Field(..., description="The session id of the chat to cancel")