from pydantic import BaseModel, Field
from typing import Literal

from ai_server.utils.general import generate_id

class MessageQuery(BaseModel):
    query: str = Field(..., description="The query to be sent to the chatbot")
    id: str | None = Field(default_factory=lambda: generate_id(16, "nanoid"), description="The frontend-generated id of the message (e.g., from AI SDK)")

class ChatRequestOptions(BaseModel):
    api_type: Literal["responses", "chat_completion"] = Field(default="responses", description="OpenAI API type to use: 'responses' (default) or 'chat_completion'")

class ChatRequest(BaseModel):
    query_message: MessageQuery
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default_factory=lambda: generate_id(24, "mongodb"), description="The session id of the chat")
    user_id: str | None = Field(default=None, description="The user id of the user")
    new_chat: bool = Field(default=False, description="Whether a new chat should be started")
    new_user: bool = Field(default=False, description="Whether a new user has started the chat")
    options: ChatRequestOptions = Field(default_factory=ChatRequestOptions, description="Request options including API type selection")