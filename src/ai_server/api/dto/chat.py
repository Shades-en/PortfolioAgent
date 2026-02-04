from pydantic import BaseModel, Field

from ai_server.utils.general import generate_id

class MessageQuery(BaseModel):
    query: str = Field(..., description="The query to be sent to the chatbot")
    id: str | None = Field(default=None, description="The frontend-generated id of the message (e.g., from AI SDK)")

class ChatRequest(BaseModel):
    query_message: MessageQuery
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default=lambda: generate_id(24, "mongodb"), description="The session id of the chat")
    user_id: str | None = Field(default=None, description="The user id of the user")
    new_chat: bool = Field(default=False, description="Whether a new chat should be started")
    new_user: bool = Field(default=False, description="Whether a new user has started the chat")