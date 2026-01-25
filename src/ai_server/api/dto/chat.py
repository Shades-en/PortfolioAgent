from pydantic import BaseModel, Field

from ai_server.config import MONGODB_OBJECTID_LENGTH
from ai_server.utils.general import generate_id

class MessageQuery(BaseModel):
    query: str = Field(..., description="The query to be sent to the chatbot")
    id: str = Field(default_factory=lambda: generate_id(MONGODB_OBJECTID_LENGTH), description="The id of the message")

class ChatRequest(BaseModel):
    query_message: MessageQuery
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default=None, description="The session id of the chat")
    user_id: str | None = Field(default=None, description="The user id of the user")
    new_chat: bool = Field(default=False, description="Whether a new chat should be started")
    new_user: bool = Field(default=False, description="Whether a new user has started the chat")