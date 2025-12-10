from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="The query to be sent to the chatbot")
    user_cookie: str = Field(..., description="The cookie id of the user")
    session_id: str | None = Field(default=None, description="The session id of the chat")
    user_id: str | None = Field(default=None, description="The user id of the user")
    turn_number: int = Field(default=1, description="The turn number of the conversation")
    new_chat: bool = Field(default=False, description="Whether a new chat should be started")
    new_user: bool = Field(default=False, description="Whether a new user has started the chat")