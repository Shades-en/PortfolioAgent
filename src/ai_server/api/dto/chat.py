from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="The query to be sent to the chatbot") # required
    session_id: str = Field(..., description="The session id of the chat") # required