from pydantic import BaseModel, Field
from typing import Optional


class SessionStarredRequest(BaseModel):
    """Request model for updating session starred status."""
    starred: bool = Field(..., description="Whether the session should be starred (true) or unstarred (false)")


class SessionRenameRequest(BaseModel):
    """Request model for renaming a session."""
    name: str = Field(..., description="New name for the session", min_length=1, max_length=200)


class GenerateChatNameRequest(BaseModel):
    """Request model for generating a chat name."""
    query: Optional[str] = Field(None, description="User query for context (required if no session_id)")
    session_id: Optional[str] = Field(None, description="Session ID to use context from (optional)")
    turns_between_chat_name: int = Field(default=20, description="Number of turns between chat name regeneration", ge=1)
    max_chat_name_length: int = Field(default=50, description="Maximum length for generated chat names", ge=10, le=200)
    max_chat_name_words: int = Field(default=5, description="Maximum words for generated chat names", ge=1, le=20)
