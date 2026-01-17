from pydantic import BaseModel, Field


class SessionStarredRequest(BaseModel):
    """Request model for updating session starred status."""
    starred: bool = Field(..., description="Whether the session should be starred (true) or unstarred (false)")


class SessionRenameRequest(BaseModel):
    """Request model for renaming a session."""
    name: str = Field(..., description="New name for the session", min_length=1, max_length=200)
