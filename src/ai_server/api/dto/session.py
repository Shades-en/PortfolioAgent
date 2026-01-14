from pydantic import BaseModel, Field


class SessionStarredRequest(BaseModel):
    """Request model for updating session starred status."""
    starred: bool = Field(..., description="Whether the session should be starred (true) or unstarred (false)")
