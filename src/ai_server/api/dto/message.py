from pydantic import BaseModel, Field
from enum import Enum


class FeedbackType(str, Enum):
    """Feedback types for messages."""
    LIKE = "liked"
    DISLIKE = "disliked"
    NEUTRAL = "neutral"


class MessageFeedbackRequest(BaseModel):
    """Request model for updating message feedback."""
    feedback: FeedbackType = Field(..., description="The feedback type: liked, disliked, or neutral")
