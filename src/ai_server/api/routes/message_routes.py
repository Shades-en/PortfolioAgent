from fastapi import APIRouter, HTTPException, Depends

from ai_server.api.services import MessageService
from ai_server.api.dto.message import MessageFeedbackRequest, FeedbackType
from ai_server.api.dependencies import get_user_id
from omniagent.types import Feedback
from omniagent.exceptions import MessageUpdateError

router = APIRouter()

@router.patch("/messages/{client_message_id}/feedback", tags=["Message"])
async def update_message_feedback(
    client_message_id: str,
    request: MessageFeedbackRequest,
    user_id: str = Depends(get_user_id)
) -> dict:
    """
    Update the feedback for a message.
    
    Args:
        client_message_id: The frontend-generated message ID (from AI SDK)
        request: The feedback request containing feedback type (liked, disliked, or neutral)
        user_id: User's MongoDB document ID from X-User-Id header
    
    Returns:
        Dictionary with update info:
        - message_updated: Whether the message was updated (true/false)
        - message_id: The message ID that was updated
        - feedback: The feedback value that was set (liked, disliked, or null for neutral)
    
    Raises:
        HTTPException 401: If X-User-Id header is missing
        HTTPException 404: If message not found or doesn't belong to user
        HTTPException 500: If update fails
    """
    try:
        # Convert FeedbackType to Feedback enum or None
        if request.feedback == FeedbackType.NEUTRAL:
            feedback = None
        elif request.feedback == FeedbackType.LIKE:
            feedback = Feedback.LIKE
        elif request.feedback == FeedbackType.DISLIKE:
            feedback = Feedback.DISLIKE
        else:
            feedback = None
        
        return await MessageService.update_message_feedback(client_message_id=client_message_id, feedback=feedback, user_id=user_id)
    except MessageUpdateError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Message not found: {client_message_id}")
        raise HTTPException(status_code=500, detail=f"Failed to update message feedback: {str(e)}")

@router.delete("/messages/{client_message_id}", tags=["Message"])
async def delete_message(
    client_message_id: str,
    user_id: str = Depends(get_user_id)
) -> dict:
    """
    Delete a message by its client ID.
    
    Args:
        client_message_id: The frontend-generated message ID (from AI SDK)
        user_id: User's MongoDB document ID from X-User-Id header
    
    Returns:
        Dictionary with deletion info:
        - message_deleted: Whether the message was deleted (true/false)
        - deleted_count: Number of documents deleted (0 or 1)
    
    Raises:
        HTTPException 401: If X-User-Id header is missing
    """
    return await MessageService.delete_message(client_message_id=client_message_id, user_id=user_id)