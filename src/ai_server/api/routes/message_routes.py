from fastapi import APIRouter, HTTPException
from bson.errors import InvalidId

from ai_server.api.services import MessageService

router = APIRouter()

@router.delete("/messages/{message_id}", tags=["Message"])
async def delete_message(message_id: str) -> dict:
    """
    Delete a message by its ID and remove its reference from any Turn.
    This operation is atomic and cannot be undone.
    
    Args:
        message_id: The message ID to delete
    
    Returns:
        Dictionary with deletion info:
        - message_deleted: Whether the message was deleted (true/false)
        - turns_updated: Number of turns updated (0 or 1)
    """
    try:
        return await MessageService.delete_message(message_id=message_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid message ID format: {message_id}")