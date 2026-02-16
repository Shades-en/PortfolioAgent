from fastapi import APIRouter, Query, HTTPException, Depends

from ai_server.api.services import UserService
from ai_server.api.dependencies import get_cookie_id

router = APIRouter()

@router.get("/users", tags=["User"])
async def get_user(
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Get a user by their cookie ID.
    
    Args:
        cookie_id: Cookie ID of the user
    
    Returns:
        Dictionary with user data
    
    Raises:
        HTTPException 404: If user not found
    """
    user = await UserService.get_user(cookie_id=cookie_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Use mode='json' to properly serialize ObjectId and datetime fields
    data = user.model_dump(mode="json")
    if "client_id" in data:
        data["cookie_id"] = data.pop("client_id")
    return data

@router.delete("/users", tags=["User"])
async def delete_user(
    cookie_id: str = Depends(get_cookie_id),
    cascade: bool = Query(default=True, description="If true, also delete all sessions and related data")
) -> dict:
    """
    Delete a user by their cookie ID and optionally cascade delete all related data.
    This operation is atomic and cannot be undone.
    
    Args:
        cookie_id: Cookie ID of the user
        cascade: If true, also delete all sessions (and their messages/turns/summaries)
    
    Returns:
        Dictionary with deletion counts:
        - user_deleted: Whether the user was deleted (true/false)
        - sessions_deleted: Number of sessions deleted
        - messages_deleted: Number of messages deleted (only if cascade=true)
        - turns_deleted: Number of turns deleted (only if cascade=true)
        - summaries_deleted: Number of summaries deleted (only if cascade=true)
    """
    return await UserService.delete_user(
        cookie_id=cookie_id,
        cascade=cascade
    )
