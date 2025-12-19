from fastapi import APIRouter, Query, HTTPException

from ai_server.api.services import UserService

router = APIRouter()

@router.get("/users", tags=["User"])
async def get_user(
    user_id: str | None = Query(default=None, description="User ID"),
    cookie_id: str | None = Query(default=None, description="Cookie ID")
) -> dict:
    """
    Get a user by their ID or cookie ID.
    At least one of user_id or cookie_id must be provided.
    """
    try:
        user = await UserService.get_user(user_id=user_id, cookie_id=cookie_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Use mode='json' to properly serialize ObjectId and datetime fields
        return user.model_dump(mode='json')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/users", tags=["User"])
async def delete_user(
    user_id: str | None = Query(default=None, description="User ID"),
    cookie_id: str | None = Query(default=None, description="Cookie ID"),
    cascade: bool = Query(default=True, description="If true, also delete all sessions and related data")
) -> dict:
    """
    Delete a user by their ID or cookie ID and optionally cascade delete all related data.
    At least one of user_id or cookie_id must be provided.
    This operation is atomic and cannot be undone.
    
    Args:
        user_id: MongoDB document ID of the user (optional)
        cookie_id: Cookie ID of the user (optional)
        cascade: If true, also delete all sessions (and their messages/turns/summaries)
    
    Returns:
        Dictionary with deletion counts:
        - user_deleted: Whether the user was deleted (true/false)
        - sessions_deleted: Number of sessions deleted
        - messages_deleted: Number of messages deleted (only if cascade=true)
        - turns_deleted: Number of turns deleted (only if cascade=true)
        - summaries_deleted: Number of summaries deleted (only if cascade=true)
    """
    try:
        return await UserService.delete_user(
            user_id=user_id,
            cookie_id=cookie_id,
            cascade=cascade
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
