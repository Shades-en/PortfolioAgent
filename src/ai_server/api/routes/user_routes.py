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
        
        return user.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
