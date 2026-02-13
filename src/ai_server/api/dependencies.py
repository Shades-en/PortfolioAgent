"""FastAPI dependencies for request validation and common parameters."""

from fastapi import Header, HTTPException


async def get_user_id(
    x_user_id: str = Header(..., description="User's MongoDB document ID for authentication")
) -> str:
    """
    FastAPI dependency to extract and validate X-User-Id header.
    
    This dependency ensures that all protected routes have a valid user_id.
    Queries filter by user_id to ensure users can only access their own data.
    
    Args:
        x_user_id: The user's MongoDB document ID from the X-User-Id header
        
    Returns:
        The user_id string
        
    Raises:
        HTTPException 401: If the header is missing or empty
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing or empty X-User-Id header"
        )
    return x_user_id.strip()


async def get_optional_user_id(
    x_user_id: str | None = Header(None, description="User's MongoDB document ID (optional)")
) -> str | None:
    """
    FastAPI dependency to extract optional X-User-Id header.
    
    Returns None if header is missing or empty, otherwise returns the user_id.
    """
    if not x_user_id or not x_user_id.strip():
        return None
    return x_user_id.strip()
