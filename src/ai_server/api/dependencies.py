"""FastAPI dependencies for request validation and common parameters."""

from fastapi import Header, HTTPException


async def get_cookie_id(
    x_cookie_id: str = Header(..., description="User's cookie ID for authorization")
) -> str:
    """
    FastAPI dependency to extract and validate X-Cookie-Id header.
    
    This dependency ensures that all protected routes have a valid cookie_id.
    Queries filter by cookie_id to ensure users can only access their own data.
    
    Args:
        x_cookie_id: The user's cookie ID from the X-Cookie-Id header
        
    Returns:
        The cookie_id string
        
    Raises:
        HTTPException 401: If the header is missing or empty
    """
    if not x_cookie_id or not x_cookie_id.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing or empty X-Cookie-Id header"
        )
    return x_cookie_id.strip()


async def get_optional_cookie_id(
    x_cookie_id: str | None = Header(None, description="User's cookie ID (optional)")
) -> str | None:
    """
    FastAPI dependency to extract optional X-Cookie-Id header.
    
    Returns None if header is missing or empty, otherwise returns the cookie_id.
    """
    if not x_cookie_id or not x_cookie_id.strip():
        return None
    return x_cookie_id.strip()
