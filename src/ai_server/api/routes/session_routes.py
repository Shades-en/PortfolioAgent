from fastapi import APIRouter, Query, HTTPException, Depends
from bson.errors import InvalidId

from ai_server.api.services import SessionService
from ai_server.api.dto.session import SessionStarredRequest, SessionRenameRequest, GenerateChatNameRequest
from ai_server.api.dependencies import get_cookie_id, get_optional_cookie_id
from omniagent.exceptions import SessionUpdateError
from omniagent.config import (
    DEFAULT_MESSAGE_PAGE_SIZE,
    MAX_MESSAGE_PAGE_SIZE,
    DEFAULT_SESSION_PAGE_SIZE,
    MAX_SESSION_PAGE_SIZE
)

router = APIRouter()

@router.get("/sessions/all", tags=["Session"])
async def get_all_user_sessions(
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Get all sessions for a user by cookie ID.
    Returns all sessions sorted by most recent first.
    Warning: This may return a large amount of data for users with many sessions.
    
    Returns:
        {
            "count": int,
            "results": List[dict]
        }
    """
    return await SessionService.get_all_user_sessions(cookie_id=cookie_id)

@router.get("/sessions/starred", tags=["Session"])
async def get_starred_user_sessions(
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Get all starred sessions for a user by cookie ID.
    Returns starred sessions sorted by most recently updated first.
    
    Returns:
        {
            "count": int,
            "results": List[dict]
        }
    """
    return await SessionService.get_starred_user_sessions(cookie_id=cookie_id)

@router.get("/sessions/{session_id}", tags=["Session"])
async def get_session(
    session_id: str,
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Get a session by its ID.
    
    Args:
        session_id: The session ID to retrieve
        cookie_id: User's cookie ID for authorization
    
    Returns:
        Dictionary with session data
    
    Raises:
        HTTPException 400: If session ID format is invalid
        HTTPException 404: If session not found or doesn't belong to user
    """
    try:
        session = await SessionService.get_session(session_id=session_id, cookie_id=cookie_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return session
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")

@router.get("/sessions/{session_id}/messages", tags=["Session"])
async def get_session_messages(
    session_id: str,
    cookie_id: str = Depends(get_cookie_id),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=DEFAULT_MESSAGE_PAGE_SIZE, ge=1, le=MAX_MESSAGE_PAGE_SIZE, description="Number of messages per page")
) -> dict:
    """
    Get paginated messages for a session.
    Returns most recent messages in chronological order (oldest to newest).
    
    Args:
        session_id: The session ID
        cookie_id: User's cookie ID for authorization
    
    Returns:
        {
            "count": int,
            "results": List[dict]
        }
    """
    return await SessionService.get_session_messages(
        session_id=session_id,
        cookie_id=cookie_id,
        page=page,
        page_size=page_size
    )

@router.get("/sessions/{session_id}/messages/all", tags=["Session"])
async def get_all_session_messages(
    session_id: str,
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Get all messages for a session in chronological order (oldest to newest).
    Warning: This may return a large amount of data for sessions with many messages.
    
    Args:
        session_id: The session ID
        cookie_id: User's cookie ID for authorization
    
    Returns:
        {
            "count": int,
            "results": List[dict]
        }
    """
    return await SessionService.get_all_session_messages(session_id=session_id, cookie_id=cookie_id)

@router.get("/sessions", tags=["Session"])
async def get_user_sessions(
    cookie_id: str = Depends(get_cookie_id),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=DEFAULT_SESSION_PAGE_SIZE, ge=1, le=MAX_SESSION_PAGE_SIZE, description="Number of sessions per page")
) -> dict:
    """
    Get paginated sessions for a user by cookie ID.
    Returns sessions sorted by most recent first.
    
    Returns:
        {
            "count": int,
            "results": List[dict]
        }
    """
    return await SessionService.get_user_sessions(
        cookie_id=cookie_id,
        page=page,
        page_size=page_size
    )

@router.patch("/sessions/{session_id}/name", tags=["Session"])
async def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Rename a session.
    
    Args:
        session_id: The session ID to rename
        request: The rename request containing new name
        cookie_id: User's cookie ID for authorization
    
    Returns:
        Dictionary with update info:
        - session_updated: Whether the session was updated (true/false)
        - session_id: The session ID that was updated
        - name: The new name that was set
    
    Raises:
        HTTPException 400: If session ID format is invalid
        HTTPException 404: If session not found or doesn't belong to user
        HTTPException 500: If update fails
    """
    try:
        result = await SessionService.rename_session(session_id=session_id, name=request.name, cookie_id=cookie_id)
        if not result["session_updated"]:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return result
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
    except SessionUpdateError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")

@router.patch("/sessions/{session_id}/starred", tags=["Session"])
async def update_session_starred(
    session_id: str,
    request: SessionStarredRequest,
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Update the starred status for a session.
    
    Args:
        session_id: The session ID to update
        request: The starred request containing starred status (true/false)
        cookie_id: User's cookie ID for authorization
    
    Returns:
        Dictionary with update info:
        - session_updated: Whether the session was updated (true/false)
        - session_id: The session ID that was updated
        - starred: The starred status that was set (true/false)
    
    Raises:
        HTTPException 400: If session ID format is invalid
        HTTPException 404: If session not found or doesn't belong to user
        HTTPException 500: If update fails
    """
    try:
        return await SessionService.update_session_starred(session_id=session_id, starred=request.starred, cookie_id=cookie_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
    except SessionUpdateError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        raise HTTPException(status_code=500, detail=f"Failed to update session starred status: {str(e)}")

@router.delete("/sessions/{session_id}", tags=["Session"])
async def delete_session(
    session_id: str,
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Delete a session and all its related data (messages, turns, summaries).
    This operation is atomic and cannot be undone.
    
    Args:
        session_id: The session ID to delete
        cookie_id: User's cookie ID for authorization
    
    Returns:
        Dictionary with deletion counts:
        - messages_deleted: Number of messages deleted
        - turns_deleted: Number of turns deleted
        - summaries_deleted: Number of summaries deleted
        - session_deleted: Whether the session was deleted (true/false)
    """
    return await SessionService.delete_session(session_id=session_id, cookie_id=cookie_id)

@router.delete("/sessions", tags=["Session"])
async def delete_all_user_sessions(
    cookie_id: str = Depends(get_cookie_id)
) -> dict:
    """
    Delete all sessions for a user and all related data (messages, summaries).
    This operation is atomic and cannot be undone.
    
    Args:
        cookie_id: The user's cookie ID
    
    Returns:
        Dictionary with deletion counts:
        - sessions_deleted: Number of sessions deleted
        - messages_deleted: Number of messages deleted
        - summaries_deleted: Number of summaries deleted
    
    Raises:
        HTTPException 500: If deletion fails
    """
    try:
        return await SessionService.delete_all_user_sessions(cookie_id=cookie_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user sessions: {str(e)}")


@router.post("/sessions/generate-name", tags=["Session"])
async def generate_chat_name(
    request: GenerateChatNameRequest,
    cookie_id: str | None = Depends(get_optional_cookie_id)
) -> dict:
    """
    Generate a chat name based on query and/or session context.
    
    - If session_id is provided: Uses session context (summary + recent messages) + query
    - If no session_id: Uses only the query (required in this case)
    
    Args:
        request: Request containing query, optional session_id, and chat name configuration
        cookie_id: User's cookie ID (required if session_id is provided)
    
    Returns:
        Dictionary with generated name:
        - name: The generated chat name
        - session_id: The session ID or null
    
    Raises:
        HTTPException 400: If query is missing when no session_id provided
        HTTPException 400: If session ID format is invalid
        HTTPException 500: If generation fails
    """
    # Validate: query is required if no session_id
    if not request.session_id and not request.query:
        raise HTTPException(status_code=400, detail="Query is required when no session_id is provided")
    
    try:
        return await SessionService.generate_chat_name(
            query=request.query or "",
            turns_between_chat_name=request.turns_between_chat_name,
            max_chat_name_length=request.max_chat_name_length,
            max_chat_name_words=request.max_chat_name_words,
            session_id=request.session_id,
            cookie_id=cookie_id
        )
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {request.session_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate chat name: {str(e)}")
