from fastapi import APIRouter, Query
from typing import List

from ai_server.api.services import SessionService
from ai_server.config import (
    DEFAULT_MESSAGE_PAGE_SIZE,
    MAX_MESSAGE_PAGE_SIZE,
    DEFAULT_SESSION_PAGE_SIZE,
    MAX_SESSION_PAGE_SIZE
)

router = APIRouter()

@router.get("/sessions/{session_id}/messages", tags=["Session"])
async def get_session_messages(
    session_id: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=DEFAULT_MESSAGE_PAGE_SIZE, ge=1, le=MAX_MESSAGE_PAGE_SIZE, description="Number of messages per page")
) -> List[dict]:
    """
    Get paginated messages for a session.
    Returns most recent messages in chronological order (oldest to newest).
    """
    return await SessionService.get_session_messages(
        session_id=session_id,
        page=page,
        page_size=page_size
    )

@router.get("/sessions/{session_id}/messages/all", tags=["Session"])
async def get_all_session_messages(session_id: str) -> List[dict]:
    """
    Get all messages for a session in chronological order (oldest to newest).
    Warning: This may return a large amount of data for sessions with many messages.
    """
    return await SessionService.get_all_session_messages(session_id=session_id)

@router.get("/sessions", tags=["Session"])
async def get_user_sessions(
    cookie_id: str = Query(..., description="User's cookie ID"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=DEFAULT_SESSION_PAGE_SIZE, ge=1, le=MAX_SESSION_PAGE_SIZE, description="Number of sessions per page")
) -> List[dict]:
    """
    Get paginated sessions for a user by cookie ID.
    Returns sessions sorted by most recent first.
    """
    return await SessionService.get_user_sessions(
        cookie_id=cookie_id,
        page=page,
        page_size=page_size
    )

@router.get("/sessions/all", tags=["Session"])
async def get_all_user_sessions(
    cookie_id: str = Query(..., description="User's cookie ID")
) -> List[dict]:
    """
    Get all sessions for a user by cookie ID.
    Returns all sessions sorted by most recent first.
    Warning: This may return a large amount of data for users with many sessions.
    """
    return await SessionService.get_all_user_sessions(cookie_id=cookie_id)
