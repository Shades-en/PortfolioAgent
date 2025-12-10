from typing import List

from ai_server.schemas import Session, Message
from ai_server.config import DEFAULT_MESSAGE_PAGE_SIZE, DEFAULT_SESSION_PAGE_SIZE


class SessionService:
    @classmethod
    async def get_session_messages(
        cls,
        session_id: str,
        page: int = 1,
        page_size: int = DEFAULT_MESSAGE_PAGE_SIZE
    ) -> List[dict]:
        """
        Get paginated messages for a session.
        Returns most recent messages in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            page: Page number (1-indexed)
            page_size: Number of messages per page
            
        Returns:
            List of message dictionaries in chronological order
        """
        messages = await Message.get_paginated_by_session(
            session_id=session_id,
            page=page,
            page_size=page_size
        )
        return [msg.model_dump() for msg in messages]
    
    @classmethod
    async def get_all_session_messages(cls, session_id: str) -> List[dict]:
        """
        Get all messages for a session in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            
        Returns:
            List of all message dictionaries in chronological order
        """
        messages = await Message.get_all_by_session(session_id=session_id)
        return [msg.model_dump() for msg in messages]
    
    @classmethod
    async def get_user_sessions(
        cls,
        cookie_id: str,
        page: int = 1,
        page_size: int = DEFAULT_SESSION_PAGE_SIZE
    ) -> List[dict]:
        """
        Get paginated sessions for a user by cookie ID, sorted by most recent first.
        
        Args:
            cookie_id: The user's cookie ID
            page: Page number (1-indexed)
            page_size: Number of sessions per page
            
        Returns:
            List of session dictionaries sorted by most recent first
        """
        sessions = await Session.get_paginated_by_user_cookie(
            cookie_id=cookie_id,
            page=page,
            page_size=page_size
        )
        return [session.model_dump() for session in sessions]
    
    @classmethod
    async def get_all_user_sessions(cls, cookie_id: str) -> List[dict]:
        """
        Get all sessions for a user by cookie ID, sorted by most recent first.
        
        Args:
            cookie_id: The user's cookie ID
            
        Returns:
            List of all session dictionaries sorted by most recent first
        """
        sessions = await Session.get_all_by_user_cookie(cookie_id=cookie_id)
        return [session.model_dump() for session in sessions]
