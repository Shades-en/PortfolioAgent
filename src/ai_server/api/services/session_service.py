import asyncio
import math

from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from ai_server.schemas import Session, Message
from ai_server.config import DEFAULT_MESSAGE_PAGE_SIZE, DEFAULT_SESSION_PAGE_SIZE
from ai_server.utils.tracing import trace_operation


class SessionService:
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session_messages(
        cls,
        session_id: str,
        page: int = 1,
        page_size: int = DEFAULT_MESSAGE_PAGE_SIZE
    ) -> dict:
        """
        Get paginated messages for a session.
        Returns most recent messages in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            page: Page number (1-indexed)
            page_size: Number of messages per page
            
        Returns:
            Dictionary with count, total_count, and results: {
                "count": int,  # Number of messages in current page
                "total_count": int,  # Total number of messages for the session
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        # Fetch paginated messages and total count in parallel
        messages, total_count = await asyncio.gather(
            Message.get_paginated_by_session(
                session_id=session_id,
                page=page,
                page_size=page_size
            ),
            Message.count_by_session(session_id=session_id)
        )
        
        # Calculate pagination metadata
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        has_next = page < total_pages
        has_previous = page > 1
        
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [msg.model_dump(mode='json', exclude={'session', 'previous_summary'}) for msg in messages]
        return {
            "count": len(results),
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_previous": has_previous,
            "results": results
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_all_session_messages(cls, session_id: str) -> dict:
        """
        Get all messages for a session in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            
        Returns:
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        messages = await Message.get_all_by_session(session_id=session_id)
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [msg.model_dump(mode='json', exclude={'session', 'previous_summary'}) for msg in messages]
        return {
            "count": len(results),
            "results": results
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_user_sessions(
        cls,
        cookie_id: str,
        page: int = 1,
        page_size: int = DEFAULT_SESSION_PAGE_SIZE
    ) -> dict:
        """
        Get paginated sessions for a user by cookie ID, sorted by most recent first.
        
        Args:
            cookie_id: The user's cookie ID
            page: Page number (1-indexed)
            page_size: Number of sessions per page
            
        Returns:
            Dictionary with count, total_count, and results: {
                "count": int,  # Number of sessions in current page
                "total_count": int,  # Total number of sessions for the user
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        # Fetch paginated sessions and total count in parallel
        sessions, total_count = await asyncio.gather(
            Session.get_paginated_by_user_cookie(
                cookie_id=cookie_id,
                page=page,
                page_size=page_size
            ),
            Session.count_by_user_cookie(cookie_id=cookie_id)
        )
        
        # Calculate pagination metadata
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        has_next = page < total_pages
        has_previous = page > 1
        
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [session.model_dump(mode='json', exclude={'user'}) for session in sessions]
        return {
            "count": len(results),
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_previous": has_previous,
            "results": results
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_all_user_sessions(cls, cookie_id: str) -> dict:
        """
        Get all sessions for a user by cookie ID, sorted by most recent first.
        
        Args:
            cookie_id: The user's cookie ID
            
        Returns:
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        sessions = await Session.get_all_by_user_cookie(cookie_id=cookie_id)
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [session.model_dump(mode='json', exclude={'user'}) for session in sessions]
        return {
            "count": len(results),
            "results": results
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_starred_user_sessions(cls, cookie_id: str) -> dict:
        """
        Get all starred sessions for a user by cookie ID, sorted by most recently updated first.
        
        Args:
            cookie_id: The user's cookie ID
            
        Returns:
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        sessions = await Session.get_starred_by_user_cookie(cookie_id=cookie_id)
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [session.model_dump(mode='json', exclude={'user'}) for session in sessions]
        return {
            "count": len(results),
            "results": results
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def update_session_starred(cls, session_id: str, starred: bool) -> dict:
        """
        Update the starred status for a session.
        
        Args:
            session_id: The session ID to update
            starred: Whether the session should be starred (True) or unstarred (False)
            
        Returns:
            Dictionary with update info: {
                "session_updated": bool,
                "session_id": str,
                "starred": bool
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Session.update_starred(session_id, starred)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_session(cls, session_id: str) -> dict:
        """
        Delete a session and all its related data (messages, turns, summaries).
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            Dictionary with deletion counts: {
                "messages_deleted": int,
                "turns_deleted": int,
                "summaries_deleted": int,
                "session_deleted": bool
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Session.delete_with_related(session_id)
