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
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        messages = await Message.get_paginated_by_session(
            session_id=session_id,
            page=page,
            page_size=page_size
        )
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [msg.model_dump(mode='json', exclude={'session', 'previous_summary'}) for msg in messages]
        return {
            "count": len(results),
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
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        sessions = await Session.get_paginated_by_user_cookie(
            cookie_id=cookie_id,
            page=page,
            page_size=page_size
        )
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        results = [session.model_dump(mode='json', exclude={'user'}) for session in sessions]
        return {
            "count": len(results),
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
