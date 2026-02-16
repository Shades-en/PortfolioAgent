import asyncio
import math

from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.schemas import Session, Message, Summary
from omniagent.ai.providers import get_llm_provider
from omniagent.config import DEFAULT_MESSAGE_PAGE_SIZE, DEFAULT_SESSION_PAGE_SIZE, LLM_PROVIDER
from ai_server.utils.tracing import trace_operation


class SessionService:
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session(cls, session_id: str, user_id: str) -> dict:
        """
        Get a session by its ID.
        
        Args:
            session_id: The session ID
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with session data or None if not found
        
        Traced as CHAIN span for service-level orchestration.
        """
        session = await Session.get_by_id(session_id=session_id, user_id=user_id)
        if session is None:
            return None
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        return session.model_dump(mode='json', exclude={'user'})
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session_messages(
        cls,
        session_id: str,
        user_id: str,
        page: int = 1,
        page_size: int = DEFAULT_MESSAGE_PAGE_SIZE
    ) -> dict:
        """
        Get paginated messages for a session.
        Returns most recent messages in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            user_id: The user's MongoDB document ID for authorization
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
        # First verify the session belongs to the user
        session = await Session.get_by_id(session_id=session_id, user_id=user_id)
        if session is None:
            return {
                "count": 0,
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "has_next": False,
                "has_previous": False,
                "results": []
            }
        
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
        
        # Convert to MessageDTO and serialize
        message_dtos = Message.to_dtos(messages)
        results = [dto.model_dump(mode='json') for dto in message_dtos]
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
    async def get_all_session_messages(cls, session_id: str, user_id: str) -> dict:
        """
        Get all messages for a session in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        # First verify the session belongs to the user
        session = await Session.get_by_id(session_id=session_id, user_id=user_id)
        if session is None:
            return {
                "count": 0,
                "results": []
            }
        
        messages = await Message.get_all_by_session(session_id=session_id)
        # Convert to MessageDTO and serialize
        message_dtos = Message.to_dtos(messages)
        results = [dto.model_dump(mode='json') for dto in message_dtos]
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
    async def update_session_starred(cls, session_id: str, starred: bool, user_id: str) -> dict:
        """
        Update the starred status for a session.
        
        Args:
            session_id: The session ID to update
            starred: Whether the session should be starred (True) or unstarred (False)
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with update info: {
                "session_updated": bool,
                "session_id": str,
                "starred": bool
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Session.update_starred(session_id, starred, user_id=user_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def rename_session(cls, session_id: str, name: str, user_id: str) -> dict:
        """
        Rename a session.
        
        Args:
            session_id: The session ID to rename
            name: New name for the session
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with update info: {
                "session_updated": bool,
                "session_id": str,
                "name": str
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        session = await Session.get_by_id(session_id=session_id, user_id=user_id)
        
        if session is None:
            return {
                "session_updated": False,
                "session_id": session_id,
                "name": name
            }
        
        await session.update_name(new_name=name)
        return {
            "session_updated": True,
            "session_id": session_id,
            "name": name
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_session(cls, session_id: str, user_id: str) -> dict:
        """
        Delete a session and all its related data (messages, turns, summaries).
        
        Args:
            session_id: The session ID to delete
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with deletion counts: {
                "messages_deleted": int,
                "turns_deleted": int,
                "summaries_deleted": int,
                "session_deleted": bool
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Session.delete_with_related(session_id, user_id=user_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_all_user_sessions(cls, user_id: str) -> dict:
        """
        Delete all sessions for a user and all related data (messages, summaries).
        
        Args:
            user_id: The user's MongoDB document ID
            
        Returns:
            Dictionary with deletion counts: {
                "sessions_deleted": int,
                "messages_deleted": int,
                "summaries_deleted": int
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Session.delete_all_by_user_id(user_id=user_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def generate_chat_name(
        cls,
        query: str,
        turns_between_chat_name: int,
        max_chat_name_length: int,
        max_chat_name_words: int,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        """
        Generate a chat name for a session.
        
        For new chats (no session_id): Generates name from query only.
        For existing sessions: Fetches context (summary + recent messages) and generates name.
        
        Args:
            query: The user's query for context
            turns_between_chat_name: Number of turns between chat name regeneration (from frontend)
            max_chat_name_length: Maximum length for generated chat names (from frontend)
            max_chat_name_words: Maximum words for generated chat names (from frontend)
            session_id: Optional session ID (if generating for existing session)
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with generated name: {
                "name": str,
                "session_id": str | None
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        llm_provider = get_llm_provider(provider_name=LLM_PROVIDER)
        
        # For new chats - generate from query only
        if not session_id:
            chat_name = await llm_provider.generate_chat_name(
                query=query,
                max_chat_name_length=max_chat_name_length,
                max_chat_name_words=max_chat_name_words,
            )
            return {
                "name": chat_name,
                "session_id": None
            }
        
        # For existing sessions - fetch context first
        session = await Session.get_by_id(session_id=session_id, user_id=user_id)
        if session is None:
            # Session not found - generate from query only as fallback
            chat_name = await llm_provider.generate_chat_name(
                query=query,
                max_chat_name_length=max_chat_name_length,
                max_chat_name_words=max_chat_name_words,
            )
            return {
                "name": chat_name,
                "session_id": session_id
            }
        
        # Calculate context size: 2 * turns_between_chat_name
        chat_name_context_max_messages = 2 * turns_between_chat_name
        
        # Fetch summary and recent messages in parallel
        summary_task = Summary.get_latest_by_session(session_id=session_id)
        messages_task = Message.get_paginated_by_session(
            session_id=session_id,
            page=1,
            page_size=chat_name_context_max_messages,
        )
        summary, messages = await asyncio.gather(summary_task, messages_task)
        
        # Convert messages to MessageDTO format
        conversation_to_summarize = Message.to_dtos(messages) if messages else None
        
        # Generate chat name with context
        chat_name = await llm_provider.generate_chat_name(
            query=query,
            previous_summary=summary,
            conversation_to_summarize=conversation_to_summarize,
            max_chat_name_length=max_chat_name_length,
            max_chat_name_words=max_chat_name_words,
        )
        
        return {
            "name": chat_name,
            "session_id": session_id
        }
