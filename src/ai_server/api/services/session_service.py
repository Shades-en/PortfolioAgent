import asyncio
import math

from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.config import DEFAULT_MESSAGE_PAGE_SIZE, DEFAULT_SESSION_PAGE_SIZE
from omniagent.db.document_models import get_document_models
from omniagent.session import MongoSessionManager
from ai_server.utils.tracing import trace_operation


class SessionService:
    @staticmethod
    def _session_model():
        return get_document_models().session

    @staticmethod
    def _message_model():
        return get_document_models().message

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session(cls, session_id: str, cookie_id: str) -> dict | None:
        """
        Get a session by its ID.
        
        Args:
            session_id: The session ID
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with session data or None if not found
        
        Traced as CHAIN span for service-level orchestration.
        """
        session_model = cls._session_model()
        session = await session_model.get_by_id_and_client_id(session_id=session_id, client_id=cookie_id)
        if session is None:
            return None
        # Use mode='json' to serialize ObjectIds and exclude Link fields
        return session.model_dump(mode='json', exclude={'user'})
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session_messages(
        cls,
        session_id: str,
        cookie_id: str,
        page: int = 1,
        page_size: int = DEFAULT_MESSAGE_PAGE_SIZE
    ) -> dict:
        """
        Get paginated messages for a session.
        Returns most recent messages in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            cookie_id: The user's cookie ID for authorization
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
        session_model = cls._session_model()
        message_model = cls._message_model()
        session = await session_model.get_by_id_and_client_id(session_id=session_id, client_id=cookie_id)
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
            message_model.get_paginated_by_session(
                session_id=session_id,
                page=page,
                page_size=page_size
            ),
            message_model.count_by_session(session_id=session_id)
        )
        
        # Calculate pagination metadata
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        has_next = page < total_pages
        has_previous = page > 1
        
        results = message_model.to_public_dicts(messages)
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
    async def get_all_session_messages(cls, session_id: str, cookie_id: str) -> dict:
        """
        Get all messages for a session in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with count and results: {
                "count": int,
                "results": List[dict]
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        # First verify the session belongs to the user
        session_model = cls._session_model()
        message_model = cls._message_model()
        session = await session_model.get_by_id_and_client_id(session_id=session_id, client_id=cookie_id)
        if session is None:
            return {
                "count": 0,
                "results": []
            }
        
        messages = await message_model.get_all_by_session(session_id=session_id)
        results = message_model.to_public_dicts(messages)
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
        session_model = cls._session_model()
        sessions, total_count = await asyncio.gather(
            session_model.get_paginated_by_user_client_id(
                client_id=cookie_id,
                page=page,
                page_size=page_size
            ),
            session_model.count_by_user_client_id(client_id=cookie_id)
        )
        
        # Calculate pagination metadata
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        has_next = page < total_pages
        has_previous = page > 1
        
        results = session_model.to_public_dicts(sessions)
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
        session_model = cls._session_model()
        sessions = await session_model.get_all_by_user_client_id(client_id=cookie_id)
        results = session_model.to_public_dicts(sessions)
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
        session_model = cls._session_model()
        sessions = await session_model.get_starred_by_user_client_id(client_id=cookie_id)
        results = session_model.to_public_dicts(sessions)
        return {
            "count": len(results),
            "results": results
        }
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def update_session_starred(cls, session_id: str, starred: bool, cookie_id: str) -> dict:
        """
        Update the starred status for a session.
        
        Args:
            session_id: The session ID to update
            starred: Whether the session should be starred (True) or unstarred (False)
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with update info: {
                "session_updated": bool,
                "session_id": str,
                "starred": bool
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        session_model = cls._session_model()
        return await session_model.update_starred_by_client_id(session_id, starred, client_id=cookie_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def rename_session(cls, session_id: str, name: str, cookie_id: str) -> dict:
        """
        Rename a session.
        
        Args:
            session_id: The session ID to rename
            name: New name for the session
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with update info: {
                "session_updated": bool,
                "session_id": str,
                "name": str
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        session_model = cls._session_model()
        return await session_model.update_name_by_client_id(session_id=session_id, name=name, client_id=cookie_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_session(cls, session_id: str, cookie_id: str) -> dict:
        """
        Delete a session and all its related data (messages, turns, summaries).
        
        Args:
            session_id: The session ID to delete
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with deletion counts: {
                "messages_deleted": int,
                "turns_deleted": int,
                "summaries_deleted": int,
                "session_deleted": bool
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        session_model = cls._session_model()
        return await session_model.delete_with_related_by_client_id(session_id, client_id=cookie_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_all_user_sessions(cls, cookie_id: str) -> dict:
        """
        Delete all sessions for a user and all related data (messages, summaries).
        
        Args:
            cookie_id: The user's cookie ID
            
        Returns:
            Dictionary with deletion counts: {
                "sessions_deleted": int,
                "messages_deleted": int,
                "summaries_deleted": int
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        session_model = cls._session_model()
        return await session_model.delete_all_by_user_client_id(client_id=cookie_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def generate_chat_name(
        cls,
        query: str,
        turns_between_chat_name: int,
        max_chat_name_length: int,
        max_chat_name_words: int,
        session_id: str | None = None,
        cookie_id: str | None = None,
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
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with generated name: {
                "name": str,
                "session_id": str | None
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        chat_name = await MongoSessionManager.generate_chat_name(
            query=query,
            turns_between_chat_name=turns_between_chat_name,
            max_chat_name_length=max_chat_name_length,
            max_chat_name_words=max_chat_name_words,
            session_id=session_id,
            client_id=cookie_id,
        )

        return {"name": chat_name, "session_id": session_id}
