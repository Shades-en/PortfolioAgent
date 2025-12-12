from __future__ import annotations

from beanie import Document, Link
import pymongo
from bson import ObjectId

from datetime import datetime, timezone
from pydantic import Field
from typing import List, TYPE_CHECKING
import asyncio

from opentelemetry.trace import SpanKind

if TYPE_CHECKING:
    from pymongo.client_session import ClientSession
    from ai_server.schemas.message import Message
    from ai_server.schemas.summary import Summary
    from ai_server.schemas.turn import Turn

from ai_server.schemas.user import User
from ai_server.types.message import MessageDTO
from ai_server.api.exceptions.db_exceptions import (
    SessionRetrievalFailedException,
    SessionCreationFailedException,
    SessionUpdateFailedException,
    SessionDeletionFailedException,
    MessageCreationFailedException,
    TurnCreationFailedException
)
from ai_server.config import DEFAULT_SESSION_NAME, DEFAULT_SESSION_PAGE_SIZE
from ai_server.utils.tracing import trace_method, trace_operation, CustomSpanKinds

class Session(Document):
    name: str = Field(default_factory=lambda: DEFAULT_SESSION_NAME)
    user: Link[User]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "sessions"
        indexes = [
            [("user.$id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)]
        ]
    
    @classmethod
    async def get_by_id(cls, session_id: str) -> Session | None:
        """Retrieve a session by its MongoDB document ID."""
        try:
            return await cls.get(ObjectId(session_id))
        except Exception as e:
            raise SessionRetrievalFailedException(
                message="Failed to retrieve session by ID",
                note=f"session_id={session_id}, error={str(e)}"
            )
    
    @classmethod
    @trace_method(
        kind=SpanKind.INTERNAL,
        graph_node_id="db_create_session_with_user",
        capture_input=False,
        capture_output=False
    )
    async def create_with_user(
        cls,
        cookie_id: str,
        session_name: str = DEFAULT_SESSION_NAME
    ) -> Session:
        """
        Create a new session with a new user atomically using MongoDB transaction.
        
        Args:
            cookie_id: Cookie ID for the new user
            session_name: Name for the session
            
        Returns:
            Created Session document
            
        Raises:
            SessionCreationFailedException: If transaction fails
        
        Traced as INTERNAL span for database transaction.
        """
        from ai_server.db import MongoDB

        client = MongoDB.get_client()
        
        async with client.start_session() as session_txn:
            try:
                async with await session_txn.start_transaction():
                    # Create new user
                    new_user = User(cookie_id=cookie_id)
                    await new_user.insert(session=session_txn)
                    
                    # Create and save session
                    new_session = cls(
                        name=session_name,
                        user=new_user
                    )
                    await new_session.insert(session=session_txn)
                    
                    return new_session
                    
            except Exception as e:
                # Transaction will automatically abort on exception
                raise SessionCreationFailedException(
                    message="Failed to create session with user in transaction",
                    note=f"cookie_id={cookie_id}, session_name={session_name}, error={str(e)}"
                )
    
    @classmethod
    async def create_for_existing_user(
        cls,
        user: User,
        session_name: str = DEFAULT_SESSION_NAME
    ) -> Session:
        """
        Create a new session for an existing user.
        
        Args:
            user: Existing User document (must have an id)
            session_name: Name for the session
            
        Returns:
            Created Session document
            
        Raises:
            SessionCreationFailedException: If session creation fails or user has no id
        """
        if not user.id:
            raise SessionCreationFailedException(
                message="Cannot create session for user without id",
                note="User must be saved to database before creating a session"
            )
        
        try:
            new_session = cls(
                name=session_name,
                user=user
            )
            await new_session.insert()
            return new_session
            
        except Exception as e:
            raise SessionCreationFailedException(
                message="Failed to create session for existing user",
                note=f"user_id={user.id}, session_name={session_name}, error={str(e)}"
            )
    
    async def update_name(self, new_name: str) -> None:
        """
        Update the session name.
        
        Args:
            new_name: New name for the session
            
        Raises:
            SessionUpdateFailedException: If update fails
        """
        if not self.id:
            raise SessionUpdateFailedException(
                message="Cannot update name for unsaved session",
                note="Session must be saved to database before updating name"
            )
        
        try:
            self.name = new_name
            await self.save()
            
        except Exception as e:
            raise SessionUpdateFailedException(
                message="Failed to update session name",
                note=f"session_id={self.id}, new_name={new_name}, error={str(e)}"
            )
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.DATABASE.value)
    async def delete_with_related(cls, session_id: str) -> dict:
        """
        Delete session and all related documents (messages, turns, summaries) in a transaction.
        This prevents orphaned documents with dangling references.
        
        Args:
            session_id: MongoDB document ID of the session to delete
        
        Returns:
            Dictionary with deletion counts: {
                "messages_deleted": int, 
                "turns_deleted": int, 
                "summaries_deleted": int,
                "session_deleted": bool
            }
        
        Raises:
            SessionDeletionFailedException: If deletion fails
        
        Traced as INTERNAL span for database transaction with cascade delete.
        """
        from ai_server.schemas.message import Message
        from ai_server.schemas.turn import Turn
        from ai_server.schemas.summary import Summary
        from ai_server.db import MongoDB
        
        try:
            obj_id = ObjectId(session_id)
            session = await cls.get(obj_id)
            if not session:
                return {
                    "messages_deleted": 0,
                    "turns_deleted": 0,
                    "summaries_deleted": 0,
                    "session_deleted": False
                }
            
            client = MongoDB.get_client()
            
            async with client.start_session() as session_txn:
                async with await session_txn.start_transaction():
                    # Delete session, messages, turns, and summaries in parallel
                    delete_results = await asyncio.gather(
                        session.delete(session=session_txn),
                        Message.find(Message.session.id == obj_id).delete(session=session_txn),
                        Turn.find(Turn.session.id == obj_id).delete(session=session_txn),
                        Summary.find(Summary.session.id == obj_id).delete(session=session_txn)
                    )
                    
                    messages_deleted = delete_results[1].deleted_count if delete_results[1] else 0
                    turns_deleted = delete_results[2].deleted_count if delete_results[2] else 0
                    summaries_deleted = delete_results[3].deleted_count if delete_results[3] else 0
                    
                    return {
                        "messages_deleted": messages_deleted,
                        "turns_deleted": turns_deleted,
                        "summaries_deleted": summaries_deleted,
                        "session_deleted": True
                    }
                    
        except Exception as e:
            raise SessionDeletionFailedException(
                message="Failed to delete session with related documents",
                note=f"session_id={session_id}, error={str(e)}"
            )
    
    async def insert_messages(
        self, 
        messages: List[MessageDTO],
        session: ClientSession | None = None
    ) -> List[Message]:
        """
        Bulk insert messages for this session.
        
        Args:
            messages: List of MessageDTO objects to insert
            session: Optional MongoDB session for transaction support
            
        Returns:
            List of inserted Message documents
            
        Raises:
            MessageCreationFailedException: If bulk insert fails
        """
        if not self.id:
            raise MessageCreationFailedException(
                message="Cannot insert messages for unsaved session",
                note="Session must be saved to database before inserting messages"
            )
        
        if not messages:
            return []
        
        try:
            from ai_server.schemas.message import Message

            # Convert MessageDTOs to Message documents
            message_docs = []
            for msg_dto in messages:
                message_doc = Message(
                    role=msg_dto.role.value,  # Extract string value from enum
                    tool_call_id=msg_dto.tool_call_id,
                    metadata=msg_dto.metadata,
                    content=msg_dto.content,
                    function_call=msg_dto.function_call,
                    token_count=msg_dto.token_count,
                    error=msg_dto.error,
                    session=self
                )
                message_docs.append(message_doc)
            
            # Bulk insert all messages
            await Message.insert_many(message_docs, session=session)
            
            return message_docs
            
        except Exception as e:
            raise MessageCreationFailedException(
                message="Failed to bulk insert messages for session",
                note=f"session_id={self.id}, message_count={len(messages)}, error={str(e)}"
            )
    
    async def insert_turn(
        self,
        turn_number: int,
        message_docs: List[Message],
        summary: Summary | None = None,
        session: ClientSession | None = None
    ) -> Turn:
        """
        Create and insert a Turn for this session.
        
        Args:
            turn_number: The turn number for this turn
            message_docs: List of Message documents that belong to this turn
            summary: Optional Summary document (previous summary for this turn)
            session: Optional MongoDB session for transaction support
            
        Returns:
            Created Turn document
            
        Raises:
            TurnCreationFailedException: If turn creation fails
        """
        if not self.id:
            raise TurnCreationFailedException(
                message="Cannot insert turn for unsaved session",
                note="Session must be saved to database before inserting turns"
            )
        
        if not message_docs:
            raise TurnCreationFailedException(
                message="Cannot create turn without messages",
                note="At least one message is required to create a turn"
            )
        
        try:
            from ai_server.schemas.turn import Turn
            # Calculate total token count for the turn
            turn_token_count = sum(msg.token_count for msg in message_docs)
            
            # Create turn document
            new_turn = Turn(
                turn_number=turn_number,
                session=self,
                turn_token_count=turn_token_count,
                previous_summary=summary,
                messages=message_docs
            )
            
            await new_turn.insert(session=session)
            return new_turn
            
        except Exception as e:
            raise TurnCreationFailedException(
                message="Failed to create turn for session",
                note=f"session_id={self.id}, turn_number={turn_number}, message_count={len(message_docs)}, error={str(e)}"
            )
    
    @trace_method(
        kind=SpanKind.INTERNAL,
        graph_node_id="db_insert_messages_and_turn",
        capture_input=False,
        capture_output=False
    )
    async def insert_messages_and_turn(
        self,
        turn_number: int,
        messages: List[MessageDTO],
        summary: Summary | None = None
    ) -> tuple[List[Message], Turn]:
        """
        Insert messages and create a turn atomically in a single transaction.
        
        Args:
            turn_number: The turn number for this turn
            messages: List of MessageDTO objects to insert
            summary: Optional Summary document (previous summary for this turn)
            
        Returns:
            Tuple of (inserted Message documents, created Turn document)
            
        Raises:
            MessageCreationFailedException: If message insertion fails
            TurnCreationFailedException: If turn creation fails
        
        Traced as INTERNAL span for database transaction.
        """
        from ai_server.db import MongoDB

        if not self.id:
            raise TurnCreationFailedException(
                message="Cannot insert messages and turn for unsaved session",
                note="Session must be saved to database before inserting messages and turns"
            )
        
        if not messages:
            raise MessageCreationFailedException(
                message="Cannot create turn without messages",
                note="At least one message is required to create a turn"
            )
        
        client = MongoDB.get_client()
        
        async with client.start_session() as session_txn:
            try:
                async with await session_txn.start_transaction():
                    # Insert messages using the existing method
                    message_docs = await self.insert_messages(messages, session=session_txn)
                    
                    # Insert turn using the existing method
                    new_turn = await self.insert_turn(
                        turn_number=turn_number,
                        message_docs=message_docs,
                        summary=summary,
                        session=session_txn
                    )
                    
                    return message_docs, new_turn
                    
            except Exception as e:
                # Transaction will automatically abort on exception
                raise TurnCreationFailedException(
                    message="Failed to insert messages and turn in transaction",
                    note=f"session_id={self.id}, turn_number={turn_number}, message_count={len(messages)}, error={str(e)}"
                )
    
    @classmethod
    async def get_paginated_by_user_cookie(
        cls,
        cookie_id: str,
        page: int = 1,
        page_size: int = DEFAULT_SESSION_PAGE_SIZE
    ) -> List[Session]:
        """
        Get paginated sessions for a user by cookie ID, sorted by most recent first.
        Uses MongoDB aggregation with $lookup to join with users collection.
        
        Args:
            cookie_id: The user's cookie ID
            page: Page number (1-indexed)
            page_size: Number of sessions per page
            
        Returns:
            List of Session documents sorted by most recent first
            
        Raises:
            SessionRetrievalFailedException: If retrieval fails
        """
        try:
            skip = (page - 1) * page_size
            
            # Aggregation pipeline to lookup user by cookie_id and get sessions
            pipeline = [
                {
                    "$lookup": {
                        "from": "users",
                        "localField": "user.$id",
                        "foreignField": "_id",
                        "as": "user_data"
                    }
                },
                {
                    "$unwind": "$user_data"
                },
                {
                    "$match": {
                        "user_data.cookie_id": cookie_id
                    }
                },
                {
                    "$sort": {"created_at": -1}  # Most recent first
                },
                {
                    "$skip": skip
                },
                {
                    "$limit": page_size
                }
            ]
            
            sessions = await cls.aggregate(pipeline).to_list()
            
            # Convert aggregation results back to Session documents
            return [cls.model_validate(session) for session in sessions]
            
        except Exception as e:
            raise SessionRetrievalFailedException(
                message="Failed to retrieve paginated sessions for user by cookie",
                note=f"cookie_id={cookie_id}, page={page}, page_size={page_size}, error={str(e)}"
            )
    
    @classmethod
    async def get_all_by_user_cookie(cls, cookie_id: str) -> List[Session]:
        """
        Get all sessions for a user by cookie ID, sorted by most recent first.
        Uses MongoDB aggregation with $lookup to join with users collection.
        
        Args:
            cookie_id: The user's cookie ID
            
        Returns:
            List of all Session documents sorted by most recent first
            
        Raises:
            SessionRetrievalFailedException: If retrieval fails
        """
        try:
            # Aggregation pipeline to lookup user by cookie_id and get all sessions
            pipeline = [
                {
                    "$lookup": {
                        "from": "users",
                        "localField": "user.$id",
                        "foreignField": "_id",
                        "as": "user_data"
                    }
                },
                {
                    "$unwind": "$user_data"
                },
                {
                    "$match": {
                        "user_data.cookie_id": cookie_id
                    }
                },
                {
                    "$sort": {"created_at": -1}  # Most recent first
                }
            ]
            
            sessions = await cls.aggregate(pipeline).to_list()
            
            # Convert aggregation results back to Session documents
            return [cls.model_validate(session) for session in sessions]
            
        except Exception as e:
            raise SessionRetrievalFailedException(
                message="Failed to retrieve all sessions for user by cookie",
                note=f"cookie_id={cookie_id}, error={str(e)}"
            )

        