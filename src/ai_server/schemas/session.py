from __future__ import annotations

from beanie import Document, Link
import pymongo
from bson import ObjectId

from datetime import datetime, timezone
from pydantic import Field
from typing import List, TYPE_CHECKING

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
    MessageCreationFailedException,
    TurnCreationFailedException
)
from ai_server.db import MongoDB
from ai_server.config import DEFAULT_SESSION_NAME, DEFAULT_SESSION_PAGE_SIZE

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
        """
        client = MongoDB.get_client()
        
        async with await client.start_session() as session_txn:
            async with session_txn.start_transaction():
                try:
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
    
    async def insert_messages(
        self, 
        messages: List[MessageDTO],
        session: "ClientSession | None" = None
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
                    role=msg_dto.role,
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
        session: "ClientSession | None" = None
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
        """
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
        
        async with await client.start_session() as session_txn:
            async with session_txn.start_transaction():
                try:
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

        