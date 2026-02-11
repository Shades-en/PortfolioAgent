from __future__ import annotations

from beanie import Document, Link
import pymongo
from bson import ObjectId

from pydantic import Field
from typing import TYPE_CHECKING, List
from datetime import datetime, timezone

from opentelemetry.trace import SpanKind

from ai_server.utils.tracing import trace_operation, CustomSpanKinds
from ai_server.api.exceptions.db_exceptions import (
    MessageRetrievalFailedException,
    MessageDeletionFailedException,
    MessageUpdateFailedException
)
from ai_server.config import DEFAULT_MESSAGE_PAGE_SIZE, MAX_TURNS_TO_FETCH
from ai_server.types.message import Role, MessageAITextPart, MessageReasoningPart, MessageToolPart, MessageHumanTextPart, Feedback, MessageDTO

if TYPE_CHECKING:
    from ai_server.schemas.session import Session
    from ai_server.schemas.summary import Summary

class Message(Document):
    role: Role
    metadata: dict = Field(default_factory=dict)
    parts: list[MessageHumanTextPart | MessageAITextPart | MessageReasoningPart | MessageToolPart] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    previous_summary: Link[Summary] | None = None
    turn_number: int = 1
    session: Link[Session]
    feedback: Feedback | None = None
    client_message_id: str | None = Field(default=None, description="Frontend-generated message ID (e.g., from AI SDK)")

    class Settings:
        name = "messages"
        indexes = [
            # For chronological queries and pagination with role ordering
            # Ascending role ('assistant' < 'user'), reversed in code to get user first
            [
                ("session._id", pymongo.ASCENDING), 
                ("created_at", pymongo.DESCENDING),
                ("role", pymongo.ASCENDING),  # 'assistant' < 'user' alphabetically
            ],
            # For turn-based range queries (get_latest_by_session)
            [
                ("session._id", pymongo.ASCENDING),
                ("turn_number", pymongo.ASCENDING),
                ("created_at", pymongo.ASCENDING),
            ]
        ]
    
    @property
    def token_count(self) -> int:
        """
        Get the total token count for this message by summing all part token counts.
        
        Returns:
            The total token count for this message.
        """
        total = 0
        for part in self.parts:
            if hasattr(part, 'token_count'):
                total += part.token_count
            elif hasattr(part, 'input_token_count') and hasattr(part, 'output_token_count'):
                # For MessageToolPart which has separate input/output counts
                total += part.input_token_count + part.output_token_count
        return total
    
    def to_dto(self) -> MessageDTO:
        """
        Convert this Message document to a MessageDTO object.
        
        Returns:
            MessageDTO object with data from this Message document
        """
        return MessageDTO(
            id=self.client_message_id,
            role=self.role,
            parts=self.parts,
            metadata=self.metadata,
            created_at=self.created_at,
            feedback=self.feedback,
        )
    
    @classmethod
    def to_dtos(cls, messages: List[Message]) -> List[MessageDTO]:
        """
        Convert a list of Message documents to MessageDTO objects.
        
        Args:
            messages: List of Message documents
            
        Returns:
            List of MessageDTO objects
        """
        return [msg.to_dto() for msg in messages]
    
    @classmethod
    async def get_paginated_by_session(
        cls,
        session_id: str,
        page: int = 1,
        page_size: int = DEFAULT_MESSAGE_PAGE_SIZE
    ) -> List[Message]:
        """
        Get paginated messages for a session.
        Fetches the most recent messages but returns them in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            page: Page number (1-indexed)
            page_size: Number of messages per page
            
        Returns:
            List of Message documents in chronological order
            
        Raises:
            MessageRetrievalFailedException: If retrieval fails
        """
        try:
            skip = (page - 1) * page_size
            
            # First, get messages sorted by most recent (descending)
            # When created_at is the same, user messages come before assistant messages
            # Use cls.session._id for querying Link fields
            messages = await cls.find(
                cls.session._id == ObjectId(session_id)
            ).sort(
                -cls.created_at,  # Descending to get most recent
                +cls.role,  # Ascending: 'assistant' < 'user', reversed later to get user first
            ).skip(skip).limit(page_size).to_list()
            
            # Then reverse to get chronological order (oldest to newest)
            messages.reverse()
            
            return messages
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to retrieve paginated messages for session",
                note=f"session_id={session_id}, page={page}, page_size={page_size}, error={str(e)}"
            )
    
    @classmethod
    async def count_by_session(cls, session_id: str) -> int:
        """
        Get the total count of messages for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Total count of messages for the session
            
        Raises:
            MessageRetrievalFailedException: If retrieval fails
        """
        try:
            count = await cls.find(
                cls.session._id == ObjectId(session_id)
            ).count()
            
            return count
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to count messages for session",
                note=f"session_id={session_id}, error={str(e)}"
            )
    
    @classmethod
    async def get_all_by_session(cls, session_id: str) -> List[Message]:
        """
        Get all messages for a session in chronological order (oldest to newest).
        
        Args:
            session_id: The session ID
            
        Returns:
            List of all Message documents in chronological order
            
        Raises:
            MessageRetrievalFailedException: If retrieval fails
        """
        try:
            # Use cls.session._id for querying Link fields
            messages = await cls.find(
                cls.session._id == ObjectId(session_id)
            ).sort(
                +cls.created_at,  # Ascending order (oldest first)
                -cls.role,  # Descending: 'user' > 'assistant' alphabetically, user comes first
            ).to_list()
            
            return messages
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to retrieve all messages for session",
                note=f"session_id={session_id}, error={str(e)}"
            )
    
    @classmethod
    async def get_latest_by_session(
        cls, 
        session_id: str | None,
        current_turn_number: int,
        max_turns: int = MAX_TURNS_TO_FETCH,
    ) -> List[Message]:
        """
        Get messages from the latest N turns for a session in chronological order (oldest to newest).
        
        This fetches messages from the most recent turns. For example, if max_turns=5,
        it will fetch all messages from the 5 most recent turns.
        
        Args:
            session_id: The session ID (returns empty list if None)
            max_turns: Maximum number of turns to fetch messages from
            current_turn_number: The current turn number (must be >= latest turn in DB)
            
        Returns:
            List of Message documents from the latest turns in chronological order (oldest to newest).
            
        Raises:
            MessageRetrievalFailedException: If retrieval fails or validation fails
        """
        if not session_id:
            return []
        
        try:
            # Calculate the minimum turn number to fetch
            # current_turn_number is guaranteed to be >= latest_turn_number + 1
            min_turn_number = max(1, current_turn_number - max_turns)
            # Ex: current_turn_number = 106, max_turns = 100
            # min_turn_number = max(1, 106 - 100) = 6
            
            # Fetch all messages from the latest N turns in chronological order (oldest to newest).
            # Ex: turn_number >= 6 is true for turns 6 to 105 (105 turns are in DB currently)
            # 105 - 6 + 1 = 100 turns are fetched
            messages = await cls.find(
                cls.session._id == ObjectId(session_id),
                cls.turn_number >= min_turn_number
            ).sort(
                +cls.created_at,  # Ascending order (oldest first)
            ).to_list()
            
            return messages
        except MessageRetrievalFailedException:
            raise
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to retrieve latest messages for session",
                note=f"session_id={session_id}, max_turns={max_turns}, current_turn_number={current_turn_number}, error={str(e)}"
            )
    
    @classmethod
    async def get_by_client_id(cls, client_message_id: str, user_id: str) -> Message | None:
        """
        Retrieve a message by its client_message_id (frontend-generated ID), filtered by user_id.
        This ensures users can only access messages from their own sessions.
        
        Args:
            client_message_id: The frontend-generated message ID (from AI SDK)
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Message if found and belongs to user's session, None otherwise
            
        Raises:
            MessageRetrievalFailedException: If retrieval fails
        """
        try:
            user_obj_id = ObjectId(user_id)
            
            # Query message by client_message_id and filter by session's user
            pipeline = [
                {
                    "$match": {
                        "client_message_id": client_message_id
                    }
                },
                {
                    "$lookup": {
                        "from": "sessions",
                        "localField": "session._id",
                        "foreignField": "_id",
                        "as": "session_data"
                    }
                },
                {
                    "$unwind": "$session_data"
                },
                {
                    "$match": {
                        "session_data.user.$id": user_obj_id
                    }
                }
            ]
            
            results = await cls.aggregate(pipeline).to_list()
            if not results:
                return None
            return cls.model_validate(results[0])
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to retrieve message by client ID",
                note=f"client_message_id={client_message_id}, user_id={user_id}, error={str(e)}"
            )
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.DATABASE.value)
    async def update_feedback(cls, client_message_id: str, feedback: Feedback | None, user_id: str) -> dict:
        """
        Update the feedback for a message.
        
        Args:
            client_message_id: The frontend-generated message ID (from AI SDK)
            feedback: The feedback value (LIKE, DISLIKE, or None for neutral/removal)
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with update info: {
                "message_updated": bool,
                "message_id": str,
                "feedback": str | None
            }
            
        Raises:
            MessageUpdateFailedException: If update fails
        
        Traced as INTERNAL span for database operation.
        """
        try:
            message = await cls.get_by_client_id(client_message_id, user_id)
            
            if not message:
                raise MessageUpdateFailedException(
                    message="Message not found",
                    note=f"client_message_id={client_message_id}"
                )
            
            message.feedback = feedback
            await message.save()
            
            return {
                "message_updated": True,
                "message_id": client_message_id,
                "feedback": feedback.value if feedback else None
            }
                    
        except MessageUpdateFailedException:
            raise
        except Exception as e:
            raise MessageUpdateFailedException(
                message="Failed to update message feedback",
                note=f"client_message_id={client_message_id}, feedback={feedback}, error={str(e)}"
            )
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.DATABASE.value)
    async def delete_by_id(cls, client_message_id: str, user_id: str) -> dict:
        """
        Delete a message by its client ID.
        
        Args:
            client_message_id: The frontend-generated message ID (from AI SDK)
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with deletion info: {
                "message_deleted": bool,
                "deleted_count": int  # Number of documents deleted (0 or 1)
            }
            
        Raises:
            MessageDeletionFailedException: If deletion fails
        
        Traced as INTERNAL span for database operation.
        """
        try:
            message = await cls.get_by_client_id(client_message_id, user_id)
            
            if not message:
                return {
                    "message_deleted": False,
                    "deleted_count": 0
                }
            
            delete_result = await message.delete()
            
            # delete_result is a DeleteResult with deleted_count
            deleted_count = delete_result.deleted_count if delete_result else 0
            
            return {
                "message_deleted": deleted_count > 0,
                "deleted_count": deleted_count
            }
                    
        except Exception as e:
            raise MessageDeletionFailedException(
                message="Failed to delete message by client ID",
                note=f"client_message_id={client_message_id}, error={str(e)}"
            ) 