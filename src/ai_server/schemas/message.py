from __future__ import annotations

from beanie import Document, Link
import pymongo
from bson import ObjectId

from pydantic import Field, model_validator
from typing import Self, TYPE_CHECKING, List
from datetime import datetime, timezone

from opentelemetry.trace import SpanKind

from ai_server.utils.general import get_token_count
from ai_server.utils.tracing import trace_operation, CustomSpanKinds
from ai_server.api.exceptions.db_exceptions import (
    MessageRetrievalFailedException,
    MessageDeletionFailedException
)
from ai_server.config import DEFAULT_MESSAGE_PAGE_SIZE, MAX_TURNS_TO_FETCH
from ai_server.types.message import Role, FunctionCallRequest

if TYPE_CHECKING:
    from ai_server.schemas.session import Session
    from ai_server.schemas.summary import Summary

class Message(Document):
    role: Role
    # tool_call_id is given "null" when not exists because redis tag field does not accept None
    tool_call_id: str 
    metadata: dict
    content: str | None
    function_call: FunctionCallRequest | None
    token_count: int = 0
    previous_summary: Link[Summary] | None = None
    turn_number: int = 1
    error: bool = False
    session: Link[Session]
    order: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "messages"
        indexes = [
            # For chronological queries and pagination
            [
                ("session._id", pymongo.ASCENDING), 
                ("created_at", pymongo.DESCENDING), 
                ("order", pymongo.ASCENDING)
            ],
            # For turn-based range queries (get_latest_by_session) with role tie-breaker
            [
                ("session._id", pymongo.ASCENDING),
                ("turn_number", pymongo.ASCENDING),
                ("created_at", pymongo.ASCENDING),
                ("order", pymongo.ASCENDING)
            ]
        ]
    
    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from content if not explicitly provided."""
        if self.token_count == 0 and self.content:
            self.token_count = get_token_count(self.content)
        return self
    
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
            # Use cls.session._id for querying Link fields
            messages = await cls.find(
                cls.session._id == ObjectId(session_id)
            ).sort(
                -cls.created_at,  # Descending to get most recent
                -cls.order  # Ensure ai messages precede human when timestamps match
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
                +cls.order  # Ensure human messages precede ai when timestamps match
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
                +cls.order  # Ensure human messages precede ai when timestamps match
            ).to_list()
            
            # Validate: ensure current_turn_number is greater than latest fetched turn
            if messages:
                latest_fetched_turn = max(msg.turn_number for msg in messages)
                if current_turn_number != latest_fetched_turn + 1:
                    raise MessageRetrievalFailedException(
                        message="Invalid current_turn_number: must be exactly one greater than latest turn in database",
                        note=f"session_id={session_id}, current_turn_number={current_turn_number}, latest_fetched_turn={latest_fetched_turn}"
                    )
            
            return messages
        except MessageRetrievalFailedException:
            raise
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to retrieve latest messages for session",
                note=f"session_id={session_id}, max_turns={max_turns}, current_turn_number={current_turn_number}, error={str(e)}"
            )
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.DATABASE.value)
    async def delete_by_id(cls, message_id: str) -> dict:
        """
        Delete a message by its ID.
        
        Args:
            message_id: The message ID to delete
            
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
            obj_id = ObjectId(message_id)
            message = await cls.get(obj_id)
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
                message="Failed to delete message by ID",
                note=f"message_id={message_id}, error={str(e)}"
            ) 