from __future__ import annotations

from beanie import Document, Link
import pymongo
from bson import ObjectId

from pydantic import BaseModel, Field, model_validator
from typing import Self, TYPE_CHECKING, List
from enum import Enum
from datetime import datetime, timezone
import asyncio

from opentelemetry.trace import SpanKind

from ai_server.utils.general import get_token_count
from ai_server.utils.tracing import trace_operation, CustomSpanKinds
from ai_server.api.exceptions.db_exceptions import (
    MessageRetrievalFailedException,
    MessageDeletionFailedException
)
from ai_server.config import DEFAULT_MESSAGE_PAGE_SIZE

if TYPE_CHECKING:
    from ai_server.schemas.session import Session

class Role(Enum):
    HUMAN = 'human'
    SYSTEM = 'system'
    AI = 'ai'
    TOOL = 'tool'

class FunctionCallRequest(BaseModel):
    name: str
    arguments: dict

class Message(Document):
    role: Role
    # tool_call_id is given "null" when not exists because redis tag field does not accept None
    tool_call_id: str 
    metadata: dict
    content: str | None
    function_call: FunctionCallRequest | None
    token_count: int = 0
    error: bool = False
    session: Link[Session] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "messages"
        indexes = [
            [("session.$id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)]
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
            messages = await cls.find(
                cls.session.id == ObjectId(session_id)
            ).sort(
                -cls.created_at  # Descending to get most recent
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
            messages = await cls.find(
                cls.session.id == ObjectId(session_id)
            ).sort(
                +cls.created_at  # Ascending order (oldest first)
            ).to_list()
            
            return messages
        except Exception as e:
            raise MessageRetrievalFailedException(
                message="Failed to retrieve all messages for session",
                note=f"session_id={session_id}, error={str(e)}"
            )
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.DATABASE.value)
    async def delete_by_id(cls, message_id: str) -> dict:
        """
        Delete a message by its ID and remove its reference from any Turn.
        Uses a transaction to ensure atomicity.
        
        Args:
            message_id: The message ID to delete
            
        Returns:
            Dictionary with deletion info: {
                "message_deleted": bool,
                "turns_updated": int
            }
            
        Raises:
            MessageDeletionFailedException: If deletion fails
        
        Traced as INTERNAL span for database transaction.
        """
        from ai_server.schemas.turn import Turn
        from ai_server.db import MongoDB
        
        try:
            obj_id = ObjectId(message_id)
            message = await cls.get(obj_id)
            if not message:
                return {
                    "message_deleted": False,
                    "turns_updated": 0
                }
            
            client = MongoDB.get_client()
            
            async with client.start_session() as session_txn:
                async with await session_txn.start_transaction():
                    # Delete message and remove from turn in parallel
                    # Query turn that contains this message in its messages array
                    message_delete, turn_update = await asyncio.gather(
                        message.delete(session=session_txn),
                        Turn.find(
                            {"messages.$id": obj_id}
                        ).update(
                            {"$pull": {"messages.$id": obj_id}},
                            session=session_txn
                        )
                    )
                    
                    # turn_update.modified_count tells us if a turn was updated
                    turns_updated = turn_update.modified_count if turn_update else 0
                    
                    return {
                        "message_deleted": True,
                        "turns_updated": turns_updated
                    }
                    
        except Exception as e:
            raise MessageDeletionFailedException(
                message="Failed to delete message by ID",
                note=f"message_id={message_id}, error={str(e)}"
            ) 