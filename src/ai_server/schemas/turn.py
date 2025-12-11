from __future__ import annotations

from typing import List, TYPE_CHECKING
from beanie import Document, Link
import pymongo
from bson import ObjectId

if TYPE_CHECKING:
    from ai_server.schemas.message import Message
    from ai_server.schemas.summary import Summary
    from ai_server.schemas.session import Session
from ai_server.api.exceptions.db_exceptions import TurnRetrievalFailedException


class Turn(Document):
    turn_number: int
    session: Link[Session]
    turn_token_count: int
    previous_summary: Link[Summary]
    messages: List[Link[Message]]

    class Settings:
        name = "turns"
        indexes = [
            [("session.$id", pymongo.ASCENDING), ("turn_number", pymongo.ASCENDING)]
        ]
    
    @classmethod
    async def get_latest_by_session(cls, session_id: str, limit: int = 100) -> List[dict]:
        """Retrieve the latest turns for a session with messages using aggregation pipeline."""
        try:
            pipeline = [
                {"$match": {"session.$id": ObjectId(session_id)}},
                {"$sort": {"turn_number": -1}},
                {"$limit": limit},
                {"$lookup": {
                    "from": "messages",
                    "let": {"message_ids": "$messages.$id"},
                    "pipeline": [
                        {"$match": {"$expr": {"$in": ["$_id", "$$message_ids"]}}},
                        {"$sort": {"created_at": 1}}  # Sort by created_at ascending (chronological)
                    ],
                    "as": "messages"
                }}
            ]
            # Turns are returned in descending order with message inside each turn in ascending order
            return await cls.aggregate(pipeline).to_list()
        except Exception as e:
            raise TurnRetrievalFailedException(
                message="Failed to retrieve turns by session ID",
                note=f"session_id={session_id}, error={str(e)}"
            )
