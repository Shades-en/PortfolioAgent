from __future__ import annotations

from beanie import Document, Link
import pymongo
from bson import ObjectId

from datetime import datetime, timezone
from pydantic import Field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_server.schemas.session import Session

from ai_server.api.exceptions.db_exceptions import SummaryRetrievalFailedException


class Summary(Document):
    content: str
    token_count: int
    start_turn_number: int
    end_turn_number: int
    session: Link[Session]
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Settings:
        name = "summaries"
        indexes = [
            [("session.$id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)]
        ]
    
    @classmethod
    async def get_latest_by_session(cls, session_id: str) -> Summary | None:
        """Retrieve the latest summary for a session, ordered by created_at descending."""
        try:
            return await cls.find(
                cls.session.id == ObjectId(session_id)
            ).sort(-cls.created_at).first_or_none()
        except Exception as e:
            raise SummaryRetrievalFailedException(
                message="Failed to retrieve latest summary by session ID",
                note=f"session_id={session_id}, error={str(e)}"
            )