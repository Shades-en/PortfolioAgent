"""Mongo-backed session repository extension with favorites support."""

from __future__ import annotations

from dataclasses import dataclass

from omniagent.persistence import SessionRecord, SessionRepository
from omniagent.persistence.backends.mongo.repositories import session_document_to_record


@dataclass(slots=True)
class MongoSessionRepositoryWithFavorites:
    base_repo: SessionRepository
    session_model: type

    def __getattr__(self, name: str):
        return getattr(self.base_repo, name)

    async def list_starred_by_client_id(self, client_id: str) -> list[SessionRecord]:
        sessions = await self.session_model.get_starred_by_user_client_id(client_id=client_id)
        return [session_document_to_record(session) for session in sessions]

    async def update_starred_by_client_id(self, session_id: str, starred: bool, client_id: str) -> dict:
        return await self.session_model.update_starred_by_client_id(
            session_id=session_id,
            starred=starred,
            client_id=client_id,
        )
