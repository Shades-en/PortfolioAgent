"""Postgres-backed message repository extension with feedback support."""

from __future__ import annotations

from dataclasses import dataclass

from omniagent.persistence import MessageRepository

from ai_server.types import Feedback


@dataclass(slots=True)
class PostgresMessageRepositoryWithFeedback:
    base_repo: MessageRepository
    message_model: type

    def __getattr__(self, name: str):
        return getattr(self.base_repo, name)

    async def update_feedback_by_client_id(
        self,
        client_message_id: str,
        feedback: Feedback | None,
        client_id: str,
    ) -> dict:
        return await self.message_model.update_feedback_by_client_id(
            client_message_id,
            feedback,
            client_id=client_id,
        )
