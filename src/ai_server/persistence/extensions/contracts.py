"""Consumer extension repository contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omniagent.persistence import (
    MessageRepository,
    SessionRecord,
    SessionRepository,
)

from ai_server.types import Feedback


@runtime_checkable
class SessionRepositoryWithFavoritesProtocol(SessionRepository, Protocol):
    async def list_starred_by_client_id(self, client_id: str) -> list[SessionRecord]:
        ...

    async def update_starred_by_client_id(self, session_id: str, starred: bool, client_id: str) -> dict:
        ...


@runtime_checkable
class MessageRepositoryWithFeedbackProtocol(MessageRepository, Protocol):
    async def update_feedback_by_client_id(self, client_message_id: str, feedback: Feedback | None, client_id: str) -> dict:
        ...
