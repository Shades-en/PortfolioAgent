"""Persistence extension repositories for ai_server."""

from __future__ import annotations

from omniagent.persistence import MessageRepository, RepositoryOverrides, SessionRepository

from ai_server.persistence.extensions.contracts import (
    MessageRepositoryWithFeedbackProtocol,
    SessionRepositoryWithFavoritesProtocol,
)
from ai_server.persistence.extensions.mongo_feedback import MongoMessageRepositoryWithFeedback
from ai_server.persistence.extensions.mongo_favorites import MongoSessionRepositoryWithFavorites


def build_repository_overrides(
    *,
    session_model: type,
    message_model: type,
) -> RepositoryOverrides:
    """Build init-time persistence repository wrappers for ai_server."""

    def _wrap_sessions(base_repo: SessionRepository) -> SessionRepository:
        return MongoSessionRepositoryWithFavorites(
            base_repo=base_repo,
            session_model=session_model,
        )

    def _wrap_messages(base_repo: MessageRepository) -> MessageRepository:
        return MongoMessageRepositoryWithFeedback(
            base_repo=base_repo,
            message_model=message_model,
        )

    return RepositoryOverrides(
        sessions=_wrap_sessions,
        messages=_wrap_messages,
    )


__all__ = [
    "SessionRepositoryWithFavoritesProtocol",
    "MessageRepositoryWithFeedbackProtocol",
    "MongoSessionRepositoryWithFavorites",
    "MongoMessageRepositoryWithFeedback",
    "build_repository_overrides",
]
