"""Persistence extension repositories for ai_server."""

from __future__ import annotations

from omniagent.persistence import (
    PersistenceBackend,
    RepositoryOverrides,
    SharedMessageRepository,
    SharedSessionRepository,
)

from ai_server.persistence.extensions.contracts import (
    MessageRepositoryWithFeedbackProtocol,
    SessionRepositoryWithFavoritesProtocol,
)
from ai_server.persistence.extensions.mongo.feedback import MongoMessageRepositoryWithFeedback
from ai_server.persistence.extensions.mongo.favorites import MongoSessionRepositoryWithFavorites
from ai_server.persistence.extensions.postgres.feedback import PostgresMessageRepositoryWithFeedback
from ai_server.persistence.extensions.postgres.favorites import PostgresSessionRepositoryWithFavorites


def _is_postgres_model(model: type) -> bool:
    return ".postgres." in model.__module__


def _is_mongo_model(model: type) -> bool:
    return ".mongo." in model.__module__


def _resolve_backend(backend: PersistenceBackend | str) -> PersistenceBackend:
    if isinstance(backend, PersistenceBackend):
        return backend
    try:
        return PersistenceBackend(backend)
    except ValueError as exc:
        raise ValueError(f"Unsupported persistence backend for repository overrides: {backend}") from exc


def build_repository_overrides(
    *,
    backend: PersistenceBackend | str,
    session_model: type,
    message_model: type,
) -> RepositoryOverrides:
    """Build init-time persistence repository wrappers for ai_server."""

    resolved_backend = _resolve_backend(backend)

    if resolved_backend is PersistenceBackend.POSTGRES:
        if not _is_postgres_model(session_model) or not _is_postgres_model(message_model):
            raise ValueError(
                "Postgres repository overrides require postgres models for both session_model and message_model."
            )

        sessions_repository = PostgresSessionRepositoryWithFavorites(
            base_repo=SharedSessionRepository(session_model=session_model),
            session_model=session_model,
        )
        messages_repository = PostgresMessageRepositoryWithFeedback(
            base_repo=SharedMessageRepository(message_model=message_model),
            message_model=message_model,
        )

    elif resolved_backend is PersistenceBackend.MONGO:
        if not _is_mongo_model(session_model) or not _is_mongo_model(message_model):
            raise ValueError(
                "Mongo repository overrides require mongo models for both session_model and message_model."
            )

        sessions_repository = MongoSessionRepositoryWithFavorites(
            base_repo=SharedSessionRepository(session_model=session_model),
            session_model=session_model,
        )
        messages_repository = MongoMessageRepositoryWithFeedback(
            base_repo=SharedMessageRepository(message_model=message_model),
            message_model=message_model,
        )
    else:
        raise ValueError(f"Unsupported persistence backend for repository overrides: {resolved_backend.value}")

    return RepositoryOverrides(
        sessions=sessions_repository,
        messages=messages_repository,
    )


__all__ = [
    "SessionRepositoryWithFavoritesProtocol",
    "MessageRepositoryWithFeedbackProtocol",
    "MongoSessionRepositoryWithFavorites",
    "MongoMessageRepositoryWithFeedback",
    "PostgresSessionRepositoryWithFavorites",
    "PostgresMessageRepositoryWithFeedback",
    "build_repository_overrides",
]
