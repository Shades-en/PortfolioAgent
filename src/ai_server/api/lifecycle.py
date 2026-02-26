from contextlib import asynccontextmanager
from opentelemetry.trace import SpanKind

from omniagent.db.mongo import DocumentModels
from omniagent.persistence import (
    PersistenceBackend,
    initialize_persistence,
    shutdown_persistence,
)
from omniagent.schemas.mongo import User, Summary
from ai_server.utils.tracing import trace_operation
from ai_server.constants import (
    SERVER_STARTUP,
    SERVER_SHUTDOWN,
    SERVER,
)
from ai_server.config import DEV_MODE
from ai_server.persistence.extensions import (
    MessageRepositoryWithFeedbackProtocol,
    SessionRepositoryWithFavoritesProtocol,
    build_repository_overrides,
)
from ai_server.schemas import CustomMessage, CustomSession

from fastapi import FastAPI


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_STARTUP)
async def _startup():
    """Initialize persistence backend and mandatory domain repository overrides."""
    context = await initialize_persistence(
        backend=PersistenceBackend.MONGO,
        allow_index_dropping=DEV_MODE,
        models=DocumentModels(
            user=User,
            session=CustomSession,
            summary=Summary,
            message=CustomMessage,
        ),
        repository_overrides=build_repository_overrides(
            session_model=CustomSession,
            message_model=CustomMessage,
        ),
    )
    if not isinstance(context.repositories.sessions, SessionRepositoryWithFavoritesProtocol):
        raise RuntimeError(
            "Session repository override is missing required favorites behavior."
        )
    if not isinstance(context.repositories.messages, MessageRepositoryWithFeedbackProtocol):
        raise RuntimeError(
            "Message repository override is missing required feedback behavior."
        )


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_SHUTDOWN)
async def _shutdown():
    """Close Mongo persistence backend connection."""
    await shutdown_persistence()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan context manager.
    Handles startup and shutdown events.
    """
    await _startup()
    yield
    await _shutdown()
