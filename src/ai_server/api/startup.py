from contextlib import asynccontextmanager
from opentelemetry.trace import SpanKind

from omniagent.db.document_models import DocumentModels
from omniagent.persistence.backends.mongo import MongoBackendAdapter
from omniagent.schemas.mongo import User, Session, Summary
from ai_server.utils.tracing import trace_operation
from ai_server.constants import SERVER_STARTUP, SERVER_SHUTDOWN, SERVER, SESSION_BACKEND_MONGO
from ai_server.config import DEV_MODE, SESSION_BACKEND
from ai_server.schemas import CustomMessage

from fastapi import FastAPI


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_STARTUP)
async def _startup():
    """Initialize Mongo persistence backend."""
    if SESSION_BACKEND != SESSION_BACKEND_MONGO:
        raise ValueError(f"Unsupported SESSION_BACKEND '{SESSION_BACKEND}'. Expected '{SESSION_BACKEND_MONGO}'.")

    # Preserve existing Mongo model override behavior for ai_server.
    await MongoBackendAdapter.initialize(
        allow_index_dropping=DEV_MODE,
        models=DocumentModels(
            user=User,
            session=Session,
            summary=Summary,
            message=CustomMessage,
        ),
    )


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_SHUTDOWN)
async def _shutdown():
    """Close Mongo persistence backend connection."""
    await MongoBackendAdapter.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan context manager.
    Handles startup and shutdown events.
    """
    await _startup()
    yield
    await _shutdown()
