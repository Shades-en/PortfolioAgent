from contextlib import asynccontextmanager
from opentelemetry.trace import SpanKind

from omniagent.session import MongoSessionManager
from ai_server.utils.tracing import trace_operation
from ai_server.constants import SERVER_STARTUP, SERVER_SHUTDOWN, SERVER
from ai_server.config import DEV_MODE

from fastapi import FastAPI


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_STARTUP)
async def _startup():
    """Initialize MongoDB connection via OmniAgent."""
    # In DEV_MODE, allow dropping indexes to handle schema changes
    await MongoSessionManager.initialize(allow_index_dropping=DEV_MODE)


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=SERVER, category=SERVER_SHUTDOWN)
async def _shutdown():
    """Close MongoDB connection via OmniAgent."""
    await MongoSessionManager.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan context manager.
    Handles startup and shutdown events.
    """
    await _startup()
    yield
    await _shutdown()