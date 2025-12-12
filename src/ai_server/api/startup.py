from contextlib import asynccontextmanager
from opentelemetry.trace import SpanKind

from ai_server.db import MongoDB
from ai_server.utils.tracing import trace_operation, CustomSpanKinds
from ai_server.constants import SERVER_STARTUP, SERVER_SHUTDOWN
from ai_server.config import DEV_MODE

from fastapi import FastAPI


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.SERVER.value, category=SERVER_STARTUP)
async def _startup():
    """Initialize MongoDB connection."""
    # In DEV_MODE, allow dropping indexes to handle schema changes
    await MongoDB.init(allow_index_dropping=DEV_MODE)


@trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.SERVER.value, category=SERVER_SHUTDOWN)
async def _shutdown():
    """Close MongoDB connection."""
    await MongoDB.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan context manager.
    Handles startup and shutdown events.
    """
    await _startup()
    yield
    await _shutdown()