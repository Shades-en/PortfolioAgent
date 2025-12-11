from contextlib import asynccontextmanager
from opentelemetry.trace import SpanKind

from ai_server.db import MongoDB
from ai_server.utils.tracing import trace_operation

from fastapi import FastAPI


@trace_operation(kind=SpanKind.INTERNAL)
async def _startup():
    """Initialize MongoDB connection."""
    await MongoDB.init()


@trace_operation(kind=SpanKind.INTERNAL)
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