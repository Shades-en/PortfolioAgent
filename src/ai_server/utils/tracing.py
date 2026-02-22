"""ai_server tracing adapters built on top of omniagent tracing utilities.

This module preserves ai_server-facing function signatures (notably `user_cookie`)
while delegating core tracing behavior to omniagent's canonical implementation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import omniagent.tracing as _base_tracing
from omniagent.tracing import (
    add_graph_attributes,
    pop_graph_node,
    trace_method,
    trace_operation,
)


def set_trace_context(
    query: str | None = None,
    session_id: str | None = None,
    user_cookie: str | None = None,
) -> None:
    """Set request-scoped tracing context for ai_server.

    ai_server uses `user_cookie`, which maps to omniagent's `user_client_id`.
    """
    _base_tracing.set_trace_context(
        query=query,
        session_id=session_id,
        user_client_id=user_cookie,
    )


def get_trace_context() -> dict[str, Any]:
    """Get current tracing context with ai_server compatibility key."""
    ctx = _base_tracing.get_trace_context()
    if "user_cookie" not in ctx and "user_client_id" in ctx:
        return {**ctx, "user_cookie": ctx.get("user_client_id")}
    return ctx


def clear_trace_context() -> None:
    """Clear request-scoped tracing context."""
    _base_tracing.clear_trace_context()


@asynccontextmanager
async def trace_context(
    query: str | None = None,
    session_id: str | None = None,
    user_cookie: str | None = None,
) -> AsyncIterator[None]:
    """Context manager that keeps ai_server's `user_cookie` API."""
    async with _base_tracing.trace_context(
        query=query,
        session_id=session_id,
        user_client_id=user_cookie,
    ):
        yield


__all__ = [
    "set_trace_context",
    "get_trace_context",
    "clear_trace_context",
    "trace_context",
    "add_graph_attributes",
    "pop_graph_node",
    "trace_method",
    "trace_operation",
]
