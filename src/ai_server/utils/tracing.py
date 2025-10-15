from openinference.semconv.trace import SpanAttributes
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Mapping, Any
from opentelemetry import trace
from opentelemetry.trace import Tracer, Status, StatusCode

@contextmanager
def spanner(
    tracer: Tracer,
    query: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    kind: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    with tracer.start_as_current_span(
        "spanner",
        attributes={
            "app.query": query or "",
            "app.session_id": session_id or "",
            "app.user_id": user_id or "",
            "app.turn_id": turn_id or "",
            "app.kind": kind or "",
            "app.metadata": metadata or {},
        },
    ):
        yield

# COMMON

# Query
# Session ID
# User ID
# Turn ID
# Time
# Kind
# Input
# Metadata


