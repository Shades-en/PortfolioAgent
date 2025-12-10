from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Mapping, Any
from opentelemetry import trace
from opentelemetry.trace import Tracer, Status, StatusCode, SpanKind

import time
import functools
import json

def _set_span_attributes(span: trace.Span, attributes: Mapping[str, Any]):
    for key, value in attributes.items():
        if value:
            span.set_attribute(key, value)

@contextmanager
def spanner(
    tracer: Tracer,
    *,
    name: Optional[str] = "AgentSpan",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    turn_number: Optional[str] = None,
    kind: str | SpanKind | OpenInferenceSpanKindValues = OpenInferenceSpanKindValues.UNKNOWN,
    input: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    with tracer.start_as_current_span(name) as span:
        _set_span_attributes(span, {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: (getattr(kind, "value", kind) if kind is not None else ""),
            SpanAttributes.INPUT_VALUE: input or "",
            "session_id": session_id or "",
            "user_id": user_id or "",
            "turn_number": turn_number or "",
            "created_at": time.time(),
            "metadata": json.dumps(metadata) if isinstance(metadata, Mapping) else (metadata if isinstance(metadata, (str, int, float, bool)) else ""),
        })
        try:
            yield
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        else:
            span.set_status(Status(StatusCode.OK))

@asynccontextmanager
async def async_spanner(
    tracer: Tracer,
    *,
    name: Optional[str] = "AgentSpan",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    turn_number: Optional[str] = None,
    kind: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    with tracer.start_as_current_span(name) as span:
        _set_span_attributes(span, {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: (getattr(kind, "value", kind) if kind is not None else ""),
            SpanAttributes.INPUT_VALUE: input or "",
            "session_id": session_id or "",
            "user_id": user_id or "",
            "turn_number": turn_number or "",
            "created_at": time.time(),
            "metadata": json.dumps(metadata) if isinstance(metadata, Mapping) else (metadata if isinstance(metadata, (str, int, float, bool)) else ""),
        })
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        else:
            # Set output attribute if provided
            if output is not None:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, output)
            span.set_status(Status(StatusCode.OK))

def trace_operation(
    name: Optional[str] = "AgentSpan",
    kind: Optional[str] = None,
    data: Optional[Mapping[str, Any]] = {
        "session_id": "",
        "user_id": "",
        "turn_number": "",
        "input": "",
    },
    metadata: Optional[Mapping[str, Any]] = None,
):
    def decorator(fn):
        tr = trace.get_tracer(fn.__module__)
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if metadata is not None:
                try:
                    meta = {**metadata, **kwargs}
                except Exception:
                    meta = metadata
            else:
                meta = kwargs
            with spanner(
                tracer=tr,
                name=name,
                session_id=data.get("session_id"),
                user_id=data.get("user_id"),
                turn_number=data.get("turn_number"),
                kind=kind,
                input=data.get("input"),
                metadata=meta,
            ):
                return fn(*args, **kwargs)
        return wrapper
    return decorator

__all__ = ["spanner", "async_spanner", "trace_operation"]
