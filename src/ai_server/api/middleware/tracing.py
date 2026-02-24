"""
Generic ASGI tracing middleware for capturing request metadata.

This middleware creates one server span per request, preserves the span through
the full response lifecycle (including streaming), and propagates correlation
headers (`x-request-id`, `x-trace-id`, `traceparent`) back to clients.
"""

from __future__ import annotations

import json
from uuid import uuid4
from urllib.parse import parse_qsl

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from starlette.datastructures import URL
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_server.constants import (
    EMPTY_TRACE_HEADER_VALUE,
    HEADER_REQUEST_ID,
    HEADER_TRACE_ID,
    HEADER_TRACEPARENT,
    REQUEST_ID_SCOPE_KEY,
    REQUEST_ID_SPAN_ATTRIBUTE,
    SERVER,
    TRACEPARENT_VERSION,
)


class GenericTracingMiddleware:
    """
    Generic middleware to add request attributes to tracing spans.

    Captures request data (body, query params, path params, headers) and sets
    response tracing headers for both standard and streaming responses.
    """

    def __init__(self, app: ASGIApp):
        self.app = app
        self.tracer = trace.get_tracer(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_headers = self._decode_headers(scope)
        request_id = request_headers.get(HEADER_REQUEST_ID, "").strip() or str(uuid4())
        self._set_scope_request_id(scope=scope, request_id=request_id)

        parent_context = extract(request_headers)
        request_body = await self._read_full_body(receive)
        method = scope.get("method", "")
        url = URL(scope=scope)

        with self.tracer.start_as_current_span(
            f"{method} {scope.get('path', '')}",
            kind=SpanKind.SERVER,
            context=parent_context,
        ) as span:
            response_status_code: int | None = None

            try:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SERVER)
                span.set_attribute(REQUEST_ID_SPAN_ATTRIBUTE, request_id)
                span.set_attribute("http.method", method)
                span.set_attribute("http.url", str(url))
                span.set_attribute("http.route", scope.get("path", ""))
                span.set_attribute("http.scheme", url.scheme)
                span.set_attribute("http.host", url.hostname or "")

                user_agent = request_headers.get("user-agent")
                if user_agent:
                    span.set_attribute("http.user_agent", user_agent)

                input_data = self._build_input_data(scope=scope, method=method, body=request_body)
                if input_data:
                    self._set_input_attributes(span=span, input_data=input_data)
                    span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(input_data))

                replay_receive = self._make_replay_receive(request_body)

                async def send_wrapper(message: Message) -> None:
                    nonlocal response_status_code

                    if message["type"] == "http.response.start":
                        response_status_code = int(message.get("status", 500))
                        span.set_attribute("http.status_code", response_status_code)

                        response_headers = list(message.get("headers", []))
                        response_headers = self._upsert_header(
                            response_headers, HEADER_REQUEST_ID, request_id
                        )

                        trace_id_header, traceparent_header = self._build_trace_headers(span)
                        response_headers = self._upsert_header(
                            response_headers, HEADER_TRACE_ID, trace_id_header
                        )
                        response_headers = self._upsert_header(
                            response_headers, HEADER_TRACEPARENT, traceparent_header
                        )

                        message = {
                            **message,
                            "headers": response_headers,
                        }

                    await send(message)

                await self.app(scope, replay_receive, send_wrapper)

                if response_status_code is not None and response_status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response_status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.set_attribute("http.status_code", 500)
                span.set_attribute("error.type", type(exc).__name__)
                error_message = str(exc) if str(exc) else type(exc).__name__
                span.set_status(Status(StatusCode.ERROR, error_message))
                span.record_exception(exc)
                raise

    @staticmethod
    def _decode_headers(scope: Scope) -> dict[str, str]:
        return {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }

    @staticmethod
    def _set_scope_request_id(scope: Scope, request_id: str) -> None:
        state_obj = scope.setdefault("state", {})
        if isinstance(state_obj, dict):
            state_obj[REQUEST_ID_SCOPE_KEY] = request_id
            return
        setattr(state_obj, REQUEST_ID_SCOPE_KEY, request_id)

    @staticmethod
    async def _read_full_body(receive: Receive) -> bytes:
        body_chunks: list[bytes] = []
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                break
            if message_type != "http.request":
                continue

            body = message.get("body", b"")
            if body:
                body_chunks.append(body)

            if not message.get("more_body", False):
                break

        return b"".join(body_chunks)

    @staticmethod
    def _make_replay_receive(body: bytes) -> Receive:
        queue: list[Message] = [
            {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        ]

        async def replay_receive() -> Message:
            if queue:
                return queue.pop(0)
            return {"type": "http.disconnect"}

        return replay_receive

    @staticmethod
    def _build_input_data(scope: Scope, method: str, body: bytes) -> dict[str, object]:
        input_data: dict[str, object] = {}

        query_string = scope.get("query_string", b"").decode("latin-1")
        query_params = dict(parse_qsl(query_string, keep_blank_values=True))
        if query_params:
            input_data["query_params"] = query_params

        path_params = scope.get("path_params") or {}
        if path_params:
            input_data["path_params"] = dict(path_params)

        if method in {"POST", "PUT", "PATCH"} and body:
            try:
                input_data["body"] = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

        return input_data

    @staticmethod
    def _set_input_attributes(span: Span, input_data: dict[str, object]) -> None:
        query_params = input_data.get("query_params")
        if isinstance(query_params, dict):
            for key, value in query_params.items():
                span.set_attribute(f"http.query.{key}", str(value))

        path_params = input_data.get("path_params")
        if isinstance(path_params, dict):
            for key, value in path_params.items():
                span.set_attribute(f"http.path.{key}", str(value))

        body = input_data.get("body")
        if isinstance(body, dict):
            for key, value in body.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"request.body.{key}", value)
                else:
                    span.set_attribute(f"request.body.{key}", json.dumps(value))

    @staticmethod
    def _upsert_header(headers: list[tuple[bytes, bytes]], key: str, value: str) -> list[tuple[bytes, bytes]]:
        normalized_key = key.lower().encode("latin-1")
        encoded_value = value.encode("latin-1")

        filtered_headers = [(k, v) for k, v in headers if k.lower() != normalized_key]
        filtered_headers.append((normalized_key, encoded_value))
        return filtered_headers

    @staticmethod
    def _build_trace_headers(span: Span) -> tuple[str, str]:
        span_context = span.get_span_context()
        if not span_context.is_valid:
            return EMPTY_TRACE_HEADER_VALUE, EMPTY_TRACE_HEADER_VALUE

        trace_id = f"{span_context.trace_id:032x}"
        traceparent = (
            f"{TRACEPARENT_VERSION}-{trace_id}-{span_context.span_id:016x}-"
            f"{int(span_context.trace_flags):02x}"
        )
        return trace_id, traceparent
