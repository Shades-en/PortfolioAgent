from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from ai_server.api.dto.chat import ChatRequest
from ai_server.api.services import ChatService
from ai_server.utils.tracing import trace_context, add_graph_attributes, pop_graph_node
from ai_server.constants import (
    STREAM_EVENT_DATA_SESSION,
    STREAM_EVENT_ERROR,
    STREAM_DONE_SENTINEL,
    STREAM_HEADER_NAME,
    STREAM_HEADER_VERSION,
)

import asyncio
import json

router = APIRouter()

@router.post("/chat", tags=["Chat"])
async def chat(chat_request: ChatRequest):
    """
    Chat endpoint with request-scoped tracing context and agent graph visualization.
    
    Sets business context using trace_context manager.
    Middleware captures all request data as span attributes.
    Creates root orchestrator node for agent graph visualization.
    """
    span = trace.get_current_span()
    if span.is_recording():
        add_graph_attributes(span, node_id="chat_orchestrator")
    try:
        # Set business context
        async with trace_context(
            query=chat_request.query,
            session_id=chat_request.session_id,
            user_id=chat_request.user_id,
            user_cookie=chat_request.user_cookie,
            turn_number=chat_request.turn_number,
            new_chat=chat_request.new_chat,
            new_user=chat_request.new_user
        ):
            return await ChatService.chat(
                query=chat_request.query,
                session_id=chat_request.session_id,
                user_id=chat_request.user_id,
                user_cookie=chat_request.user_cookie,
                turn_number=chat_request.turn_number,
                new_chat=chat_request.new_chat,
                new_user=chat_request.new_user
            )
    finally:
        # Pop orchestrator from stack
        pop_graph_node()


@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(chat_request: ChatRequest):
    span = trace.get_current_span()
    if span.is_recording():
        add_graph_attributes(span, node_id="chat_orchestrator")

    queue: asyncio.Queue = asyncio.Queue()

    async def stream_callback(event):
        await queue.put(event)

    async def run_chat():
        try:
            async with trace_context(
                query=chat_request.query,
                session_id=chat_request.session_id,
                user_id=chat_request.user_id,
                user_cookie=chat_request.user_cookie,
                turn_number=chat_request.turn_number,
                new_chat=chat_request.new_chat,
                new_user=chat_request.new_user
            ):
                result = await ChatService.chat(
                    query=chat_request.query,
                    session_id=chat_request.session_id,
                    user_id=chat_request.user_id,
                    user_cookie=chat_request.user_cookie,
                    turn_number=chat_request.turn_number,
                    new_chat=chat_request.new_chat,
                    new_user=chat_request.new_user,
                    on_stream_event=stream_callback,
                )
                await queue.put({"type": STREAM_EVENT_DATA_SESSION, "data": result})
        except Exception as exc:
            await queue.put({"type": STREAM_EVENT_ERROR, "errorText": str(exc)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_chat())

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                payload = f"data: {json.dumps(event)}\n\n"
                yield payload
            yield f"data: {STREAM_DONE_SENTINEL}\n\n"
        finally:
            task.cancel()
            pop_graph_node()

    headers = {
        "Cache-Control": "no-cache",
        STREAM_HEADER_NAME: STREAM_HEADER_VERSION,
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
