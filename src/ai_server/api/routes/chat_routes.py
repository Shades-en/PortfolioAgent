from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from ai_server.api.dto.chat import ChatRequest, CancelChatRequest
from ai_server.api.services import ChatService
from ai_server.utils.tracing import trace_context, add_graph_attributes, pop_graph_node
from omniagent.utils import cancel_task, get_streaming_headers

router = APIRouter()

@router.post("/chat", tags=["Chat"])
async def chat(chat_request: ChatRequest, response: Response):
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
        # Use the shared omniagent helper for Vercel AI SDK-compatible headers
        response.headers.update(get_streaming_headers())

        # Set business context
        async with trace_context(
            query=chat_request.query_message.query,
            session_id=chat_request.session_id,
            user_cookie=chat_request.user_cookie,
        ):
            return await ChatService.chat(
                query_message=chat_request.query_message,
                session_id=chat_request.session_id,
                user_cookie=chat_request.user_cookie,
            )
    finally:
        # Pop orchestrator from stack
        pop_graph_node()


@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(chat_request: ChatRequest):
    """
    Streaming chat endpoint using omniagent's run_stream API.
    
    Returns SSE stream with formatted events. Queue management, event formatting,
    and task registration are all handled by omniagent internally.
    """
    span = trace.get_current_span()
    if span.is_recording():
        add_graph_attributes(span, node_id="chat_orchestrator")

    async def event_generator():
        try:
            async with trace_context(
                query=chat_request.query_message.query,
                session_id=chat_request.session_id,
                user_cookie=chat_request.user_cookie,
            ):
                # Get stream generator and result future from omniagent
                stream_gen, _ = await ChatService.chat_stream(
                    query_message=chat_request.query_message,
                    session_id=chat_request.session_id,
                    user_cookie=chat_request.user_cookie,
                )
                async for event in stream_gen:
                    yield event
        finally:
            pop_graph_node()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=get_streaming_headers()
    )


@router.post("/chat/cancel", tags=["Chat"])
async def cancel_chat(request: CancelChatRequest):
    """
    Cancel an in-progress chat stream.
    
    Args:
        request: Contains session_id of the chat to cancel
    
    Returns:
        Dictionary with cancellation status:
        - cancelled: True if task was found and cancelled, False otherwise
    """
    cancelled = cancel_task(request.session_id)
    return {"cancelled": cancelled}
