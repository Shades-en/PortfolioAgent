from fastapi import APIRouter
from opentelemetry import trace

from ai_server.api.dto.chat import ChatRequest
from ai_server.api.services import ChatService
from ai_server.utils.tracing import trace_context, add_graph_attributes, pop_graph_node

router = APIRouter()

@router.post("/chat", tags=["Chat"])
async def chat(chat_request: ChatRequest):
    """
    Chat endpoint with request-scoped tracing context and agent graph visualization.
    
    Sets business context using trace_context manager.
    Middleware captures all request data as span attributes.
    Creates root orchestrator node for agent graph visualization.
    """
    tracer = trace.get_tracer(__name__)
    
    # Create root orchestrator span for agent graph
    with tracer.start_as_current_span("chat_orchestrator") as span:
        # Add root graph node (automatically becomes parent for children)
        add_graph_attributes(span, "chat_orchestrator")
        
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
