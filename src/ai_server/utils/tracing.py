from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from functools import wraps
from contextvars import ContextVar
from contextlib import asynccontextmanager
from enum import Enum
from datetime import datetime, date

import asyncio
import time
import json
from typing import Mapping, Any, Callable
import logging

from beanie import PydanticObjectId
from bson import ObjectId

logger = logging.getLogger(__name__)

# ============================================================================
# Context Variables for Request-Scoped Tracing
# ============================================================================

# Global context variable for tracing context (isolated per async request)
_trace_context: ContextVar[dict[str, Any]] = ContextVar('trace_context', default={})

# Stack to track graph node hierarchy for automatic parent detection
_graph_node_stack: ContextVar[list[str]] = ContextVar('graph_node_stack', default=[])


def set_trace_context(
    query: str | None = None,
    session_id: str | None = None,
    user_cookie: str | None = None,
) -> None:
    """
    Set tracing context for the current async execution context.
    
    This should be called once at the entry point of a request (e.g., ChatService.chat).
    The context will automatically propagate through all async calls within the same request.
    
    Args:
        query: User's input query/message
        session_id: MongoDB session ID
        user_cookie: User's cookie ID
    
    Example:
        >>> set_trace_context(
        ...     query="What's the weather like?",
        ...     session_id="507f1f77bcf86cd799439011",
        ...     user_cookie="abc123",
        ... )
    
    Note:
        Each async request gets its own isolated context via ContextVar.
        Concurrent requests will not interfere with each other.
    """
    _trace_context.set({
        "query": query,
        "session_id": session_id,
        "user_cookie": user_cookie,
    })


def get_trace_context() -> dict[str, Any]:
    """
    Get the current tracing context for this async execution.
    
    Returns:
        Dictionary containing query, session_id, user_cookie.
        Returns empty dict if no context has been set.
    
    Example:
        >>> ctx = get_trace_context()
        >>> print(ctx["query"])
        "What's the weather like?"
        >>> print(ctx["session_id"])
        '507f1f77bcf86cd799439011'
    """
    return _trace_context.get()


def clear_trace_context() -> None:
    """
    Clear the tracing context for the current async execution.
    
    This is optional - ContextVars are automatically cleaned up when the async context ends.
    Use this if you need explicit cleanup for testing or special cases.
    """
    _trace_context.set({})


@asynccontextmanager
async def trace_context(
    query: str | None = None,
    session_id: str | None = None,
    user_cookie: str | None = None,
):
    """
    Context manager for tracing context lifecycle.
    
    Sets the tracing context on entry and clears it on exit.
    Provides explicit lifecycle management with guaranteed cleanup.
    
    Args:
        query: User's input query/message
        session_id: MongoDB session ID
        user_cookie: User's cookie ID
    
    Usage:
        >>> async with trace_context(
        ...     query="What's the weather?",
        ...     session_id="507f1f77bcf86cd799439011",
        ... ):
        ...     # Context is set here
        ...     result = await runner.run(query)
        ...     # Context automatically cleared on exit
    
    Example in ChatService:
        >>> @classmethod
        >>> async def chat(cls, query, session_id, user_id, ...):
        ...     async with trace_context(query, session_id, user_id, ...):
        ...         session_manager = SessionManager(...)
        ...         runner = Runner(...)
        ...         return await runner.run(query)
    """
    # Set context on entry
    set_trace_context(
        query=query,
        session_id=session_id,
        user_cookie=user_cookie,
    )
    
    try:
        yield
    finally:
        # Clear context on exit (guaranteed cleanup)
        clear_trace_context()


# ============================================================================
# Helper Functions
# ============================================================================

def _set_span_attributes(span: trace.Span, attributes: Mapping[str, Any]):
    """Set multiple attributes on a span, skipping None/empty values."""
    for key, value in attributes.items():
        if value:
            span.set_attribute(key, value)


def _serialize_for_json(obj):
    """
    Custom JSON serializer that handles enums, datetime, and other non-serializable types.
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (ObjectId, PydanticObjectId)):
        return str(obj)
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    else:
        return obj


def _attach_output_to_span(span: trace.Span, result):
    """Helper function to attach output to span if serializable."""
    try:
        if result is not None:
            output_value = None
            
            # Handle Pydantic models
            if hasattr(result, 'model_dump'):
                output_value = json.dumps(result.model_dump())
            # Handle list of Pydantic models
            elif isinstance(result, list) and all(hasattr(item, 'model_dump') for item in result):
                output_value = json.dumps([item.model_dump() for item in result])
            # Handle dict, list, primitives (with enum support)
            elif isinstance(result, (dict, list, str, int, float, bool)):
                # Use custom serializer to handle enums
                serialized = _serialize_for_json(result)
                output_value = json.dumps(serialized)
            
            # Set attribute if we have a value
            if output_value:
                # Truncate if too large (OpenTelemetry has attribute size limits)
                max_length = 10000  # 10KB limit
                if len(output_value) > max_length:
                    output_value = output_value[:max_length] + "... [truncated]"
                    logger.debug(f"Output truncated to {max_length} chars for span")
                
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
                logger.debug(f"Attached output to span: {len(output_value)} chars")
            else:
                logger.debug(f"Could not serialize output of type {type(result)}")
                
    except Exception as e:
        # If serialization fails, log and skip output attachment
        logger.warning(f"Failed to attach output to span: {e}")


def add_graph_attributes(
    span: trace.Span,
    node_id: str,
    parent_id: str | None = None,
    display_name: str | None = None
):
    """
    Add graph visualization attributes to a span for Arize agent visualization.
    
    This enables the Agent Graph & Path visualization in Arize, showing:
    - Agent execution flow
    - Parent-child relationships
    - Common execution paths
    - Agent loops and failures
    
    Automatically detects parent node from the graph node stack (ContextVar).
    
    Args:
        span: OpenTelemetry span to add attributes to
        node_id: Unique identifier for this node/agent (e.g., "chat_service", "agent_runner")
        parent_id: Parent node identifier (optional - auto-detected if not provided)
        display_name: Human-readable name (defaults to formatted node_id)
    
    Example:
        with tracer.start_as_current_span("operation") as span:
            add_graph_attributes(span, "agent_runner")  # Auto-detects parent
            # Your logic here
    
    Graph hierarchy example:
        orchestrator (root)
        ├── chat_service
        │   └── session_manager
        └── agent_runner
            ├── query_handler
            └── tool_weather
    
    Reference: https://arize.com/docs/ax/observe/tracing/agents
    """
    span.set_attribute("graph.node.id", node_id)
    
    # Auto-detect parent from graph node stack if not explicitly provided
    if parent_id is None:
        try:
            stack = _graph_node_stack.get()
            if stack and len(stack) > 0:
                # Parent is the last node in the stack
                parent_id = stack[-1]
        except LookupError:
            # No stack set yet (first node)
            pass
    
    # Set parent_id if we have one (either explicit or auto-detected)
    if parent_id:
        span.set_attribute("graph.node.parent_id", parent_id)
    
    # Set display name
    if display_name:
        span.set_attribute("graph.node.display_name", display_name)
    else:
        # Auto-generate display name from node_id
        span.set_attribute("graph.node.display_name", 
                         node_id.replace("_", " ").title())
    
    # Push current node to stack for children to use as parent
    try:
        stack = _graph_node_stack.get().copy()  # Copy to avoid mutation
    except LookupError:
        stack = []
    stack.append(node_id)
    _graph_node_stack.set(stack)


def pop_graph_node():
    """
    Pop the current node from the graph node stack.
    
    Should be called when exiting a span that has graph attributes.
    Usually handled automatically by the trace_method decorator.
    """
    try:
        stack = _graph_node_stack.get().copy()
        if stack:
            stack.pop()
            _graph_node_stack.set(stack)
    except (LookupError, IndexError):
        pass


def track_state_change(key: str, old_value: Any, new_value: Any) -> None:
    """
    Track a state change by adding attributes and events to the current span.
    
    Adds both attributes (for filtering/searching) and events (for timeline visualization)
    to help debug state mutations throughout the request lifecycle.
    
    Args:
        key: State key that changed (e.g., "turn_number", "total_tokens")
        old_value: Previous value before the change
        new_value: New value after the change
    
    Example:
        track_state_change("turn_number", 1, 2)
        # Adds to current span:
        #   - Attribute: state.turn_number.before = "1"
        #   - Attribute: state.turn_number.after = "2"
        #   - Event: state_updated (key=turn_number, old=1, new=2)
    
    Note:
        - Only tracks if a span is currently active and recording
        - Values are converted to strings for compatibility
        - Safe to call even if no span exists (no-op)
    """
    current_span = trace.get_current_span()
    
    if current_span and current_span.is_recording():
        # Add attributes for filtering and searching
        current_span.set_attribute(f"state.{key}.before", str(old_value))
        current_span.set_attribute(f"state.{key}.after", str(new_value))
        
        # Add event for timeline visualization
        current_span.add_event(
            "state_updated",
            attributes={
                "key": key,
                "old_value": str(old_value),
                "new_value": str(new_value)
            }
        )


# ============================================================================
# Tracing Decorators
# ============================================================================

def trace_method(
    name: str | None = None,
    kind=None,
    capture_input: bool = True,
    capture_output: bool = True,
    graph_node_id: str | Callable | None = None,
):
    """
    Decorator for tracing async methods with automatic context from ContextVar.
    
    Automatically reads tracing context (session_id, user_id, etc.) from the
    request-scoped ContextVar set by trace_context().
    
    Args:
        name: Custom span name (defaults to ClassName.method_name)
        kind: OpenInference span kind (AGENT, CHAIN, LLM, TOOL, etc.)
        capture_input: Whether to capture first argument as input (default: True)
        capture_output: Whether to capture return value as output (default: True)
        graph_node_id: Node ID for agent graph visualization (optional)
                       Can be a string or a callable that takes (self) and returns string
    
    Usage:
        # Static node ID
        @trace_method(
            kind=OpenInferenceSpanKindValues.AGENT,
            graph_node_id="agent_runner"
        )
        async def run(self, query: str):
            ...
        
        # Dynamic node ID from instance attribute
        @trace_method(
            kind=OpenInferenceSpanKindValues.AGENT,
            graph_node_id=lambda self: f"agent_{self.agent.__class__.__name__.lower()}"
        )
        async def run(self, query: str):
            # Node ID will be "agent_aboutmeagent" at runtime
            ...
    
    Example:
        class Runner:
            @trace_method(
                kind=OpenInferenceSpanKindValues.AGENT,
                graph_node_id=lambda self: self.agent.__class__.__name__
            )
            async def run(self, query: str):
                # Node ID dynamically set from agent name
                return await self._handle_query(query)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            tracer = trace.get_tracer(__name__)
            span_name = name or f"{self.__class__.__name__}.{func.__name__}"
            
            # Get context from ContextVar
            ctx = get_trace_context()
            
            # Extract input from first argument if requested
            input_value = None
            if capture_input and args:
                input_value = args[0] if isinstance(args[0], str) else ctx.get("query")
            elif capture_input:
                input_value = ctx.get("query")
            
            # Build metadata
            metadata = {
                "class": self.__class__.__name__,
                "method": func.__name__,
            }
            
            # Add agent metadata if available
            if hasattr(self, 'agent') and self.agent:
                metadata["agent_name"] = self.agent.__class__.__name__
                metadata["agent_description"] = self.agent.description
                metadata["tools"] = [tool.__class__.__name__ for tool in self.agent.tools]
            
            # Start span with context from ContextVar
            with tracer.start_as_current_span(span_name) as span:
                try:
                    _set_span_attributes(span, {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: (
                            getattr(kind, "value", kind) if kind is not None else ""
                        ),
                        SpanAttributes.INPUT_VALUE: input_value or "",
                        "session_id": ctx.get("session_id", ""),
                        "user_cookie": ctx.get("user_cookie", ""),
                        "turn_number": ctx.get("turn_number", ""),
                        "created_at": time.time(),
                        "metadata": json.dumps(metadata),
                    })
                    
                    # Add graph visualization attributes if provided
                    if graph_node_id:
                        # Resolve graph_node_id if it's a callable
                        resolved_node_id = graph_node_id(self) if callable(graph_node_id) else graph_node_id
                        add_graph_attributes(span, resolved_node_id)
                    
                    try:
                        result = await func(self, *args, **kwargs)
                        
                        # Attach output if requested
                        if capture_output:
                            _attach_output_to_span(span, result)
                        
                        span.set_status(Status(StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        error_msg = str(e) if str(e) else type(e).__name__
                        span.set_status(Status(StatusCode.ERROR, str(error_msg)))
                        span.record_exception(e)
                        raise
                
                finally:
                    # Pop graph node from stack when exiting span (not function!)
                    if graph_node_id:
                        pop_graph_node()
        
        return wrapper
    return decorator

def trace_operation(
    kind: str | None = None,
    open_inference_kind: str | None = None,
    category: str | None = None,
    capture_input: bool = False,
    capture_output: bool = False
):
    """
    Simple decorator for tracing CRUD operations and non-agent functions.
    
    Unlike trace_method, this decorator:
    - Does NOT require agent context
    - Does NOT use graph nodes
    - Does NOT push/pop from graph stack
    - Just creates a simple span with optional I/O capture
    - Automatically generates span name from class.method
    
    Args:
        kind: OpenTelemetry span kind (e.g., SpanKind.INTERNAL, SpanKind.SERVER)
        category: Custom span category for filtering (e.g., DATABASE, CACHE, SERVER)
        capture_input: Whether to capture function input as span attributes
        capture_output: Whether to capture function output as span attributes
    
    Example:
        >>> from ai_server.constants import DATABASE
        >>> @trace_operation(kind=SpanKind.INTERNAL, category=DATABASE)
        ... @classmethod
        ... async def delete_message(cls, message_id: str):
        ...     return await Message.delete_by_id(message_id)
        ... # Span name: "MessageService.delete_message"
        ... # Span kind: INTERNAL
        ... # Span category: "DATABASE"
    
    Use this for:
    - CRUD service methods
    - Database operations
    - Utility functions
    - Non-agent workflows
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            
            # Auto-generate span name from class.method or just method
            if args and hasattr(args[0], '__class__'):
                # Instance or class method
                class_name = args[0].__class__.__name__ if not isinstance(args[0], type) else args[0].__name__
                span_name = f"{class_name}.{func.__name__}"
            else:
                # Regular function
                span_name = func.__name__
            
            # Start span with optional kind
            span_kwargs = {"name": span_name}
            if kind:
                span_kwargs["kind"] = kind
            
            with tracer.start_as_current_span(**span_kwargs) as span:
                try:
                    # Set custom category if provided
                    if category:
                        span.set_attribute("span.category", category)
                    
                    # Capture input if requested
                    if capture_input:
                        span.set_attribute("input.args", str(args))
                        span.set_attribute("input.kwargs", str(kwargs))

                    if open_inference_kind:
                        # Extract value from enum if needed
                        kind_value = getattr(open_inference_kind, 'value', open_inference_kind)
                        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind_value)
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Capture output if requested
                    if capture_output:
                        span.set_attribute("output", str(result))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    error_msg = str(e) if str(e) else type(e).__name__
                    span.set_status(Status(StatusCode.ERROR, str(error_msg)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            
            # Auto-generate span name from class.method or just method
            if args and hasattr(args[0], '__class__'):
                # Instance or class method
                class_name = args[0].__class__.__name__ if not isinstance(args[0], type) else args[0].__name__
                span_name = f"{class_name}.{func.__name__}"
            else:
                # Regular function
                span_name = func.__name__
            
            # Start span with optional kind
            span_kwargs = {"name": span_name}
            if kind:
                span_kwargs["kind"] = kind
            
            with tracer.start_as_current_span(**span_kwargs) as span:
                try:
                    # Set custom category if provided
                    if category:
                        span.set_attribute("span.category", category)
                    
                    # Capture input if requested
                    if capture_input:
                        span.set_attribute("input.args", str(args))
                        span.set_attribute("input.kwargs", str(kwargs))
                    
                    # Set OpenInference span kind if provided
                    if open_inference_kind:
                        # Extract value from enum if needed
                        kind_value = getattr(open_inference_kind, 'value', open_inference_kind)
                        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind_value)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Capture output if requested
                    if capture_output:
                        span.set_attribute("output", str(result))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    error_msg = str(e) if str(e) else type(e).__name__
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class CustomSpanKinds(Enum):
    INIT = "INIT"
    DATABASE = "DATABASE"
    CACHE = "CACHE"
    VECTOR_INDEX = "VECTOR_INDEX"
    SERVER = "SERVER"


__all__ = [
    # Context management
    "trace_context",        # Context manager (recommended)
    "set_trace_context",    # Manual set (if needed)
    "get_trace_context",    # Get current context
    "clear_trace_context",  # Manual clear (if needed)
    # Decorators
    "trace_method",         # Trace instance methods (for agents)
    "trace_operation",      # Trace simple operations (for CRUD)
    # Graph visualization
    "add_graph_attributes", # Add agent graph attributes to spans
    "pop_graph_node",       # Pop graph node from stack
    # State tracking
    "track_state_change",   # Track state mutations in spans
    # Span kinds
    "CustomSpanKinds",     # Custom span kind enum
]
