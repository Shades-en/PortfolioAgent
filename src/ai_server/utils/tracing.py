from openinference.semconv.trace import SpanAttributes
from contextlib import asynccontextmanager
from typing import Mapping, Any, Callable
from opentelemetry import trace
from opentelemetry.trace import Tracer, Status, StatusCode
from functools import wraps

import time
import json

from ai_server.utils.general import safe_get_arg

def _set_span_attributes(span: trace.Span, attributes: Mapping[str, Any]):
    for key, value in attributes.items():
        if value:
            span.set_attribute(key, value)


# @asynccontextmanager
# async def async_spanner(
#     tracer: Tracer,
#     *,
#     name: str | None = "AgentSpan",
#     session_id: str | None = None,
#     user_id: str | None = None,
#     user_cookie: str | None = None,
#     turn_number: str | None = None,
#     kind: str | None = None,
#     input: str | None = None,
#     new_chat: bool = False,
#     new_user: bool = False,
#     metadata: Mapping[str, Any] | None = None,
# ):
#     with tracer.start_as_current_span(name) as span:
#         _set_span_attributes(span, {
#             SpanAttributes.OPENINFERENCE_SPAN_KIND: (getattr(kind, "value", kind) if kind is not None else ""),
#             SpanAttributes.INPUT_VALUE: input or "",
#             "session_id": session_id or "",
#             "user_id": user_id or "",
#             "user_cookie": user_cookie or "",
#             "turn_number": turn_number or "",
#             "new_chat": new_chat,
#             "new_user": new_user,
#             "created_at": time.time(),
#             "metadata": json.dumps(metadata) if isinstance(metadata, Mapping) else (metadata if isinstance(metadata, (str, int, float, bool)) else ""),
#         })
#         try:
#             yield span
#         except Exception as e:
#             span.set_status(Status(StatusCode.ERROR, str(e)))
#             span.record_exception(e)
#             raise
#         else:
#             span.set_status(Status(StatusCode.OK))

# class AgentTracer:
#     """Helper class for tracing agent and runner operations with common context."""
    
#     def __init__(self, session_manager=None, agent=None):
#         """
#         Initialize the AgentTracer with session manager and agent context.
        
#         Args:
#             session_manager: SessionManager instance for extracting session/user context
#             agent: Agent instance for extracting agent metadata
#         """
#         self.session_manager = session_manager
#         self.agent = agent
#         self.tracer = trace.get_tracer(__name__)
    
#     def trace_execution(
#         self, 
#         name: str, 
#         kind, 
#         query: str | None = None, 
#         **extra_metadata
#     ):
#         """
#         Create a traced execution context with automatic session/user/agent context.
        
#         Args:
#             name: Name of the span
#             kind: OpenInference span kind (AGENT, CHAIN, LLM, TOOL, etc.)
#             query: Input query/message
#             **extra_metadata: Additional metadata to include in the span
        
#         Returns:
#             async_spanner context manager
#         """
#         metadata = {}
        
#         # Add agent metadata if available
#         if self.agent:
#             metadata["agent_name"] = self.agent.__class__.__name__
#             metadata["agent_description"] = self.agent.description
#             metadata["tools"] = [tool.__class__.__name__ for tool in self.agent.tools]
        
#         # Merge with extra metadata
#         metadata.update(extra_metadata)
        
#         # Extract session manager context
#         session_id = None
#         user_id = None
#         user_cookie = None
#         turn_number = None
#         new_chat = False
#         new_user = False
        
#         if self.session_manager:
#             session_id = str(self.session_manager.session.id) if self.session_manager.session else None
#             user_id = str(self.session_manager.user.id) if self.session_manager.user else None
#             user_cookie = self.session_manager.user.cookie_id if self.session_manager.user else None
#             turn_number = str(self.session_manager.state.turn_number)
#             new_chat = self.session_manager.state.new_chat
#             new_user = self.session_manager.state.new_user
        
#         return async_spanner(
#             tracer=self.tracer,
#             name=name,
#             kind=kind,
#             session_id=session_id,
#             user_id=user_id,
#             user_cookie=user_cookie,
#             turn_number=turn_number,
#             input=query,
#             new_chat=new_chat,
#             new_user=new_user,
#             metadata=metadata
#         )


# def _attach_output_to_span(span: trace.Span, result):
#     """Helper function to attach output to span if serializable."""
#     try:
#         if result is not None:
#             if hasattr(result, 'model_dump'):
#                 span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(result.model_dump()))
#             elif isinstance(result, list) and all(hasattr(item, 'model_dump') for item in result):
#                 span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps([item.model_dump() for item in result]))
#             elif isinstance(result, (list, dict, str, int, float, bool)):
#                 span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(result))
#     except Exception:
#         # If serialization fails, skip output attachment
#         pass


# def trace_classmethod(
#     name: str | None = None, 
#     kind=None, 
#     extract_input: bool = True,
#     session_id_index: int = 1,
#     user_id_index: int = 2,
#     user_cookie_index: int = 3,
#     turn_number_index: int = 4,
#     new_chat_index: int = 5,
#     new_user_index: int = 6,
# ):
#     """
#     Decorator for automatically tracing async classmethods with OpenTelemetry.
    
#     Extracts tracing context from method arguments by index position.
    
#     Args:
#         name: Custom span name (defaults to ClassName.method_name)
#         kind: OpenInference span kind (AGENT, CHAIN, etc.)
#         extract_input: Whether to extract first argument as input (default: True)
#         session_id_index: Index of session_id in args (default: 1, after query)
#         user_id_index: Index of user_id in args (default: 2)
#         user_cookie_index: Index of user_cookie in args (default: 3)
#         turn_number_index: Index of turn_number in args (default: 4)
#         new_chat_index: Index of new_chat in args (default: 5)
#         new_user_index: Index of new_user in args (default: 6)
    
#     Usage:
#         @trace_classmethod(kind=OpenInferenceSpanKindValues.CHAIN)
#         @classmethod
#         async def chat(cls, query: str, session_id: str | None, user_id: str | None, ...):
#             # Your code here
#             return result
#     """
#     def decorator(func: Callable) -> Callable:
#         @wraps(func)
#         async def wrapper(cls, *args, **kwargs):
#             tracer = trace.get_tracer(__name__)
#             span_name = name or f"{cls.__name__}.{func.__name__}"
            
#             # Extract input from first argument if requested
#             input_value = None
#             if extract_input and args:
#                 input_value = args[0] if isinstance(args[0], str) else None
            
#             # Extract context from args by index
#             session_id = safe_get_arg(args, session_id_index)
#             user_id = safe_get_arg(args, user_id_index)
#             user_cookie = safe_get_arg(args, user_cookie_index)
#             turn_number = safe_get_arg(args, turn_number_index)
#             new_chat = safe_get_arg(args, new_chat_index, False)
#             new_user = safe_get_arg(args, new_user_index, False)
            
#             # Convert turn_number to string if it's an int
#             if isinstance(turn_number, int):
#                 turn_number = str(turn_number)
            
#             # Build metadata
#             metadata = {
#                 "class": cls.__name__,
#                 "method": func.__name__,
#             }
            
#             async with async_spanner(
#                 tracer=tracer,
#                 name=span_name,
#                 kind=kind,
#                 session_id=session_id,
#                 user_id=user_id,
#                 user_cookie=user_cookie,
#                 turn_number=turn_number,
#                 input=input_value,
#                 new_chat=new_chat,
#                 new_user=new_user,
#                 metadata=metadata
#             ) as span:
#                 result = await func(cls, *args, **kwargs)
#                 _attach_output_to_span(span, result)
#                 return result
#         return wrapper
#     return decorator


# def trace_method_with_session_manager(name: str | None = None, kind=None, extract_input: bool = True):
#     """
#     Decorator for automatically tracing async instance methods with OpenTelemetry.
    
#     Args:
#         name: Custom span name (defaults to ClassName.method_name)
#         kind: OpenInference span kind (AGENT, CHAIN, etc.)
#         extract_input: Whether to extract first argument as input (default: True)
    
#     Usage:
#         @trace_method(kind=OpenInferenceSpanKindValues.AGENT)
#         async def run(self, query: str) -> List[dict]:
#             # Your code here
#             return result
#     """
#     def decorator(func: Callable) -> Callable:
#         @wraps(func)
#         async def wrapper(self, *args, **kwargs):
#             tracer = trace.get_tracer(__name__)
#             span_name = name or f"{self.__class__.__name__}.{func.__name__}"
            
#             # Extract input from first argument if requested
#             input_value = None
#             if extract_input and args:
#                 input_value = args[0] if isinstance(args[0], str) else None
            
#             # Extract context from session_manager if available
#             session_id = None
#             user_id = None
#             user_cookie = None
#             turn_number = None
#             new_chat = False
#             new_user = False
            
#             if hasattr(self, 'session_manager') and self.session_manager:
#                 sm = self.session_manager
#                 session_id = str(sm.session.id) if sm.session else None
#                 user_id = str(sm.user.id) if sm.user else None
#                 user_cookie = sm.user.cookie_id if sm.user else None
#                 turn_number = str(sm.state.turn_number)
#                 new_chat = sm.state.new_chat
#                 new_user = sm.state.new_user
            
#             # Build metadata
#             metadata = {
#                 "class": self.__class__.__name__,
#                 "method": func.__name__,
#             }
            
#             # Add agent metadata if available
#             if hasattr(self, 'agent') and self.agent:
#                 metadata["agent_name"] = self.agent.__class__.__name__
#                 metadata["agent_description"] = self.agent.description
#                 metadata["tools"] = [tool.__class__.__name__ for tool in self.agent.tools]
            
#             async with async_spanner(
#                 tracer=tracer,
#                 name=span_name,
#                 kind=kind,
#                 session_id=session_id,
#                 user_id=user_id,
#                 user_cookie=user_cookie,
#                 turn_number=turn_number,
#                 input=input_value,
#                 new_chat=new_chat,
#                 new_user=new_user,
#                 metadata=metadata
#             ) as span:
#                 result = await func(self, *args, **kwargs)
#                 _attach_output_to_span(span, result)
#                 return result
#         return wrapper
#     return decorator


# __all__ = ["async_spanner", "AgentTracer", "trace_method_with_session_manager", "trace_classmethod"]
