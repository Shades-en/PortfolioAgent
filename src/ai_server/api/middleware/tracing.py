"""
Generic tracing middleware for capturing request attributes.

This middleware is endpoint-agnostic and captures all request data
(body, query params, path params) as span attributes for observability.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import SpanKind

import json

from ai_server.constants import SERVER


class GenericTracingMiddleware(BaseHTTPMiddleware):
    """
    Generic middleware to add request attributes to tracing spans.
    
    Captures all request data (body, query params, path params, headers)
    and adds them as span attributes for comprehensive observability.
    
    Does NOT set business context - that's done by individual endpoints
    using the trace_context context manager.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Capture request data and add to span attributes.
        
        Args:
            request: FastAPI Request object
            call_next: Next middleware/handler in chain
        
        Returns:
            Response from downstream handlers
        """
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            kind=SpanKind.SERVER  # OpenTelemetry standard kind
        ) as span:
            try:
                # Custom span category (for filtering/grouping)
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, SERVER)
                
                # HTTP attributes
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.route", request.url.path)
                span.set_attribute("http.scheme", request.url.scheme)
                span.set_attribute("http.host", request.url.hostname or "")
                
                # Headers
                if "user-agent" in request.headers:
                    span.set_attribute("http.user_agent", request.headers["user-agent"])
                
                # Collect input data for OpenInference INPUT attribute
                input_data = {}
                
                # Query parameters
                if request.query_params:
                    query_params = dict(request.query_params)
                    input_data["query_params"] = query_params
                    # Also set individual attributes for easy filtering
                    for key, value in query_params.items():
                        span.set_attribute(f"http.query.{key}", value)
                
                # Path parameters
                if hasattr(request, "path_params") and request.path_params:
                    path_params = dict(request.path_params)
                    input_data["path_params"] = path_params
                    # Also set individual attributes for easy filtering
                    for key, value in path_params.items():
                        span.set_attribute(f"http.path.{key}", str(value))
                
                # Request body (for POST/PUT/PATCH)
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        body = await request.json()
                        input_data["body"] = body
                        # Also set individual body field attributes
                        for key, value in body.items():
                            if isinstance(value, (str, int, float, bool)):
                                span.set_attribute(f"request.body.{key}", value)
                            else:
                                # For complex types, serialize to JSON
                                span.set_attribute(f"request.body.{key}", json.dumps(value))
                    except Exception:
                        # If body is not JSON or can't be read, skip
                        pass
                
                # Set OpenInference INPUT attribute with all input data
                if input_data:
                    span.set_attribute(
                        SpanAttributes.INPUT_VALUE,
                        json.dumps(input_data)
                    )
                
                # Process request
                response = await call_next(request)
                
                # Add response status
                span.set_attribute("http.status_code", response.status_code)
                
                # Set span status based on HTTP status code
                if response.status_code >= 400:
                    # Error response - set ERROR status with status code
                    span.set_status(
                        Status(StatusCode.ERROR, f"HTTP {response.status_code}")
                    )
                else:
                    # Success response
                    span.set_status(Status(StatusCode.OK))
                
                return response
                
            except Exception as e:
                span.set_attribute("http.status_code", 500)
                span.set_attribute("error.type", type(e).__name__)
                error_msg = str(e) if str(e) else type(e).__name__
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.record_exception(e)
                raise
