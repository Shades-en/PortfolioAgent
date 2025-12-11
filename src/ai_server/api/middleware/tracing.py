"""
Generic tracing middleware for capturing request attributes.

This middleware is endpoint-agnostic and captures all request data
(body, query params, path params) as span attributes for observability.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

import json


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
            kind=trace.SpanKind.SERVER
        ) as span:
            try:
                # HTTP attributes
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.route", request.url.path)
                span.set_attribute("http.scheme", request.url.scheme)
                span.set_attribute("http.host", request.url.hostname or "")
                
                # Headers
                if "user-agent" in request.headers:
                    span.set_attribute("http.user_agent", request.headers["user-agent"])
                
                # Query parameters
                if request.query_params:
                    for key, value in request.query_params.items():
                        span.set_attribute(f"http.query.{key}", value)
                
                # Path parameters
                if hasattr(request, "path_params") and request.path_params:
                    for key, value in request.path_params.items():
                        span.set_attribute(f"http.path.{key}", str(value))
                
                # Request body (for POST/PUT/PATCH)
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        body = await request.json()
                        # Add each body field as attribute
                        for key, value in body.items():
                            if isinstance(value, (str, int, float, bool)):
                                span.set_attribute(f"request.body.{key}", value)
                            else:
                                # For complex types, serialize to JSON
                                span.set_attribute(f"request.body.{key}", json.dumps(value))
                    except Exception:
                        # If body is not JSON or can't be read, skip
                        pass
                
                # Process request
                response = await call_next(request)
                
                # Add response status
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(Status(StatusCode.OK))
                
                return response
                
            except Exception as e:
                span.set_attribute("http.status_code", 500)
                span.set_attribute("error.type", type(e).__name__)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
