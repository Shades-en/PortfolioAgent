"""Middleware for FastAPI application."""

from ai_server.api.middleware.tracing import GenericTracingMiddleware

__all__ = ["GenericTracingMiddleware"]
