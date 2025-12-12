from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from ai_server.api.routes.chat_routes import router as chat_router
from ai_server.api.routes.session_routes import router as session_router
from ai_server.api.routes.user_routes import router as user_router
from ai_server.api.routes.message_routes import router as message_router

# Main router that combines all route modules
router = APIRouter()

# Root endpoint - redirect to health check
@router.get("/", tags=["Health"])
def root():
    """Redirect root path to health check endpoint."""
    return RedirectResponse(url="/health")

# Health check endpoint
@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

# Include chat routes
router.include_router(chat_router)

# Include session routes
router.include_router(session_router)

# Include user routes
router.include_router(user_router)

# Include message routes
router.include_router(message_router)

__all__ = ["router"]
