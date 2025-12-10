from fastapi import APIRouter

from ai_server.api.routes.chat_routes import router as chat_router
from ai_server.api.routes.session_routes import router as session_router
from ai_server.api.routes.user_routes import router as user_router

# Main router that combines all route modules
router = APIRouter()

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

__all__ = ["router"]
