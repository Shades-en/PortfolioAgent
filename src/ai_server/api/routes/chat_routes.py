from fastapi import APIRouter

from ai_server.api.dto.chat import ChatRequest
from ai_server.api.services import ChatService

router = APIRouter()

@router.post("/chat", tags=["Chat"])
async def chat(chat_request: ChatRequest):
    return await ChatService.chat(
        query=chat_request.query,
        session_id=chat_request.session_id,
        user_id=chat_request.user_id,
        user_cookie=chat_request.user_cookie,
        turn_number=chat_request.turn_number,
        new_chat=chat_request.new_chat,
        new_user=chat_request.new_user
    )
