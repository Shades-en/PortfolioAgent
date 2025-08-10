from fastapi import APIRouter
from ai_server.api.dto.chat import ChatRequest
from ai_server.ai.runner import Runner
from ai_server.ai.agents.agent import AboutMeAgent
from ai_server.ai.tools.tools import GetWeather, GetHoroscope

router = APIRouter()

@router.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

@router.post("/chat", tags=["Chat"])
def chat(chat_request: ChatRequest):
    agent = AboutMeAgent(
        description="AboutMeAgent",
        instructions="AboutMeAgent",
        tools=[GetWeather(), GetHoroscope()],
    )
    return Runner.run(agent, chat_request.query)