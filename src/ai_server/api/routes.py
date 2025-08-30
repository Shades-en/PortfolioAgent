from fastapi import APIRouter

from ai_server.api.dto.chat import ChatRequest
from ai_server.ai.runner import Runner
from ai_server.ai.agents.agent import AboutMeAgent
from ai_server.ai.tools.tools import GetWeather, GetHoroscope

from ai_server.redis.client import RedisClient

router = APIRouter()

# redis_client = RedisClient() # TODO: This should be in fast api startup events

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

# @router.delete("/conversation_history", tags=["DEV - Conversation History"])
# def delete_all_conversation_history():
#     redis_client.delete_conv_index_data()
#     return {"status": "ok"} # TODO: change to DTO

# @router.post("/conversation_history", tags=["DEV - Conversation History"])
# def create_conversation_history():
#     redis_client.create_conv_memory_index()
#     return {"status": "ok"} # TODO: change to DTO