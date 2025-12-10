from typing import List

from ai_server.ai.runner import Runner
from ai_server.ai.agents.agent import AboutMeAgent
from ai_server.session_manager import SessionManager
from ai_server.ai.tools.tools import GetWeather, GetHoroscope

class ChatService:
    @classmethod
    async def chat(
        cls, 
        query: str, 
        session_id: str | None, 
        user_id: str | None, 
        user_cookie: str | None, 
        turn_number: int,
        new_chat: bool,
        new_user: bool,
    ) -> List[dict]:
        session_manager = SessionManager(
            user_id=user_id, 
            session_id=session_id,
            user_cookie=user_cookie,
            turn_number=turn_number, 
            new_chat=new_chat, 
            new_user=new_user
        )

        agent = AboutMeAgent(
            description="An agent that can answer questions about itself",
            instructions="You are to answer any question posed to you",
            tools=[GetWeather(), GetHoroscope()],
        )

        runner = Runner(agent=agent, session_manager=session_manager)
        return await runner.run(query)
