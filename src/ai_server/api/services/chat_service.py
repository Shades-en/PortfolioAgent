from openinference.semconv.trace import OpenInferenceSpanKindValues

from ai_server.ai.runner import Runner
from ai_server.ai.agents.agent import AboutMeAgent
from ai_server.ai.tools.tools import GetWeather, GetHoroscope

from ai_server.session_manager import SessionManager
from ai_server.utils.tracing import trace_method

class ChatService:
    @classmethod
    @trace_method(
        kind=OpenInferenceSpanKindValues.CHAIN,
        graph_node_id="chat_service"
    )
    async def chat(
        cls, 
        query: str, 
        session_id: str | None, 
        user_id: str | None, 
        user_cookie: str | None, 
        turn_number: int,
        new_chat: bool,
        new_user: bool,
        on_stream_event=None,
    ) -> dict:
        """
        Handle a chat request by orchestrating session management, agent creation, and query execution.
        
        Tracing context is set by endpoint. This method is traced as a CHAIN span
        representing the service-level workflow orchestration.
        
        Graph node: chat_service (parent inferred from span hierarchy)
        """
        # Initialize session manager with user context
        session_manager = SessionManager(
            user_id=user_id, 
            session_id=session_id,
            user_cookie=user_cookie,
            turn_number=turn_number, 
            new_chat=new_chat,
            new_user=new_user
        )

        # Create agent with tools
        agent = AboutMeAgent(
            description="An agent that can answer questions about itself",
            instructions="You are to answer any question posed to you",
            tools=[GetWeather(), GetHoroscope()],
        )

        # Execute query through runner
        runner = Runner(agent=agent, session_manager=session_manager)
        return await runner.run(query, on_stream_event=on_stream_event)
