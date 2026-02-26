from openinference.semconv.trace import OpenInferenceSpanKindValues
from typing import AsyncGenerator, Tuple
import asyncio

from omniagent.ai.agents.agent import Agent
from omniagent.ai.runner import Runner
from omniagent.config import BASE_MODEL
from omniagent.persistence import get_context
from omniagent.types import MessageQuery, RunnerOptions

from ai_server.tools import GetCompanyName, GetHoroscope
from ai_server.utils.tracing import trace_method


class ChatService:
    @classmethod
    def _build_runner(
        cls,
        *,
        session_id: str | None,
        user_cookie: str,
        stream: bool,
    ) -> Runner:
        session_manager_cls = get_context().session_manager_cls
        session_manager = session_manager_cls(
            session_id=session_id,
            user_client_id=user_cookie,
        )

        agent = Agent(
            name="AboutMeAgent",
            description="An agent that can answer questions about itself",
            instructions="You are to answer any question posed to you",
            model=BASE_MODEL,
            tools=[GetCompanyName(), GetHoroscope()],
        )

        runner_options = RunnerOptions(stream=stream)
        return Runner(agent=agent, session_manager=session_manager, options=runner_options)

    @classmethod
    @trace_method(
        kind=OpenInferenceSpanKindValues.CHAIN,
        graph_node_id="chat_service"
    )
    async def chat(
        cls, 
        query_message: MessageQuery, 
        session_id: str | None, 
        user_cookie: str, 
    ) -> dict:
        """
        Handle a chat request by orchestrating session management, agent creation, and query execution.
        
        Tracing context is set by endpoint. This method is traced as a CHAIN span
        representing the service-level workflow orchestration.
        
        Graph node: chat_service (parent inferred from span hierarchy)
        """
        runner = cls._build_runner(
            session_id=session_id,
            user_cookie=user_cookie,
            stream=False,
        )
        return await runner.run(query_message=query_message)

    @classmethod
    @trace_method(
        kind=OpenInferenceSpanKindValues.CHAIN,
        graph_node_id="chat_service"
    )
    async def chat_stream(
        cls, 
        query_message: MessageQuery, 
        session_id: str | None, 
        user_cookie: str, 
    ) -> Tuple[AsyncGenerator[str, None], asyncio.Future]:
        """
        Handle a streaming chat request, returning a generator and result future.
        
        This method provides a cleaner streaming API where:
        - The generator yields formatted SSE events (including DONE sentinel)
        - The result dict is available via the future after stream completes
        
        Returns:
            Tuple of (event_generator, result_future)
        """
        runner = cls._build_runner(
            session_id=session_id,
            user_cookie=user_cookie,
            stream=True,
        )
        return await runner.run_stream(query_message=query_message)
