from ai_server import config
from ai_server.ai.agents.agent import Agent
from ai_server.ai.providers import get_llm_provider
from ai_server.types.message import MessageDTO, Role
from ai_server.schemas import Summary
from ai_server.session_manager import SessionManager
from ai_server.api.exceptions.agent_exceptions import MaxStepsReachedException

import asyncio
from typing import List
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Result of handling a query with messages, summary, and fallback status."""
    messages: List[MessageDTO]
    summary: Summary | None
    fallback: bool

class Runner:
    def __init__(self, agent: Agent, session_manager: SessionManager) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.skip_cache = config.SKIP_CACHE
        self.llm_provider = get_llm_provider(
            provider_name=config.LLM_PROVIDER, 
            chat_completion=config.CHAT_COMPLETION
        )

    async def _handle_query(self, query: str) -> QueryResult:
        turn_completed = False
        tool_call = False

        conversation_history: List[MessageDTO] = []
        previous_conversation: List[MessageDTO] = []
        summary: Summary | None = None
        new_summary: Summary | None = None
        messages: List[MessageDTO] = []

        user_query_message = MessageDTO(
            role=Role.HUMAN,
            tool_call_id="null",
            metadata={},
            content=query,
            function_call=None,
        )

        try:
            while not turn_completed:
                if not tool_call:
                    previous_conversation, summary = await self.session_manager.get_context_and_update_state()
                    system_message = self.llm_provider.build_system_message(
                        instructions=self.agent.instructions,
                        summary=summary.content if summary else None,
                    )
                    conversation_history = [system_message] + previous_conversation
                    conversation_history.append(user_query_message)

                (messages, tool_call), new_summary = await asyncio.gather(
                    self.llm_provider.generate_response(
                        conversation_history=conversation_history,
                        tools=self.agent.tools,
                    ),
                    # TODO: Generate summary ai call we can give it a structured output so it generates summary and the chat title in future
                    self.llm_provider.generate_summary(
                        conversation_to_summarize=previous_conversation,
                        previous_summary=summary,
                        query=query,
                        turns_after_last_summary=self.session_manager.state.turns_after_last_summary,
                        context_token_count=self.session_manager.state.total_token_after_last_summary,
                        tool_call=tool_call,
                        new_chat=self.session_manager.state.new_chat
                    ),
                )
                # If tool call is not made then turn is completed. If tool call is made 
                # then turn will be completed once AI executes the tool call, in the next iteration.
                if tool_call:
                    conversation_history.extend(messages)
                else:
                    turn_completed = True
                
                if not turn_completed:
                    self.session_manager.update_state(step=self.session_manager.state.step+1)
                    if self.session_manager.state.step > config.MAX_STEPS:
                        raise MaxStepsReachedException(
                            message="Agent exceeded maximum number of steps allowed",
                            note="The agent made too many tool calls in a single turn. Consider simplifying the task or increasing MAX_STEPS.",
                            current_step=self.session_manager.state.step,
                            max_steps=config.MAX_STEPS
                        )
                # If new summary is generated it means that the current turn's previous conversation is now the new summary
                # This is because new summary encapsulates all information from the previous conversation except the current turn
                turn_previous_summary = new_summary or summary

            return QueryResult(
                messages=[user_query_message, *messages],
                summary=turn_previous_summary,
                fallback=False
            )
        except Exception as e:
            fallback_messages = self.session_manager.create_fallback_messages(user_query_message)
            previous_summary = summary or Summary.get_latest_by_session(session_id=str(self.session.id))
            return QueryResult(
                messages=fallback_messages,
                summary=previous_summary,
                fallback=True
            )

    async def run(self, query: str, skip_cache = config.SKIP_CACHE) -> List[dict]:
        result: QueryResult = await self._handle_query(query, skip_cache)
        await self.session_manager.update_user_session(
            messages=result.messages,
            summary=result.summary
        )
        return [msg.model_dump() for msg in result.messages]


# add tracing
# add delete methods
# implement redis caching
# Maybe need better exception handling?