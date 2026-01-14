from __future__ import annotations

from opentelemetry.trace import SpanKind

from ai_server import config
from ai_server.ai.agents.agent import Agent
from ai_server.ai.providers import get_llm_provider
from ai_server.ai.providers.llm_provider import StreamCallback, dispatch_stream_event
from ai_server.api.exceptions.db_exceptions import (
    SessionNotFoundException,
    UserNotFoundException,
    MessageRetrievalFailedException,
)
from ai_server.types.message import MessageDTO, Role
from ai_server.schemas import Summary
from ai_server.session_manager import SessionManager
from ai_server.api.exceptions.agent_exceptions import MaxStepsReachedException
from ai_server.utils.tracing import trace_method
from ai_server.utils.general import generate_id
from ai_server.constants import (
    STREAM_EVENT_START,
    STREAM_EVENT_ERROR,
    STREAM_EVENT_FINISH,
)

import asyncio
from typing import List
from dataclasses import dataclass

from openinference.semconv.trace import OpenInferenceSpanKindValues
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of handling a query with messages, summary, and fallback status."""
    messages: List[MessageDTO]
    summary: Summary | None
    chat_name: str | None
    fallback: bool
    regenerated_summary: bool

class Runner:
    def __init__(self, agent: Agent, session_manager: SessionManager) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.skip_cache = config.SKIP_CACHE
        self.llm_provider = get_llm_provider(
            provider_name=config.LLM_PROVIDER, 
        )

    @trace_method(
        kind=SpanKind.INTERNAL,
        graph_node_id="llm_parallel_generation",
        capture_input=False,
        capture_output=False
    )
    async def _generate_response_and_metadata(
        self,
        conversation_history: List[MessageDTO],
        previous_conversation: List[MessageDTO],
        summary: Summary | None,
        query: str,
        tool_call: bool,
        on_stream_event: StreamCallback | None = None,
    ) -> tuple[tuple[List[MessageDTO], bool], tuple[Summary | None, str | None]]:
        """
        Generate LLM response and metadata (summary/chat_name) in parallel.
        
        Runs two LLM operations concurrently:
        1. Generate response (with potential tool calls)
        2. Generate summary or chat name (based on context)
        
        Args:
            conversation_history: Full conversation including system message and user query
            previous_conversation: Previous turns (for summarization)
            summary: Current summary (if any)
            query: User's query
            tool_call: Whether this is a tool call iteration
            
        Returns:
            Tuple of ((messages, tool_call), (new_summary, chat_name))
        
        Traced as INTERNAL span for parallel LLM operations.
        """
        # Use mock methods if MOCK_AI_RESPONSE is enabled
        if config.MOCK_AI_RESPONSE:
            return await asyncio.gather(
                self.llm_provider.mock_generate_response(),
                self.llm_provider.mock_generate_summary_or_chat_name(
                    query=query,
                    new_chat=self.session_manager.state.new_chat,
                    turns_after_last_summary=self.session_manager.state.turns_after_last_summary,
                    turn_number=self.session_manager.state.turn_number
                ),
            )
        
        # Use real LLM methods
        stream_enabled = on_stream_event is not None
        return await asyncio.gather(
            self.llm_provider.generate_response(
                conversation_history=conversation_history,
                tools=self.agent.tools,
                stream=stream_enabled,
                on_stream_event=on_stream_event if stream_enabled else None,
            ),
            self.llm_provider.generate_summary_or_chat_name(
                conversation_to_summarize=previous_conversation,
                previous_summary=summary,
                query=query,
                turns_after_last_summary=self.session_manager.state.turns_after_last_summary,
                context_token_count=self.session_manager.state.total_token_after_last_summary,
                tool_call=tool_call,
                new_chat=self.session_manager.state.new_chat,
                turn_number=self.session_manager.state.turn_number
            ),
        )

    @staticmethod
    async def _stream_fallback_response(
        on_stream_event: StreamCallback,
        response_message: MessageDTO,
    ) -> None:
        """
        Emit streaming events for a fallback response so the client receives error text in real time.
        """
        message_id = response_message.response_id or f"fallback_{generate_id(12)}"
        content = response_message.content or "Something went wrong while processing your request."

        await dispatch_stream_event(on_stream_event, {"type": STREAM_EVENT_START, "messageId": message_id})
        await dispatch_stream_event(
            on_stream_event,
            {"type": STREAM_EVENT_ERROR, "errorText": content},
        )
        await dispatch_stream_event(on_stream_event, {"type": STREAM_EVENT_FINISH})

    @trace_method(
        kind=OpenInferenceSpanKindValues.CHAIN,
        capture_output=False,
        graph_node_id="query_handler"
    )
    async def _handle_query(
        self,
        query: str,
        skip_cache: bool = config.SKIP_CACHE,
        on_stream_event: StreamCallback | None = None,
    ) -> QueryResult:
        turn_completed = False
        tool_call = False

        conversation_history: List[MessageDTO] = []
        previous_conversation: List[MessageDTO] = []
        summary: Summary | None = None
        new_summary: Summary | None = None
        messages: List[MessageDTO] = []
        chat_name: str | None = None

        user_query_message = MessageDTO(
            role=Role.HUMAN,
            tool_call_id="null",
            metadata={},
            content=query,
            function_call=None,
            order=1,
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

                # Generate LLM response and metadata in parallel
                (ai_messages, tool_call), (new_summary, chat_name) = await self._generate_response_and_metadata(
                    conversation_history=conversation_history,
                    previous_conversation=previous_conversation,
                    summary=summary,
                    query=query,
                    tool_call=tool_call,
                    on_stream_event=on_stream_event,
                )
                messages.extend(ai_messages)

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
                
                turn_previous_summary = summary
                regenerated_summary = False
                if new_summary:
                    turn_previous_summary = new_summary
                    regenerated_summary = True

            return QueryResult(
                messages=[user_query_message, *messages],
                summary=turn_previous_summary,
                chat_name=chat_name,
                fallback=False,
                regenerated_summary=regenerated_summary
            )
        except (SessionNotFoundException, UserNotFoundException, MessageRetrievalFailedException) as e:
            raise e
        except Exception as e:
            logger.error(f"Error in _handle_query: {e}")
            fallback_messages = self.session_manager.create_fallback_messages(user_query_message)
            if on_stream_event and fallback_messages:
                await self._stream_fallback_response(on_stream_event, fallback_messages[-1])
            
            # Try to get summary: use existing, fetch from DB, or None
            previous_summary = summary
            if not previous_summary and self.session_manager.session:
                previous_summary = await Summary.get_latest_by_session(
                    session_id=str(self.session_manager.session.id)
                )
            
            return QueryResult(
                messages=fallback_messages,
                summary=previous_summary,
                chat_name=chat_name,
                fallback=True,
                regenerated_summary=False
            )

    # Incase in future if handoff is required then make sure this function is retriggered
    @trace_method(
        kind=OpenInferenceSpanKindValues.AGENT,
        graph_node_id=lambda self: self.agent.name.lower()
    )
    async def run(
        self,
        query: str,
        skip_cache = config.SKIP_CACHE,
        on_stream_event: StreamCallback | None = None,
    ) -> dict:
        result: QueryResult = await self._handle_query(query, skip_cache, on_stream_event)
        messages = await self.session_manager.update_user_session(
            messages=result.messages,
            summary=result.summary,
            chat_name=result.chat_name,
            regenerated_summary=result.regenerated_summary
        )
        
        return {
            "messages": [
                msg.model_dump(mode='json', exclude={"session", "previous_summary"})
                for msg in messages
            ],
            "summary": (
                result.summary.model_dump(mode='json', exclude={"session"})
                if result.summary
                else None
            ),
            "chat_name": result.chat_name,
            "session_id": str(self.session_manager.session.id)
        }

# Test summary and chatname for chatcompeltion and responses with tool calls
# implement search chat feature
# implement starred messages endpoint and modify db schema