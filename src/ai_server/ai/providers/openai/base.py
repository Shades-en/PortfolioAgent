from ai_server.ai.providers.llm_provider import (
    LLMProvider,
    StreamCallback,
    dispatch_stream_event,
)
from ai_server.ai.tools.tools import Tool
from ai_server.ai.prompts.summary import CONVERSATION_SUMMARY_PROMPT
from ai_server.ai.prompts.chat_name import CHAT_NAME_SYSTEM_PROMPT, CHAT_NAME_USER_PROMPT
from ai_server.ai.providers.utils import create_tool_output_available_event

from ai_server.utils.general import get_token_count, generate_order
from ai_server.utils.tracing import trace_method

from ai_server.schemas.summary import Summary
from ai_server.types.message import MessageDTO, Role
from ai_server.config import (
    BASE_MODEL, 
    MAX_TOKEN_THRESHOLD, 
    MAX_TURNS_TO_FETCH, 
    TURNS_BETWEEN_CHAT_NAME, 
    MAX_CHAT_NAME_WORDS, 
    MAX_CHAT_NAME_LENGTH,
    CHAT_NAME_CONTEXT_MAX_MESSAGES,
)
from ai_server.constants import (
    OPENAI,
    STREAM_EVENT_FINISH,
)

from openinference.semconv.trace import OpenInferenceSpanKindValues
import logging

import asyncio
import json

import openai
from openai.types.responses import Response
from openai.types.chat.chat_completion import ChatCompletion

import os
from typing import List, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider, ABC):
    provider: str = OPENAI
    temperature: float = 0.7
    async_client: openai.AsyncOpenAI = None

    @staticmethod
    def _safe_json_loads(value: str | None) -> dict | str | None:
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

    @classmethod
    def _get_client(cls) -> openai.AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if cls.async_client is None:
            cls.async_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return cls.async_client
    
    @classmethod
    def _extract_text_from_response(cls, response: Response | ChatCompletion) -> str:
        """
        Extract text content from OpenAI API response.
        Handles both Responses API and Chat Completion API response types.
        
        Args:
            response: Response from OpenAI API (_call_llm)
            
        Returns:
            Extracted text content as string
            
        Raises:
            ValueError: If response type is not recognized
        """
        if isinstance(response, Response):
            # OpenAI Responses API
            return response.output_text.strip()
        elif isinstance(response, ChatCompletion):
            # OpenAI Chat Completion API
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

    @classmethod
    @abstractmethod
    async def _call_llm(
        cls,
        input_messages: List[Dict],
        model: str = BASE_MODEL,
        temperature: float | None = None,
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
        instructions: str | None = None,
    ) -> any:
        """Generic wrapper for OpenAI API calls. Implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def _convert_to_openai_compatible_messages(cls, message: MessageDTO) -> Dict:
        """Convert MessageDTO to OpenAI-compatible format. Implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def _convert_tools_to_openai_compatible(cls, tools: List[Tool]) -> List[Dict]:
        """Convert tools to OpenAI-compatible format. Implemented by subclasses."""
        pass

    @classmethod
    async def _process_tool_call_responses(
        cls,
        function_call_tasks: List,
        function_call_messages: List[tuple[MessageDTO, str]],
        response_id: str,
        step: int,
        order: int,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> List[MessageDTO]:
        """
        Common method to process tool call responses with error handling.
        
        Args:
            function_call_tasks: List of async tasks for tool execution
            function_call_messages: List of (message_ai, call_id) tuples
            response_id: Response ID for the messages
            step: Current step number
            order: Starting order number for messages
            stream: Whether streaming is enabled
            on_stream_event: Callback for streaming events
            
        Returns:
            List of MessageDTO objects (AI messages + tool response messages)
        """
        
        messages: List[MessageDTO] = []
        
        if not function_call_tasks:
            return messages
        
        # Execute all function calls in parallel with exception handling
        function_responses = await asyncio.gather(*function_call_tasks, return_exceptions=True)
        
        # Create tool messages with the responses
        for i, (message_ai, call_id) in enumerate(function_call_messages):
            tool_output = function_responses[i]
            tool_success = not isinstance(tool_output, Exception)
            
            if tool_success:
                if hasattr(tool_output, "model_dump"):
                    tool_output_payload = tool_output.model_dump()
                else:
                    tool_output_payload = tool_output
                if stream:
                    await dispatch_stream_event(
                        on_stream_event,
                        create_tool_output_available_event(call_id, tool_output_payload),
                    )
                content = str(tool_output)
            else:
                content = f"Error: {str(tool_output)}"
            
            message_tool = MessageDTO(
                role=Role.TOOL,
                tool_call_id=call_id,
                metadata={
                    "tool_success": tool_success
                },
                content=content,
                function_call=None,
                order=generate_order(step, order),
                response_id=response_id
            )
            messages.append(message_tool)
        
        return messages

    @classmethod
    @abstractmethod
    async def _handle_ai_messages_and_tool_calls(
        cls, 
        response: any, 
        tools: List[Tool],
        step: int,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> List[MessageDTO]:
        """Handle AI response and tool calls. Implemented by subclasses."""
        pass

    @classmethod
    @trace_method(
        kind=OpenInferenceSpanKindValues.LLM,
        graph_node_id="llm_generate_response",
        capture_input=False,  # Don't capture full conversation history
        capture_output=False  # Don't capture full response
    )
    async def generate_response(
        cls, 
        conversation_history: List[MessageDTO], 
        tools: List[Tool] = [],
        tool_choice: str = "auto",
        model_name: str = BASE_MODEL,
        step: int = 1,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> tuple[List[MessageDTO], bool]:
        """
        Generate a response from the LLM.
        
        Traced as LLM span with model name and tool information.
        OpenAI instrumentation will automatically capture tokens, latency, and costs.
        """
        tool_call = False
        input_messages = list(map(cls._convert_to_openai_compatible_messages, conversation_history))
        response = await cls._call_llm(
            input_messages=input_messages,
            model=model_name,
            tools=cls._convert_tools_to_openai_compatible(tools),
            tool_choice=tool_choice,
            stream=stream,
            on_stream_event=on_stream_event,
        )
        ai_messages = await cls._handle_ai_messages_and_tool_calls(
            response,
            tools,
            step,
            stream,
            on_stream_event,
        )
        if stream:
            await dispatch_stream_event(on_stream_event, {"type": STREAM_EVENT_FINISH})
        if ai_messages[-1].role == Role.TOOL:
            tool_call = True
        return ai_messages, tool_call

    @classmethod
    async def _summarise(
        cls,
        conversation_to_summarize: List[MessageDTO], 
        previous_summary: str | None, 
    ) -> str:
        """Generate a summary of the conversation combined with previous summary."""
        # Build the input for summarization
        summary_input = []
        if previous_summary:
            summary_input.append({
                "role": "user",
                "content": f"Previous conversation summary:\n{previous_summary}"
            })
        
        # Add conversation messages
        conversation_text = "\n".join([
            f"{msg.role.value}: {msg.content}" 
            for msg in conversation_to_summarize 
            if msg.content
        ])
        summary_input.append({
            "role": "user", 
            "content": f"New conversation to incorporate:\n{conversation_text}\n\nPlease provide an updated summary."
        })
        
        response = await cls._call_llm(
            input_messages=summary_input,
            instructions=CONVERSATION_SUMMARY_PROMPT,
            temperature=0.3,  # Lower temperature for more consistent summaries
        )
        return cls._extract_text_from_response(response)

    @classmethod
    @trace_method(
        kind=OpenInferenceSpanKindValues.LLM,
        graph_node_id="llm_generate_summary_or_chat_name",
        capture_input=False,
        capture_output=False
    )
    async def generate_summary_or_chat_name(
        cls, 
        conversation_to_summarize: List[MessageDTO], 
        previous_summary: str | None, 
        query: str, 
        turns_after_last_summary: int,
        context_token_count: int,
        tool_call: bool = False,
        new_chat: bool = False,
        turn_number: int = 1,
    ) -> tuple[str | None, str | None]:
        """
        Generate a summary and/or chat name based on conversation state.
        
        Traced as LLM span for summary/chat name generation.
        
        Summary generation:
        - Triggered when token threshold or turn limit is exceeded
        - Only for non-tool-call, non-new-chat scenarios
        
        Chat name generation:
        - For new chats: Generate immediately using just the query
        - For existing chats: Generate every N turns (TURNS_BETWEEN_CHAT_NAME) using query and summary
        - Never generated during tool calls
        
        Args:
            conversation_to_summarize: Messages to summarize
            previous_summary: Existing summary to incorporate
            query: Current user query
            turns_after_last_summary: Number of turns since last summary
            context_token_count: Current context token count
            tool_call: Whether this is a tool call (skips both summary and chat name)
            new_chat: Whether this is a new chat (generates chat name immediately)
            turn_number: Current turn number (for periodic chat name generation)
            
        Returns:
            Tuple of (summary, chat_name) where either can be None
        """
        summary: str | None = None
        chat_name: str | None = None
        
        # Generate summary if threshold exceeded (not for tool calls or new chats)
        if not tool_call and not new_chat:
            query_tokens = get_token_count(query)
            should_summarize = (
                (query_tokens + context_token_count >= MAX_TOKEN_THRESHOLD) or 
                (turns_after_last_summary >= MAX_TURNS_TO_FETCH)
            )
            if should_summarize:
                summary = await cls._summarise(conversation_to_summarize, previous_summary)
                summary = Summary(
                    content=summary, 
                    end_turn_number=turn_number-1,
                    start_turn_number=turn_number-turns_after_last_summary # This should be previous summary end turn number + 1
                )
        
        # Generate chat name (not during tool calls)
        if not tool_call:
            if new_chat:
                # New chat: generate name from query only
                chat_name = await cls._generate_chat_name(query)
            elif turn_number % TURNS_BETWEEN_CHAT_NAME == 0:
                # Existing chat: generate name periodically with context
                chat_name = await cls._generate_chat_name(
                    query,
                    previous_summary,
                    conversation_to_summarize,
                )
        return summary, chat_name
    
    @classmethod
    def _build_chat_name_context(
        cls,
        conversation_to_summarize: List[MessageDTO],
        max_messages: int = CHAT_NAME_CONTEXT_MAX_MESSAGES,
    ) -> str:
        """Format recent conversation snippets for chat name context."""
        if not conversation_to_summarize:
            return ""

        relevant_messages = conversation_to_summarize[-max_messages:]
        parts: List[str] = []
        part_tokens: List[int] = []
        total_tokens = 0
        for message in relevant_messages:
            if not message.content:
                continue
            role_label = message.role.value.capitalize()
            snippet = f"{role_label}: {message.content.strip()}"
            tokens = get_token_count(snippet)
            parts.append(snippet)
            part_tokens.append(tokens)
            total_tokens += tokens
        while parts and total_tokens > MAX_TOKEN_THRESHOLD:
            total_tokens -= part_tokens.pop(0)
            parts.pop(0)

        context = "\n".join(parts).strip()
        return context

    @classmethod
    async def _generate_chat_name(
        cls,
        query: str,
        previous_summary: str | None = None,
        conversation_to_summarize: List[MessageDTO] | None = None,
    ) -> str:
        """
        Generate a meaningful, concise chat name based on the query and optional previous summary.
        
        Args:
            query: The user's query
            previous_summary: Optional previous conversation summary for context
            conversation_to_summarize: Optional recent conversation messages when summary unavailable
            
        Returns:
            A concise chat name (respects MAX_CHAT_NAME_LENGTH and MAX_CHAT_NAME_WORDS config)
        """
        context_sections: List[str] = []
        if previous_summary:
            context_sections.append(f"Previous summary:\n{previous_summary.content}")

        if conversation_to_summarize:
            conversation_context = cls._build_chat_name_context(conversation_to_summarize)
            if conversation_context:
                context_sections.append(
                    "Recent conversation snippets:\n" + conversation_context
                )

        context = "\n\n".join(context_sections)
        if context:
            context = f"{context}\n\n"

        user_prompt = CHAT_NAME_USER_PROMPT.format(
            context=context, 
            query=query,
            max_words=MAX_CHAT_NAME_WORDS
        )

        try:
            # Build messages for _call_llm
            input_messages = [
                {"role": "system", "content": CHAT_NAME_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await cls._call_llm(
                input_messages=input_messages,
                model=BASE_MODEL,
                temperature=0.7
            )
            
            # Extract text from response
            chat_name = cls._extract_text_from_response(response)
            
            # Ensure it's not too long
            max_length = MAX_CHAT_NAME_LENGTH
            if len(chat_name) > max_length:
                chat_name = chat_name[:max_length - 3] + "..."
            
            return chat_name
            
        except Exception:
            # Fallback: use first few words of query
            max_words = MAX_CHAT_NAME_WORDS
            words = query.split()[:max_words]
            fallback_name = " ".join(words)
            max_length = MAX_CHAT_NAME_LENGTH
            return fallback_name[:max_length] if len(fallback_name) <= max_length else fallback_name[:max_length - 3] + "..."

    @classmethod
    def build_system_message(
        cls,
        instructions: str,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> MessageDTO:
        """
        Build a system message with optional summary context.
        
        Args:
            instructions: Base system prompt (e.g., "You are a helpful assistant.")
            summary: Optional summary of earlier conversation to include
            metadata: Optional metadata dict
            
        Returns:
            MessageDTO object with Role.SYSTEM
        """
        if summary:
            content = f"{instructions}\n\n---\nSummary of earlier conversation:\n{summary}"
        else:
            content = instructions
        
        return MessageDTO(
            role=Role.SYSTEM,
            tool_call_id="null",
            metadata=metadata or {},
            content=content,
            function_call=None,
            order=0
        )
