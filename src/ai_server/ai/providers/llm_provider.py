from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Awaitable, Any
import inspect

from ai_server import config
from ai_server.schemas.summary import Summary
from ai_server.types.message import MessageDTO, Role
from ai_server.ai.tools.tools import Tool
from ai_server.config import BASE_MODEL
from ai_server.utils.general import generate_id, generate_order


StreamEvent = dict[str, Any]
StreamCallback = Callable[[StreamEvent], Awaitable[None] | None]


async def dispatch_stream_event(callback: StreamCallback | None, event: StreamEvent) -> None:
    if not callback:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


class LLMProvider(ABC):
    provider: str = ""
    temperature: float = 0.7

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
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> any:
        """Generic wrapper for LLM API calls. Implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    async def _handle_ai_messages_and_tool_calls(
        cls, 
        response: any, 
        tools: List[Tool],
        ai_message: MessageDTO,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> bool:
        """
        Handle AI response and tool calls. Implemented by subclasses.
        
        Args:
            response: LLM response object
            tools: Available tools for function calling
            ai_message: MessageDTO to update with AI response parts
            stream: Whether streaming is enabled
            on_stream_event: Callback for streaming events
            
        Returns:
            bool: True if tool calls were made, False otherwise
        
        Note: Tracing is applied to concrete implementations, not abstract methods.
        """
        pass

    @classmethod
    @abstractmethod
    async def generate_response(
        cls, 
        conversation_history: List[MessageDTO], 
        tools: List[Tool] = [],
        tool_choice: str = "auto",
        model_name: str = BASE_MODEL,
        ai_message: MessageDTO | None = None,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> bool:
        """
        Generate a response from the LLM.
        
        Args:
            conversation_history: List of previous messages
            tools: Available tools for function calling
            tool_choice: Tool selection strategy ("auto", "required", "none")
            model_name: LLM model to use
            ai_message: MessageDTO to update with AI response (must be provided)
            stream: Whether streaming is enabled
            on_stream_event: Callback for streaming events
            
        Returns:
            bool: True if tool calls were made, False otherwise
        
        Note: Tracing is applied to concrete implementations, not abstract methods.
        """
        pass

    @classmethod
    @abstractmethod
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
        Returns tuple of (summary, chat_name) where either can be None.
        
        Note: Tracing is applied to concrete implementations, not abstract methods.
        """
        pass

    @classmethod
    @abstractmethod
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
        
        return MessageDTO.create_system_message(text=content, message_id="system")

    @classmethod
    async def _call_function(
        cls, 
        function_name: str, 
        function_arguments: dict, 
        tools: List[Tool]
    ) -> str:
        """Execute a function call from the tools list."""
        for tool in tools:
            if tool.name == function_name:
                return await tool(tool.Arguments(**function_arguments))
        return ""
    
    @classmethod
    async def mock_generate_response(cls, step: int) -> tuple[List[MessageDTO], bool]:
        """
        Mock implementation of generate_response for testing purposes.
        Returns a dummy AI message with random content and tool_call as False.
        
        Returns:
            Tuple of ([MessageDTO], False) where MessageDTO contains dummy content
        """
        mock_message = MessageDTO(
            role=Role.AI,
            content="This is a mock AI response. The actual LLM call has been bypassed for testing purposes.",
            metadata={"mock": True},
            order=generate_order(step, 2),
            response_id=f"mock_response_{generate_id(16, 'nanoid')}"
        )
        
        return [mock_message], False
    
    @classmethod
    async def mock_generate_summary_or_chat_name(
        cls,
        query: str,
        previous_summary: str | None = None,
        conversation_to_summarize: List[MessageDTO] | None = None,
        new_chat: bool = False,
        turns_after_last_summary: int = 0,
        turn_number: int = 1,
    ) -> tuple[str | None, str | None]:
        """
        Mock implementation of generate_summary_or_chat_name for testing purposes.
        Returns dummy summary and chat name without making actual LLM calls.
        
        Args:
            previous_summary: Summary text available before this turn
            conversation_to_summarize: Recent conversation messages when no summary exists
            new_chat: Whether this is a new chat (used to determine chat name generation)
            query: Current user query (used in mock chat name)
            
        Returns:
            Tuple of (summary, chat_name) where:
            - summary: Mock summary string
            - chat_name: Mock chat name or None (generated for new chats)
        """
        # Generate mock summary
        if config.MOCK_AI_SUMMARY:
            mock_summary = "Mock summary: This is a test summary of the conversation without actual LLM processing."
            mock_summary = Summary(
                content=mock_summary,
                end_turn_number=turn_number-1,
                start_turn_number=turn_number-turns_after_last_summary # This should be previous summary end turn number + 1
            )
        else:
            mock_summary = None
        # Generate mock chat name for new chats
        mock_chat_name = None
        if new_chat or config.MOCK_AI_CHAT_NAME:
            # Use first few words of query for chat name
            context_hint = ""
            if previous_summary:
                context_hint = " (summary context)"
            elif conversation_to_summarize:
                context_hint = " (recent convo)"
            query_words = query.split()[:3]
            mock_chat_name = f"Mock Chat{context_hint}: {' '.join(query_words)}"
        
        return mock_summary, mock_chat_name