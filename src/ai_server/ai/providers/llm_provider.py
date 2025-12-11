from abc import ABC, abstractmethod
from typing import List, Dict

from ai_server.types.message import MessageDTO, FunctionCallRequest
from ai_server.ai.tools.tools import Tool
from ai_server.config import BASE_MODEL


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
    ) -> any:
        """Generic wrapper for LLM API calls. Implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    async def _handle_ai_messages_and_tool_calls(
        cls, 
        response: any, 
        tools: List[Tool],
    ) -> List[MessageDTO]:
        """Handle AI response and tool calls. Implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    async def generate_response(
        cls, 
        conversation_history: List[MessageDTO], 
        tools: List[Tool] = [], 
        tool_choice: str = "auto",
        model_name: str = BASE_MODEL
    ) -> tuple[List[MessageDTO], bool]:
        """Generate a response from the LLM."""
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
        """Build a system message with optional summary context."""
        pass

    @classmethod
    async def _call_function(cls, function_call_request: FunctionCallRequest, tools: List[Tool]) -> str:
        """Execute a function call from the tools list."""
        for tool in tools:
            if tool.name == function_call_request.name:
                return await tool(tool.Arguments(**function_call_request.arguments))
        return ""