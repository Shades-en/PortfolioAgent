from ai_server.api.exceptions.openai_exceptions import UnrecognizedMessageTypeException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from ai_server.ai.providers.llm_provider import LLMProvider
from ai_server.ai.providers.embedding_provider import EmbeddingProvider
from ai_server.ai.tools.tools import Tool
from ai_server.ai.prompts.summary import CONVERSATION_SUMMARY_PROMPT

from ai_server.schemas.message import Message, Role, FunctionCallRequest
from ai_server.utils.general import get_env_int, get_token_count
from ai_server.config import BASE_MODEL, BASE_EMBEDDING_MODEL
from ai_server.constants import OPENAI

import asyncio
import json
from pydantic import ValidationError
from langchain_openai import OpenAIEmbeddings

import openai
from openai.types.responses import Response
from openai.types.chat.chat_completion import ChatCompletion

import os
from typing import List, Dict
from abc import ABC, abstractmethod
import logging

from config import MAX_TOKEN_THRESHOLD, MAX_TURNS_TO_FETCH

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider, ABC):
    def __init__(self, temperature: float = 0.7) -> None:
        super().__init__(OPENAI)
        self.temperature = temperature
        self.async_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @abstractmethod
    async def _call_llm(
        self,
        input_messages: List[Dict],
        model: str = BASE_MODEL,
        temperature: float | None = None,
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
        instructions: str | None = None,
    ) -> any:
        """Generic wrapper for OpenAI API calls. Implemented by subclasses."""
        pass

    @abstractmethod
    def _convert_to_openai_compatible_messages(self, message: Message) -> Dict:
        """Convert Message to OpenAI-compatible format. Implemented by subclasses."""
        pass

    @abstractmethod
    def _convert_tools_to_openai_compatible(self, tools: List[Tool]) -> List[Dict]:
        """Convert tools to OpenAI-compatible format. Implemented by subclasses."""
        pass

    @abstractmethod
    async def _handle_ai_messages_and_tool_calls(
        self, 
        response: any, 
        user_id: str, 
        session_id: str,
        turn_id: str,
        tools: List[Tool],
    ) -> List[Message]:
        """Handle AI response and tool calls. Implemented by subclasses."""
        pass

    async def generate_response(
        self, 
        conversation_history: List[Message], 
        user_id: str, 
        session_id: str, 
        turn_id: str,
        tools: List[Tool] = [], 
        tool_choice: str = "auto",
        model_name: str = BASE_MODEL
    ) -> List[Message]:
        """Generate a response from the LLM."""
        input_messages = list(map(self._convert_to_openai_compatible_messages, conversation_history))
        response = await self._call_llm(
            input_messages=input_messages,
            model=model_name,
            tools=self._convert_tools_to_openai_compatible(tools),
            tool_choice=tool_choice,
        )
        ai_messages = await self._handle_ai_messages_and_tool_calls(response, user_id, session_id, turn_id, tools)
        if ai_messages[-1].role == Role.TOOL:
            tool_call = True
        return ai_messages, tool_call

    async def _summarise(
        self,
        conversation_to_summarize: List[Message], 
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
        
        response = await self._call_llm(
            input_messages=summary_input,
            instructions=CONVERSATION_SUMMARY_PROMPT,
            temperature=0.3,  # Lower temperature for more consistent summaries
        )
        return response.output_text

    async def generate_summary(
        self, 
        conversation_to_summarize: List[Message], 
        previous_summary: str | None, 
        query: str, 
        turns_after_last_summary: int,
        context_token_count: int,
        tool_call: bool
    ) -> str | None:
        """
        Generate a new summary if token threshold or turn limit is exceeded.
        
        Args:
            conversation_to_summarize: Messages to summarize
            previous_summary: Existing summary to incorporate
            query: Current user query
            turns_after_last_summary: Number of turns since last summary
            context_token_count: Current context token count
            pure_user_query: Whether the query is from the user
            
        Returns:
            New summary string if summarization triggered, None otherwise
        """
        if not tool_call:
            query_tokens = get_token_count(query)
            should_summarize = (
                (query_tokens + context_token_count >= MAX_TOKEN_THRESHOLD) or 
                (turns_after_last_summary >= MAX_TURNS_TO_FETCH)
            )
            if should_summarize:
                return await self._summarise(conversation_to_summarize, previous_summary)
        return None

    @classmethod
    def build_system_message(
        cls,
        instructions: str,
        user_id: str,
        session_id: str,
        turn_id: str,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> Message:
        """
        Build a system message with optional summary context.
        
        Args:
            instructions: Base system prompt (e.g., "You are a helpful assistant.")
            user_id: User identifier
            session_id: Session identifier
            turn_id: Turn identifier
            summary: Optional summary of earlier conversation to include
            metadata: Optional metadata dict
            
        Returns:
            Message object with Role.SYSTEM
        """
        if summary:
            content = f"{instructions}\n\n---\nSummary of earlier conversation:\n{summary}"
        else:
            content = instructions
        
        return Message(
            role=Role.SYSTEM,
            tool_call_id="null",
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            metadata=metadata or {},
            content=content,
            function_call=None,
        )

class OpenAIResponsesAPI(OpenAIProvider):
    def __init__(self, temperature: float = 0.7) -> None:
        super().__init__(temperature)

    async def _call_llm(
        self,
        input_messages: List[Dict],
        model: str = BASE_MODEL,
        temperature: float | None = None,
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
        instructions: str | None = None,
    ) -> Response:
        """Generic wrapper for OpenAI Responses API calls."""
        kwargs = {
            "model": model,
            "input": input_messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if instructions:
            kwargs["instructions"] = instructions
        return await self.async_client.responses.create(**kwargs)

    def _convert_to_openai_compatible_messages(self, message: Message) -> Dict:
        role = "user"
        match message.role:
            case Role.HUMAN:
                role = "user"
            case Role.AI:
                role = "assistant"
                if message.tool_call_id != "null":
                    return {
                        "call_id": message.tool_call_id,
                        "type": "function_call",
                        "name": message.function_call.name,
                        "arguments": json.dumps(message.function_call.arguments),
                    }
            case Role.SYSTEM:
                role = "developer"
            case Role.TOOL:
                return {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
        return {
            "role": role,
            "content": message.content,
        }

    async def _handle_ai_messages_and_tool_calls(
        self, 
        response: Response, 
        user_id: str, 
        session_id: str,
        turn_id: str,
        tools: List[Tool],
    ) -> List[Message]:
        outputs = response.output
        messages: List[Message] = []
        
        # Collect function calls and regular messages separately
        function_call_tasks = []
        function_call_messages = []
        
        try:
            for resp in outputs:
                if resp.type == "function_call":
                    function_call = FunctionCallRequest(
                        name=resp.name,
                        arguments=json.loads(resp.arguments),
                    )
                    message_ai = Message(
                        role=Role.AI,
                        tool_call_id=resp.call_id,
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata={},
                        content='',
                        function_call=function_call,
                    )
                    
                    # Create task for parallel execution
                    task = self._call_function(function_call, tools)
                    function_call_tasks.append(task)
                    function_call_messages.append((message_ai, resp.call_id))
                    
                elif resp.type == "message":
                    message = Message(
                        role=Role.AI,
                        tool_call_id="null",
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata={},
                        content=resp.content[0].text,
                        function_call=None,
                    )
                    messages.append(message)
                else:
                    raise UnrecognizedMessageTypeException(message="Unrecognized message type", note=f"Message type: {resp.type} - Implementation does not exist")
            
            # Execute all function calls in parallel
            if function_call_tasks:
                function_responses = await asyncio.gather(*function_call_tasks)
                
                # Create tool messages with the responses
                for i, (message_ai, call_id) in enumerate(function_call_messages):
                    message_tool = Message(
                        role=Role.TOOL,
                        tool_call_id=call_id,
                        user_id=user_id,
                        turn_id=turn_id,
                        session_id=session_id,
                        metadata={},
                        content=function_responses[i],
                        function_call=None,
                    )
                    messages.append(message_ai)
                    messages.append(message_tool)
            return messages
        except ValidationError as e:
            raise MessageParseException(message="Failed to parse AI response from openai responses", note=str(e))

    def _convert_tools_to_openai_compatible(self, tools: List[Tool]) -> List[Dict]:
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            }
            for arg in tool.arguments:
                openai_tool["parameters"]["properties"][arg.name] = {
                    "type": arg.type,
                    "description": arg.description,
                }
                if arg.required:
                    openai_tool["parameters"]["required"].append(arg.name)
            openai_tools.append(openai_tool)
        return openai_tools
        
class OpenAIChatCompletionAPI(OpenAIProvider):
    def __init__(self, temperature: float = 0.7) -> None:
        super().__init__(temperature)

    def _convert_to_openai_compatible_messages(self, message: Message) -> Dict:
        role = "user"
        match message.role:
            case Role.HUMAN:
                role = "user"
            case Role.AI:
                role = "assistant"
                if message.tool_call_id != "null":
                    return {
                        "role": role,
                        "tool_calls": [
                            {
                                "id": message.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": message.function_call.name,
                                    "arguments": json.dumps(message.function_call.arguments),
                                },
                            }
                        ],
                    }
            case Role.SYSTEM:
                role = "developer"
            case Role.TOOL:
                return {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
        return {
            "role": role,
            "content": message.content,
        }

    async def _handle_ai_messages_and_tool_calls(
        self, 
        response: ChatCompletion, 
        user_id: str, 
        session_id: str,
        turn_id: str,
        tools: List[Tool]
    ) -> List[Message]:
        output = response.choices[0].message
        messages: List[Message] = []
        content = output.content
        tool_calls = output.tool_calls
        try:
            if content:
                message = Message(
                    role=Role.AI,
                    tool_call_id="null",
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    metadata={},
                    content=content,
                    function_call=None,
                )
                return [message]
            if tool_calls:
                # Collect function calls for parallel execution
                function_call_tasks = []
                function_call_messages = []
                
                for tool_call in tool_calls:
                    function_call = FunctionCallRequest(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    )
                    message_ai = Message(
                        role=Role.AI,
                        tool_call_id=tool_call.id,
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata={},
                        content='',
                        function_call=function_call,
                    )
                    
                    # Create task for parallel execution
                    task = self._call_function(function_call, tools)
                    function_call_tasks.append(task)
                    function_call_messages.append((message_ai, tool_call.id))
                
                # Execute all function calls in parallel
                function_responses = await asyncio.gather(*function_call_tasks)
                
                # Create tool messages with the responses
                for i, (message_ai, call_id) in enumerate(function_call_messages):
                    message_tool = Message(
                        role=Role.TOOL,
                        tool_call_id=call_id,
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata={},
                        content=function_responses[i],
                        function_call=None,
                    )
                    messages.append(message_ai)
                    messages.append(message_tool)
            return messages
        except ValidationError as e:
            raise MessageParseException(message="Failed to parse AI response from openai responses", note=str(e))

    async def _call_llm(
        self,
        input_messages: List[Dict],
        model: str = BASE_MODEL,
        temperature: float | None = None,
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
        instructions: str | None = None,
    ) -> ChatCompletion:
        """Generic wrapper for OpenAI Chat Completions API calls."""
        kwargs = {
            "model": model,
            "messages": input_messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        # Note: instructions not supported in Chat Completions API, use system message instead
        return await self.async_client.chat.completions.create(**kwargs)

    def _convert_tools_to_openai_compatible(self, tools: List[Tool]) -> List[Dict]:
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
            for arg in tool.arguments:
                openai_tool["function"]["parameters"]["properties"][arg.name] = {
                    "type": arg.type,
                    "description": arg.description,
                }
                if arg.required:
                    openai_tool["function"]["parameters"]["required"].append(arg.name)
            openai_tools.append(openai_tool)
        return openai_tools 
        

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self, 
        model_name: str = BASE_EMBEDDING_MODEL, 
        dimensions: int | None = None
    ) -> None:
        self.model_name = model_name
        env_dims = dimensions or get_env_int("OPENAI_EMBEDDING_DIMENSIONS")
        self.client = OpenAIEmbeddings(model=model_name, dimensions=env_dims)
        self.dimensions = env_dims or self._set_dimension()
        dim_source = "env" if env_dims is not None else "probe"
        logger.info(
            "OpenAIEmbeddingProvider initialized: model=%s dims=%s source=%s",
            self.model_name,
            self.dimensions,
            dim_source,
        )
        super().__init__(
            provider=OPENAI, 
            client=self.client, 
            model_name=self.model_name, 
            dimensions=self.dimensions
        )
    
    def _set_dimension(self) -> int:
        try:
            embedding = self.client.embed_query("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the LangChain embeddings: {str(ke)}")
        except Exception as e:
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")