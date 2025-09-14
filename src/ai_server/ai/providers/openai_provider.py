
from ai_server.constants import GPT_4_1

from ai_server.api.exceptions.openai_exceptions import UnrecognizedMessageTypeException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from ai_server.ai.providers.llm_provider import LLMProvider
from ai_server.ai.tools.tools import Tool

from ai_server.schemas.message import Message, Role, FunctionCallRequest

import asyncio
import json
from pydantic import ValidationError

import openai
from openai.types.responses import Response
from openai.types.chat.chat_completion import ChatCompletion

import os
from typing import List, Dict
from abc import ABC, abstractmethod

class OpenAIProvider(LLMProvider, ABC):
    def __init__(self, temperature: float = 0.7) -> None:
        super().__init__("openai")
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @abstractmethod
    def generate_response(
        self, 
        query: str, 
        conversation_history: List[Message], 
        user_id: str, 
        session_id:str, 
        turn_id: str,
        tools: List[Tool] = [], 
        tool_choice: str = "auto",
        model_name: str = GPT_4_1
    ) -> List[Message]:
        pass

class OpenAIResponsesAPI(OpenAIProvider):
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

    async def generate_response(
        self, 
        query: str | None, 
        conversation_history: List[Message], 
        user_id: str, 
        session_id: str, 
        turn_id: str,
        tools: List[Tool] = [], 
        tool_choice: str = "auto",
        model_name: str = GPT_4_1
    ) -> List[Message]:
        if query:
            formatted_query = Message(
                role=Role.HUMAN,
                tool_call_id="null",
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                metadata={},
                content=query,
                function_call=None,
            )
            input_messages = conversation_history + [formatted_query]
        else:
            input_messages = conversation_history
        input_messages = list(map(self._convert_to_openai_compatible_messages, input_messages))
        response = self.client.responses.create(
            model=model_name,
            input=input_messages,
            temperature=self.temperature,
            tools=self._convert_tools_to_openai_compatible(tools),
            tool_choice=tool_choice,
        )
        ai_messages = await self._handle_ai_messages_and_tool_calls(response, user_id, session_id, turn_id, tools)
        return [formatted_query, *ai_messages] if query else ai_messages

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

    async def generate_response(
        self, 
        query: str | None, 
        conversation_history: List[Message], 
        user_id: str, 
        session_id: str, 
        turn_id: str,
        tools: List[Tool] = [], 
        tool_choice: str = "auto",
        model_name: str = GPT_4_1
    ) -> List[Message]:
        if query:
            formatted_query = Message(
                role=Role.HUMAN,
                tool_call_id="null",
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                metadata={},
                content=query,
                function_call=None,
            )
            input_messages = conversation_history + [formatted_query]
        else:
            input_messages = conversation_history
        input_messages = list(map(self._convert_to_openai_compatible_messages, input_messages))
        response = self.client.chat.completions.create(
            model=model_name,
            messages=input_messages,
            temperature=self.temperature,
            tools=self._convert_tools_to_openai_compatible(tools),
            tool_choice=tool_choice,
        )
        ai_messages = await self._handle_ai_messages_and_tool_calls(response, user_id, session_id, turn_id, tools)
        return [formatted_query, *ai_messages] if query else ai_messages

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
        