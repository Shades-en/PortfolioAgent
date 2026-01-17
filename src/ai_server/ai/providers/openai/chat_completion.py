from ai_server.ai.providers.openai.base import OpenAIProvider
from ai_server.ai.providers.llm_provider import StreamCallback, dispatch_stream_event
from ai_server.ai.providers.utils import (
    create_start_event,
    create_text_start_event,
    create_text_delta_event,
    create_text_end_event,
    create_tool_input_start_event,
    create_tool_input_delta_event,
    create_tool_input_available_event,
)
from ai_server.ai.tools.tools import Tool
from ai_server.types.message import MessageDTO, Role, FunctionCallRequest
from ai_server.config import BASE_MODEL
from ai_server.utils.tracing import trace_method
from ai_server.utils.general import generate_order
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from openinference.semconv.trace import OpenInferenceSpanKindValues

import json
from pydantic import ValidationError
import openai
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict


class OpenAIChatCompletionAPI(OpenAIProvider):

    @classmethod
    def _convert_to_openai_compatible_messages(cls, message: MessageDTO) -> Dict:
        role = "user"
        match message.role:
            case Role.HUMAN:
                role = "user"
            case Role.AI:
                role = "assistant"
                if message.tool_call_id is not None:
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

    @classmethod
    @trace_method(
        kind=OpenInferenceSpanKindValues.CHAIN,
        graph_node_id="response_handler",
        capture_input=False,
        capture_output=False
    )
    async def _handle_ai_messages_and_tool_calls(
        cls, 
        response: ChatCompletion, 
        tools: List[Tool],
        step: int,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> List[MessageDTO]:
        output = response.choices[0].message
        messages: List[MessageDTO] = []
        content = output.content
        tool_calls = output.tool_calls

        order = 2  # 1 prefix is assigned to human message, 2 prefix for AI
        try:
            if content:
                message = MessageDTO(
                    role=Role.AI,
                    metadata={},
                    content=content,
                    function_call=None,
                    order=generate_order(step, order),
                    response_id=response.id
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
                    message_ai = MessageDTO(
                        role=Role.AI,
                        tool_call_id=tool_call.id,
                        metadata={},
                        content='',
                        function_call=function_call,
                        order=generate_order(step, order),
                        response_id=response.id
                    )
                    order += 1
                    messages.append(message_ai)

                    # Create task for parallel execution
                    task = cls._call_function(function_call, tools)
                    function_call_tasks.append(task)
                    function_call_messages.append((message_ai, tool_call.id))
                
                # Execute all function calls in parallel and create tool messages
                tool_messages = await cls._process_tool_call_responses(
                    function_call_tasks=function_call_tasks,
                    function_call_messages=function_call_messages,
                    response_id=response.id,
                    step=step,
                    order=order,
                    stream=stream,
                    on_stream_event=on_stream_event,
                )
                messages.extend(tool_messages)
            return messages
        except ValidationError as e:
            raise MessageParseException(message="Failed to parse AI response from openai responses", note=str(e))

    @classmethod
    async def _stream_chat_completion(
        cls,
        client: openai.AsyncOpenAI,
        request_kwargs: Dict,
        on_stream_event: StreamCallback | None = None,
    ) -> ChatCompletion:
        """Stream chat completion chunks and dispatch events."""
        tool_call_buffers: Dict[int, Dict] = {}
        content_started = False
        message_id = None
        
        stream = await client.chat.completions.create(**request_kwargs, stream=True)
        
        async for chunk in stream:
            if not chunk.choices:
                continue
                
            choice = chunk.choices[0]
            delta = choice.delta
            
            # Emit start event with message ID from first chunk
            if message_id is None:
                message_id = chunk.id
                await dispatch_stream_event(
                    on_stream_event,
                    create_start_event(message_id),
                )
            
            # Handle content streaming
            if delta.content:
                if not content_started:
                    content_started = True
                    await dispatch_stream_event(
                        on_stream_event,
                        create_text_start_event(message_id),
                    )
                await dispatch_stream_event(
                    on_stream_event,
                    create_text_delta_event(message_id, delta.content),
                )
            
            # Handle tool calls streaming
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index
                    
                    # Initialize buffer for this tool call
                    if index not in tool_call_buffers:
                        tool_call_buffers[index] = {
                            "id": tool_call_delta.id,
                            "name": None,
                            "arguments": "",
                            "started": False,
                        }
                    
                    buffer = tool_call_buffers[index]
                    
                    # Update buffer with new data
                    if tool_call_delta.id:
                        buffer["id"] = tool_call_delta.id
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            buffer["name"] = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            buffer["arguments"] += tool_call_delta.function.arguments
                    
                    # Emit start event when we have the tool call ID
                    if buffer["id"] and not buffer["started"]:
                        buffer["started"] = True
                        await dispatch_stream_event(
                            on_stream_event,
                            create_tool_input_start_event(buffer["id"], buffer["name"]),
                        )
                    
                    # Emit delta event for arguments
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        await dispatch_stream_event(
                            on_stream_event,
                            create_tool_input_delta_event(buffer["id"], tool_call_delta.function.arguments),
                        )
            
            # Handle finish reason
            if choice.finish_reason:
                # Emit text end if content was streamed
                if content_started:
                    await dispatch_stream_event(
                        on_stream_event,
                        create_text_end_event(message_id),
                    )
                
                # Emit tool input available for completed tool calls
                if choice.finish_reason == "tool_calls":
                    for buffer in tool_call_buffers.values():
                        if buffer["id"] and buffer["arguments"]:
                            try:
                                parsed_args = json.loads(buffer["arguments"])
                            except json.JSONDecodeError:
                                parsed_args = buffer["arguments"]
                            
                            await dispatch_stream_event(
                                on_stream_event,
                                create_tool_input_available_event(buffer["id"], parsed_args),
                            )
        
        # Get final completion by creating a non-streaming request
        # (OpenAI's streaming API doesn't provide a get_final_completion equivalent)
        final_completion = await client.chat.completions.create(**request_kwargs, stream=False)
        return final_completion

    @classmethod
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
    ) -> ChatCompletion:
        """Generic wrapper for OpenAI Chat Completions API calls."""
        kwargs = {
            "model": model,
            "messages": input_messages,
            "temperature": temperature if temperature is not None else cls.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        # Note: instructions not supported in Chat Completions API, use system message instead
        
        client = cls._get_client()
        if not stream:
            return await client.chat.completions.create(**kwargs)
        return await cls._stream_chat_completion(client=client, request_kwargs=kwargs, on_stream_event=on_stream_event)

    @classmethod
    def _convert_tools_to_openai_compatible(cls, tools: List[Tool]) -> List[Dict]:
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
