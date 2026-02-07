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
from ai_server.types.message import MessageDTO, Role, MessageAITextPart, MessageToolPart, ToolPartState
from ai_server.config import BASE_MODEL
from ai_server.utils.tracing import trace_method
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from openinference.semconv.trace import OpenInferenceSpanKindValues

import json
from pydantic import ValidationError
import openai
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict


class OpenAIChatCompletionAPI(OpenAIProvider):

    @classmethod
    def _convert_to_openai_compatible_messages(cls, messages: List[MessageDTO]) -> List[Dict]:
        converted_messages = []
        for message in messages:
            match message.role:
                case Role.HUMAN:
                    for part in message.parts:
                        converted_messages.append({
                            "role": message.role.value,
                            "content": part.text,
                        })
                case Role.AI:
                    for part in message.parts:
                        if isinstance(part, MessageAITextPart):
                            converted_messages.append({
                                "role": message.role.value,
                                "content": part.text,
                            })
                        elif isinstance(part, MessageToolPart):
                            # Add assistant message with tool call
                            if part.toolCallId and part.input is not None and (
                                part.state == ToolPartState.INPUT_AVAILABLE or
                                part.state == ToolPartState.OUTPUT_AVAILABLE or
                                part.state == ToolPartState.OUTPUT_ERROR
                            ):
                                converted_messages.append({
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "id": part.toolCallId,
                                        "type": "function",
                                        "function": {
                                            "name": part.tool_name,
                                            "arguments": json.dumps(part.input),
                                        },
                                    }]
                                })
                            # Add tool response message
                            if part.toolCallId and part.output and part.state == ToolPartState.OUTPUT_AVAILABLE:
                                converted_messages.append({
                                    "role": "tool",
                                    "tool_call_id": part.toolCallId,
                                    "content": json.dumps(part.output) if isinstance(part.output, dict) else str(part.output),
                                })
                            # Handle tool error state - send error message back to model
                            elif part.toolCallId and part.errorText and part.state == ToolPartState.OUTPUT_ERROR:
                                error_content = json.dumps({
                                    "error": True,
                                    "error_message": part.errorText,
                                    "note": "This tool encountered an error. Please inform the user about this issue and do not retry this tool call."
                                })
                                converted_messages.append({
                                    "role": "tool",
                                    "tool_call_id": part.toolCallId,
                                    "content": error_content,
                                })
                case Role.SYSTEM:
                    for part in message.parts:
                        converted_messages.append({
                            "role": "developer" if message.role == Role.SYSTEM else message.role.value,
                            "content": part.text,
                        })
        return converted_messages

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
        ai_message: MessageDTO,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> bool:
        output = response.choices[0].message
        content = output.content
        tool_calls = output.tool_calls

        # Collect function calls
        function_call_tasks = []
        
        try:
            # Add text content if present
            if content:
                ai_message.update_ai_text_message(text=content)
            
            # Add tool calls if present
            if tool_calls:
                for tool_call in tool_calls:
                    ai_message.update_ai_tool_input_message(
                        tool_name=tool_call.function.name,
                        tool_call_id=tool_call.id,
                        input_data=json.loads(tool_call.function.arguments)
                    )
                    # Create task for parallel execution
                    task = cls._call_function(
                        function_name=tool_call.function.name,
                        function_arguments=json.loads(tool_call.function.arguments),
                        tools=tools
                    )
                    function_call_tasks.append((tool_call.id, task))
                
                # Execute all function calls in parallel and update tool parts
                await cls._process_tool_call_responses(
                    function_call_tasks=function_call_tasks,
                    ai_message=ai_message,
                    stream=stream,
                    on_stream_event=on_stream_event,
                )
                return True
            return False
        except ValidationError as e:
            raise MessageParseException(message="Failed to parse AI response from openai chat completion", note=str(e))

    @classmethod
    async def _stream_chat_completion(
        cls,
        client: openai.AsyncOpenAI,
        request_kwargs: Dict,
        on_stream_event: StreamCallback | None = None,
        message_id: str | None = None,
    ) -> ChatCompletion:
        """Stream chat completion chunks and dispatch events."""
        tool_call_buffers: Dict[int, Dict] = {}
        content_started = False
        start_emitted = False
        openai_chunk_id: str | None = None  # Track OpenAI's chunk id for text events
        
        # Use .stream() context manager which provides get_final_completion()
        async with client.chat.completions.stream(**request_kwargs) as stream:
            async for event in stream:
                event_type = getattr(event, "type", "")
                
                # Handle chunk events (raw ChatCompletionChunk)
                if event_type == "chunk":
                    chunk = event.chunk
                    
                    # Capture OpenAI's chunk id (chatcmpl-xxx) for text events
                    if openai_chunk_id is None and chunk.id:
                        openai_chunk_id = chunk.id
                    
                    if not chunk.choices:
                        continue
                        
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Emit start event with message ID from first chunk
                    if not start_emitted:
                        start_emitted = True
                        await dispatch_stream_event(
                            on_stream_event,
                            create_start_event(message_id),
                        )
                    
                    # Handle tool calls streaming from raw chunks
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
                            text_id = openai_chunk_id or message_id
                            await dispatch_stream_event(
                                on_stream_event,
                                create_text_end_event(text_id),
                            )
                        
                        # Emit tool input available for completed tool calls
                        if choice.finish_reason == "tool_calls":
                            for buffer in tool_call_buffers.values():
                                if buffer["id"]:
                                    # Handle empty arguments for tools with no parameters
                                    if buffer["arguments"]:
                                        try:
                                            parsed_args = json.loads(buffer["arguments"])
                                        except json.JSONDecodeError:
                                            parsed_args = buffer["arguments"]
                                    else:
                                        parsed_args = {}
                                    
                                    await dispatch_stream_event(
                                        on_stream_event,
                                        create_tool_input_available_event(buffer["id"], parsed_args, buffer["name"]),
                                    )
                        
                        # Note: finish event is emitted by base.py after the entire agentic loop completes
                
                # Handle content delta events (higher-level API)
                elif event_type == "content.delta":
                    # Use OpenAI's chunk id for text events (like Responses API uses item_id)
                    text_id = openai_chunk_id or message_id
                    if not content_started:
                        content_started = True
                        await dispatch_stream_event(
                            on_stream_event,
                            create_text_start_event(text_id),
                        )
                    await dispatch_stream_event(
                        on_stream_event,
                        create_text_delta_event(text_id, event.delta),
                    )
            
            # Get final completion from stream (like Responses API)
            final_completion = await stream.get_final_completion()
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
        message_id: str | None = None,
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
        return await cls._stream_chat_completion(client=client, request_kwargs=kwargs, on_stream_event=on_stream_event, message_id=message_id)

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
