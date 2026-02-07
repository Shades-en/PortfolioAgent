from ai_server.ai.providers.openai.base import OpenAIProvider
from ai_server.ai.providers.llm_provider import StreamCallback, dispatch_stream_event
from ai_server.ai.providers.utils import (
    create_start_event,
    create_text_start_event,
    create_text_delta_event,
    create_text_end_event,
    create_reasoning_start_event,
    create_reasoning_delta_event,
    create_reasoning_end_event,
    create_tool_input_start_event,
    create_tool_input_delta_event,
    create_tool_input_available_event,
    create_error_event,
)
from ai_server.ai.tools.tools import Tool

from ai_server.types.message import MessageAITextPart, MessageDTO, MessageToolPart, Role, ToolPartState
from ai_server.config import BASE_MODEL
from ai_server.constants import (
    OPENAI_EVENT_RESPONSE_CREATED,
    OPENAI_EVENT_TEXT_DELTA,
    OPENAI_EVENT_TEXT_DONE,
    OPENAI_EVENT_REASONING_DELTA,
    OPENAI_EVENT_REASONING_DONE,
    OPENAI_EVENT_FUNCTION_ARGS_DELTA,
    OPENAI_EVENT_OUTPUT_ITEM_ADDED,
    OPENAI_EVENT_OUTPUT_ITEM_DONE,
    OPENAI_EVENT_FAILED,
)

from ai_server.utils.tracing import trace_method

from ai_server.api.exceptions.openai_exceptions import UnrecognizedMessageTypeException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from openinference.semconv.trace import OpenInferenceSpanKindValues

import asyncio
import json
from pydantic import ValidationError
import openai
from openai.types.responses import Response
from typing import List, Dict


class OpenAIResponsesAPI(OpenAIProvider):

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
    ) -> Response:
        """Generic wrapper for OpenAI Responses API calls."""
        kwargs = {
            "model": model,
            "input": input_messages,
            "temperature": temperature if temperature is not None else cls.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if instructions:
            kwargs["instructions"] = instructions
        client = cls._get_client()
        if not stream:
            return await client.responses.create(**kwargs)
        return await cls._stream_responses(client=client, request_kwargs=kwargs, on_stream_event=on_stream_event, message_id=message_id)

    @classmethod
    async def _stream_responses(
        cls,
        client: openai.AsyncOpenAI,
        request_kwargs: Dict,
        on_stream_event: StreamCallback | None = None,
        message_id: str | None = None,
    ) -> Response:
        text_started: set[str] = set()
        reasoning_started: set[str] = set()
        # Track tool calls: item_id (fc_xxx) -> {call_id, name, started}
        tool_call_info: dict[str, dict] = {}
        async with client.responses.stream(**request_kwargs) as response_stream:
            async for event in response_stream:
                event_type = getattr(event, "type", "")
                event_item = getattr(event, "item", None)
                
                match event_type:
                    case _ if event_type == OPENAI_EVENT_RESPONSE_CREATED:
                        await dispatch_stream_event(
                            on_stream_event,
                            create_start_event(message_id),
                        )
                    case _ if event_type == OPENAI_EVENT_OUTPUT_ITEM_ADDED:
                        # Extract tool call info when function_call item is added
                        # Map item.id (fc_xxx) to call_id (call_xxx) and tool name
                        if event_item and getattr(event_item, "type", None) == "function_call":
                            item_id = getattr(event_item, "id", None)
                            item_call_id = getattr(event_item, "call_id", None)
                            item_tool_name = getattr(event_item, "name", None)
                            if item_id and item_call_id:
                                tool_call_info[item_id] = {"call_id": item_call_id, "name": item_tool_name}
                                # Emit tool-input-start immediately when tool call is added
                                await dispatch_stream_event(
                                    on_stream_event,
                                    create_tool_input_start_event(item_call_id, item_tool_name),
                                )
                    case _ if event_type == OPENAI_EVENT_TEXT_DELTA:
                        text_id = getattr(event, "item_id", None)
                        if text_id and text_id not in text_started:
                            text_started.add(text_id)
                            await dispatch_stream_event(on_stream_event, create_text_start_event(text_id))
                        if text_id:
                            await dispatch_stream_event(
                                on_stream_event,
                                create_text_delta_event(text_id, getattr(event, "delta", "")),
                            )
                    case _ if event_type == OPENAI_EVENT_TEXT_DONE:
                        text_id = getattr(event, "item_id", None)
                        if text_id:
                            await dispatch_stream_event(on_stream_event, create_text_end_event(text_id))
                    case _ if event_type == OPENAI_EVENT_REASONING_DELTA:
                        reasoning_id = getattr(event, "item_id", None)
                        if reasoning_id and reasoning_id not in reasoning_started:
                            reasoning_started.add(reasoning_id)
                            await dispatch_stream_event(on_stream_event, create_reasoning_start_event(reasoning_id))
                        if reasoning_id:
                            await dispatch_stream_event(
                                on_stream_event,
                                create_reasoning_delta_event(reasoning_id, getattr(event, "delta", "")),
                            )
                    case _ if event_type == OPENAI_EVENT_REASONING_DONE:
                        reasoning_id = getattr(event, "item_id", None)
                        if reasoning_id:
                            await dispatch_stream_event(on_stream_event, create_reasoning_end_event(reasoning_id))
                    case _ if event_type == OPENAI_EVENT_FUNCTION_ARGS_DELTA:
                        # Get item_id (fc_xxx) from the event to look up call_id
                        current_item_id = getattr(event, "item_id", None)
                        if current_item_id and current_item_id in tool_call_info:
                            call_id = tool_call_info[current_item_id]["call_id"]
                            # Emit tool-input-delta
                            await dispatch_stream_event(
                                on_stream_event,
                                create_tool_input_delta_event(call_id, getattr(event, "delta", "")),
                            )
                    case _ if event_type == OPENAI_EVENT_OUTPUT_ITEM_DONE:
                        # Extract tool call info and dispatch tool-input-available
                        if event_item and getattr(event_item, "type", None) == "function_call":
                            item_call_id = getattr(event_item, "call_id", None)
                            item_tool_name = getattr(event_item, "name", None)
                            item_arguments = getattr(event_item, "arguments", None)
                            parsed_args = cls._safe_json_loads(item_arguments)
                            if item_call_id:
                                await dispatch_stream_event(
                                    on_stream_event,
                                    create_tool_input_available_event(item_call_id, parsed_args, item_tool_name),
                                )
                    case _ if event_type == OPENAI_EVENT_FAILED:
                        error_obj = getattr(event, "error", None)
                        if isinstance(error_obj, dict):
                            error_text = error_obj.get("message")
                        else:
                            error_text = getattr(error_obj, "message", None) if error_obj else None
                        await dispatch_stream_event(
                            on_stream_event,
                            create_error_event(error_text or "response.failed"),
                        )

            final_response = await response_stream.get_final_response()
            return final_response

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
                            if part.toolCallId and part.input is not None and (
                                part.state == ToolPartState.INPUT_AVAILABLE or
                                part.state == ToolPartState.OUTPUT_AVAILABLE or
                                part.state == ToolPartState.OUTPUT_ERROR
                            ):
                                tool_input_message = {
                                    "call_id": part.toolCallId,
                                    "type": "function_call",
                                    "name": part.tool_name,
                                    "arguments": json.dumps(part.input),
                                }
                                converted_messages.append(tool_input_message)
                            if part.toolCallId and part.output and part.state == ToolPartState.OUTPUT_AVAILABLE:
                                output_value = part.output if isinstance(part.output, str) else json.dumps(part.output)
                                tool_output_message = {
                                    "call_id": part.toolCallId,
                                    "type": "function_call_output",
                                    "output": output_value,
                                }
                                converted_messages.append(tool_output_message)
                            # Handle tool error state - send error message back to model
                            elif part.toolCallId and part.errorText and part.state == ToolPartState.OUTPUT_ERROR:
                                error_output = json.dumps({
                                    "error": True,
                                    "error_message": part.errorText,
                                    "note": "This tool encountered an error. Please inform the user about this issue and do not retry this tool call."
                                })
                                tool_error_message = {
                                    "call_id": part.toolCallId,
                                    "type": "function_call_output",
                                    "output": error_output,
                                }
                                converted_messages.append(tool_error_message)
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
        response: Response, 
        tools: List[Tool],
        ai_message: MessageDTO | None = None,
        stream: bool = False,
        on_stream_event: StreamCallback | None = None,
    ) -> bool:
        outputs = response.output
        
        # Collect function calls
        function_call_tasks = []
        
        try:
            for resp in outputs:
                if resp.type == "function_call":
                    ai_message.update_ai_tool_input_message(
                        tool_name=resp.name,
                        tool_call_id=resp.call_id,
                        input_data=json.loads(resp.arguments)
                    )
                    # Create task for parallel execution
                    task = cls._call_function(function_name=resp.name, function_arguments=json.loads(resp.arguments), tools=tools)
                    if resp.call_id:
                        function_call_tasks.append((resp.call_id, task))
                    
                elif resp.type == "message":
                    ai_message.update_ai_text_message(text=resp.content[0].text)
                else:
                    raise UnrecognizedMessageTypeException(message="Unrecognized message type", note=f"Message type: {resp.type} - Implementation does not exist")
            
            # Execute all function calls in parallel and create tool messages
            if function_call_tasks:
                await asyncio.shield(cls._process_tool_call_responses(
                    function_call_tasks=function_call_tasks,
                    ai_message=ai_message,
                    stream=stream,
                    on_stream_event=on_stream_event,
                ))
                return True
            return False
        except ValidationError as e:
            raise MessageParseException(message="Failed to parse AI response from openai responses", note=str(e))

    @classmethod
    def _convert_tools_to_openai_compatible(cls, tools: List[Tool]) -> List[Dict]:
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
