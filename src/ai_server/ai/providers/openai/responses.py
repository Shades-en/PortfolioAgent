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
    create_tool_output_available_event,
    create_error_event,
)
from ai_server.ai.tools.tools import Tool
from ai_server.types.message import MessageDTO, Role, FunctionCallRequest
from ai_server.config import BASE_MODEL
from ai_server.constants import (
    OPENAI_EVENT_RESPONSE_CREATED,
    OPENAI_EVENT_TEXT_DELTA,
    OPENAI_EVENT_TEXT_DONE,
    OPENAI_EVENT_REASONING_DELTA,
    OPENAI_EVENT_REASONING_DONE,
    OPENAI_EVENT_FUNCTION_ARGS_DELTA,
    OPENAI_EVENT_FUNCTION_ARGS_DONE,
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
        return await cls._stream_responses(client=client, request_kwargs=kwargs, on_stream_event=on_stream_event)

    @classmethod
    async def _stream_responses(
        cls,
        client: openai.AsyncOpenAI,
        request_kwargs: Dict,
        on_stream_event: StreamCallback | None = None,
    ) -> Response:
        text_started: set[str] = set()
        reasoning_started: set[str] = set()
        tool_inputs_started: set[str] = set()
        call_id = None
        async with client.responses.stream(**request_kwargs) as response_stream:
            async for event in response_stream:
                event_type = getattr(event, "type", "")
                event_item = getattr(event, "item", None)
                call_id = call_id or (getattr(event_item, "call_id", None) if event_item else None)
                
                match event_type:
                    case _ if event_type == OPENAI_EVENT_RESPONSE_CREATED:
                        response_obj = getattr(event, "response", None)
                        response_id = getattr(response_obj, "id", None)
                        await dispatch_stream_event(
                            on_stream_event,
                            create_start_event(response_id),
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
                        call_id = call_id or getattr(event, "item_id", None)
                        if call_id and call_id not in tool_inputs_started:
                            tool_inputs_started.add(call_id)
                            await dispatch_stream_event(
                                on_stream_event,
                                create_tool_input_start_event(call_id),
                            )
                        if call_id:
                            await dispatch_stream_event(
                                on_stream_event,
                                create_tool_input_delta_event(call_id, getattr(event, "delta", "")),
                            )
                    case _ if event_type == OPENAI_EVENT_FUNCTION_ARGS_DONE:
                        call_id = call_id or getattr(event, "item_id", None)
                        parsed_args = cls._safe_json_loads(getattr(event, "arguments", None))
                        if call_id:
                            await dispatch_stream_event(
                                on_stream_event,
                                create_tool_input_available_event(call_id, parsed_args),
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
    def _convert_to_openai_compatible_messages(cls, message: MessageDTO) -> Dict:
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
        on_stream_event: StreamCallback | None = None,
    ) -> List[MessageDTO]:
        outputs = response.output
        messages: List[MessageDTO] = []
        
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
                    message_ai = MessageDTO(
                        role=Role.AI,
                        tool_call_id=resp.call_id,
                        metadata={},
                        content='',
                        function_call=function_call,
                        order=2,
                        response_id=response.id
                    )
                    
                    # Create task for parallel execution
                    task = cls._call_function(function_call, tools)
                    function_call_tasks.append(task)
                    function_call_messages.append((message_ai, resp.call_id))
                    
                elif resp.type == "message":
                    message = MessageDTO(
                        role=Role.AI,
                        tool_call_id="null",
                        metadata={},
                        content=resp.content[0].text,
                        function_call=None,
                        order=4,
                        response_id=response.id
                    )
                    messages.append(message)
                else:
                    raise UnrecognizedMessageTypeException(message="Unrecognized message type", note=f"Message type: {resp.type} - Implementation does not exist")
            
            # Execute all function calls in parallel
            if function_call_tasks:
                function_responses = await asyncio.gather(*function_call_tasks)
                
                # Create tool messages with the responses
                for i, (message_ai, call_id) in enumerate(function_call_messages):
                    tool_output = function_responses[i]
                    if hasattr(tool_output, "model_dump"):
                        tool_output_payload = tool_output.model_dump()
                    else:
                        tool_output_payload = tool_output
                    await dispatch_stream_event(
                        on_stream_event,
                        create_tool_output_available_event(call_id, tool_output_payload),
                    )
                    message_tool = MessageDTO(
                        role=Role.TOOL,
                        tool_call_id=call_id,
                        metadata={},
                        content=str(tool_output),
                        function_call=None,
                        order=3,
                        response_id=response.id
                    )
                    messages.append(message_ai)
                    messages.append(message_tool)
            return messages
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
