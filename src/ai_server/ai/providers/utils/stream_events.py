"""Centralized stream event builders for AI SDK stream protocol.

This module provides factory functions to create stream event dictionaries,
ensuring consistency across different LLM provider implementations and making
it easier to maintain event structures.

The event builders follow the AI SDK stream protocol specification.
"""

from typing import Dict, Any
from ai_server.constants import (
    STREAM_EVENT_START,
    STREAM_EVENT_TEXT_START,
    STREAM_EVENT_TEXT_DELTA,
    STREAM_EVENT_TEXT_END,
    STREAM_EVENT_REASONING_START,
    STREAM_EVENT_REASONING_DELTA,
    STREAM_EVENT_REASONING_END,
    STREAM_EVENT_TOOL_INPUT_START,
    STREAM_EVENT_TOOL_INPUT_DELTA,
    STREAM_EVENT_TOOL_INPUT_AVAILABLE,
    STREAM_EVENT_TOOL_OUTPUT_AVAILABLE,
    STREAM_EVENT_ERROR,
    STREAM_EVENT_FINISH,
)


def create_start_event(message_id: str) -> Dict[str, Any]:
    """Create a stream start event."""
    return {
        "type": STREAM_EVENT_START,
        "messageId": message_id,
    }


def create_text_start_event(text_id: str) -> Dict[str, Any]:
    """Create a text stream start event."""
    return {
        "type": STREAM_EVENT_TEXT_START,
        "id": text_id,
    }


def create_text_delta_event(text_id: str, delta: str) -> Dict[str, Any]:
    """Create a text delta event."""
    return {
        "type": STREAM_EVENT_TEXT_DELTA,
        "id": text_id,
        "delta": delta,
    }


def create_text_end_event(text_id: str) -> Dict[str, Any]:
    """Create a text stream end event."""
    return {
        "type": STREAM_EVENT_TEXT_END,
        "id": text_id,
    }


def create_reasoning_start_event(reasoning_id: str) -> Dict[str, Any]:
    """Create a reasoning stream start event."""
    return {
        "type": STREAM_EVENT_REASONING_START,
        "id": reasoning_id,
    }


def create_reasoning_delta_event(reasoning_id: str, delta: str) -> Dict[str, Any]:
    """Create a reasoning delta event."""
    return {
        "type": STREAM_EVENT_REASONING_DELTA,
        "id": reasoning_id,
        "delta": delta,
    }


def create_reasoning_end_event(reasoning_id: str) -> Dict[str, Any]:
    """Create a reasoning stream end event."""
    return {
        "type": STREAM_EVENT_REASONING_END,
        "id": reasoning_id,
    }


def create_tool_input_start_event(
    tool_call_id: str,
    tool_name: str | None = None,
) -> Dict[str, Any]:
    """Create a tool input stream start event."""
    event: Dict[str, Any] = {
        "type": STREAM_EVENT_TOOL_INPUT_START,
        "toolCallId": tool_call_id,
    }
    if tool_name is not None:
        event["toolName"] = tool_name
    return event


def create_tool_input_delta_event(
    tool_call_id: str,
    input_text_delta: str,
) -> Dict[str, Any]:
    """Create a tool input delta event."""
    return {
        "type": STREAM_EVENT_TOOL_INPUT_DELTA,
        "toolCallId": tool_call_id,
        "inputTextDelta": input_text_delta,
    }


def create_tool_input_available_event(
    tool_call_id: str,
    input_data: Any,
) -> Dict[str, Any]:
    """Create a tool input available event."""
    return {
        "type": STREAM_EVENT_TOOL_INPUT_AVAILABLE,
        "toolCallId": tool_call_id,
        "input": input_data,
    }


def create_tool_output_available_event(
    tool_call_id: str,
    output_data: Any,
) -> Dict[str, Any]:
    """Create a tool output available event."""
    return {
        "type": STREAM_EVENT_TOOL_OUTPUT_AVAILABLE,
        "toolCallId": tool_call_id,
        "output": output_data,
    }


def create_error_event(error_text: str) -> Dict[str, Any]:
    """Create an error event."""
    return {
        "type": STREAM_EVENT_ERROR,
        "errorText": error_text,
    }


def create_finish_event() -> Dict[str, Any]:
    """Create a stream finish event."""
    return {
        "type": STREAM_EVENT_FINISH,
    }
