from pydantic import BaseModel, model_validator, Field
from enum import Enum
from typing import Self
from datetime import datetime, timezone
import re

from ai_server.utils.general import get_token_count

class Feedback(Enum):
    LIKE = "liked"
    DISLIKE = "disliked"

class Role(Enum):
    HUMAN = 'user'
    SYSTEM = 'system'
    AI = 'assistant'

class ToolPartState(Enum):
    INPUT_AVAILABLE = 'input-available'
    OUTPUT_AVAILABLE = 'output-available'
    OUTPUT_ERROR = 'output-error'    

class MessageHumanTextPart(BaseModel):
    type: str = "text"
    text: str
    token_count: int = 0

    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from text if not explicitly provided."""
        if self.token_count == 0 and self.text:
            self.token_count = get_token_count(self.text)
        return self

class MessageAITextPart(BaseModel):
    type: str = "text"
    text: str
    state: str = "done"
    token_count: int = 0

    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from text if not explicitly provided."""
        if self.token_count == 0 and self.text:
            self.token_count = get_token_count(self.text)
        return self
    
class MessageReasoningPart(BaseModel):
    type: str = "reasoning"
    text: str
    state: str = "done"
    token_count: int = 0

    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from text if not explicitly provided."""
        if self.token_count == 0 and self.text:
            self.token_count = get_token_count(self.text)
        return self

class MessageToolPart(BaseModel):
    type: str = "" # Tool class name (e.g., "GetWeather", "GetHoroscope") but with "tool-" prefix
    tool_name: str  # Tool class name (e.g., "GetWeather", "GetHoroscope") 
    toolCallId: str
    state: ToolPartState = ToolPartState.OUTPUT_AVAILABLE
    input: dict | None = None
    output: dict | None = None
    errorText: str | None = None
    input_token_count: int = 0
    output_token_count: int = 0
    
    @model_validator(mode="after")
    def compute_token_count(self) -> Self:
        """Compute token count from output if not explicitly provided."""
        if self.input_token_count == 0 and self.input and self.state == ToolPartState.INPUT_AVAILABLE:
            self.input_token_count = get_token_count(str(self.input))
        if self.output_token_count == 0:
            if self.state == ToolPartState.OUTPUT_AVAILABLE and self.output:
                self.output_token_count = get_token_count(str(self.output))
            elif self.state == ToolPartState.OUTPUT_ERROR and self.errorText:
                self.output_token_count = get_token_count(self.errorText)
        return self
    
    @model_validator(mode="after")
    def set_type_from_tool_name(self) -> Self:
        """Convert tool class name to snake_case and set type as 'tool-{snake_case_name}'."""
        # Convert CamelCase to snake_case and set type if not already set
        if self.tool_name and (not self.type or not self.type.startswith("tool-")):
            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', self.tool_name).lower()
            self.type = f"tool-{snake_case}"
        return self


class MessageDTO(BaseModel):
    """
    Data Transfer Object for Message.
    Used for in-memory message representation in conversation flows.
    Does not include DB-specific fields like session Link.
    """
    id: str | None = Field(default=None, description="Frontend-generated message ID (e.g., from AI SDK)")
    role: Role
    metadata: dict = Field(default_factory=dict)
    parts: list[MessageHumanTextPart | MessageAITextPart | MessageReasoningPart | MessageToolPart] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feedback: Feedback | None = None

    @classmethod
    def create_system_message(cls, text: str, message_id: str, metadata: dict = None) -> Self:
        """Create a system message with the given text."""
        return cls(
            id=message_id,
            role=Role.SYSTEM,
            metadata=metadata or {},
            parts=[MessageHumanTextPart(text=text)],
            created_at=datetime.now(timezone.utc)
        )

    @classmethod
    def create_human_message(cls, text: str, message_id: str, metadata: dict = None) -> Self:
        """Create a human message with the given text."""
        return cls(
            id=message_id,
            role=Role.HUMAN,
            metadata=metadata or {},
            parts=[MessageHumanTextPart(text=text)],
            created_at=datetime.now(timezone.utc)
        )

    @classmethod
    def create_ai_message(cls, message_id: str, metadata: dict = None) -> Self:
        """Create an AI message with the given text."""
        return cls(
            id=message_id,
            role=Role.AI,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc)
        )

    def update_ai_text_message(self, text: str, metadata: dict = None) -> Self:
        """Add a text part to this AI message with the given text."""
        if self.role != Role.AI:
            raise TypeError(f"update_ai_text_message can only be called on AI messages, not {self.role.value}")
        
        self.metadata.update(metadata or {})
        self.parts.append(MessageAITextPart(text=text))
        return self

    def update_ai_tool_input_message(
        self, 
        tool_name: str, 
        tool_call_id: str,
        input_data: dict,
        metadata: dict = None
    ) -> Self:
        """Add a tool input part to this AI message with the given tool name, call ID, and input data."""
        if self.role != Role.AI:
            raise TypeError(f"update_ai_tool_input_message can only be called on AI messages, not {self.role.value}")
        
        self.metadata.update(metadata or {})

        parts = [MessageToolPart(
            tool_name=tool_name, 
            toolCallId=tool_call_id, 
            state=ToolPartState.INPUT_AVAILABLE,
            input=input_data
        )]
        self.parts.extend(parts)
        
        return self

    def update_ai_tool_output_message(
        self, 
        tool_call_id: str, 
        result: dict = None,
        metadata: dict = None
    ) -> Self:
        """Update the tool part with the given toolCallId to add output and change state to OUTPUT_AVAILABLE."""
        if self.role != Role.AI:
            raise TypeError(f"update_ai_tool_output_message can only be called on AI messages, not {self.role.value}")
        
        self.metadata.update(metadata or {})
        
        # Find the MessageToolPart with matching toolCallId
        for part in self.parts:
            if isinstance(part, MessageToolPart) and part.toolCallId == tool_call_id:
                part.output = result
                part.state = ToolPartState.OUTPUT_AVAILABLE
                break
        
        return self
    
    def update_ai_tool_error_message(
        self, 
        tool_call_id: str, 
        error_text: str,
        metadata: dict = None
    ) -> Self:
        """Update the tool part with the given toolCallId to add error text and change state to OUTPUT_ERROR."""
        if self.role != Role.AI:
            raise TypeError(f"update_ai_tool_error_message can only be called on AI messages, not {self.role.value}")
        
        self.metadata.update(metadata or {})
        
        # Find the MessageToolPart with matching toolCallId
        for part in self.parts:
            if isinstance(part, MessageToolPart) and part.toolCallId == tool_call_id:
                part.errorText = error_text
                part.state = ToolPartState.OUTPUT_ERROR
                break
        
        return self
        



