from abc import ABC, abstractmethod
from typing import List

from ai_server.schemas.message import Message, FunctionCallRequest
from ai_server.ai.tools.tools import Tool
from ai_server.utils.singleton import SingletonABCMeta

class LLMProvider(ABC, metaclass=SingletonABCMeta):
    def __init__(self, provider: str) -> None:
        self.provider = provider

    @abstractmethod
    async def generate_response(
        self, 
        query: str, 
        conversation_history: List[Message], 
        user_id: str, 
        session_id: str, 
        turn_id: str,
        tools: List[Tool] = []
    ) -> List[Message]:
        pass

    async def _call_function(self, function_call_request: FunctionCallRequest, tools: List[Tool]) -> str:
        for tool in tools:
            if tool.name == function_call_request.name:
                return await tool(tool.Arguments(**function_call_request.arguments))
        return ""