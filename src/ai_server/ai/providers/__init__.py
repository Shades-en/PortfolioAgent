from ai_server.ai.providers.llm_provider import LLMProvider
from ai_server.ai.providers.openai import OpenAIChatCompletionAPI, OpenAIResponsesAPI
from ai_server.config import OPENAI
from typing import Literal

def get_llm_provider(provider_name: str, api_type: Literal["responses", "chat_completion"] = "responses", **kwargs) -> LLMProvider:
    if provider_name == OPENAI:
        if api_type == "chat_completion":
            return OpenAIChatCompletionAPI
        return OpenAIResponsesAPI
    raise ValueError(f"Unknown provider: {provider_name}")