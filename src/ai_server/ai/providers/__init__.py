from ai_server.ai.providers.llm_provider import LLMProvider
from ai_server.ai.providers.openai import OpenAIChatCompletionAPI, OpenAIResponsesAPI
from ai_server.config import OPENAI, ENABLE_CHAT_COMPLETION

def get_llm_provider(provider_name: str, **kwargs) -> LLMProvider:
    if provider_name == OPENAI:
        chat_completion_api = ENABLE_CHAT_COMPLETION
        if chat_completion_api:
            return OpenAIChatCompletionAPI
        return OpenAIResponsesAPI
    raise ValueError(f"Unknown provider: {provider_name}")