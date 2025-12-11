from ai_server.ai.providers.llm_provider import LLMProvider
from ai_server.ai.providers.openai_provider import OpenAIChatCompletionAPI, OpenAIResponsesAPI
from ai_server.config import OPENAI

def get_llm_provider(provider_name: str, **kwargs) -> LLMProvider:
    if provider_name == OPENAI:
        chat_completion_api = kwargs.get("chat_completion_api")
        if chat_completion_api:
            return OpenAIChatCompletionAPI
        return OpenAIResponsesAPI
    raise ValueError(f"Unknown provider: {provider_name}")