from openinference.semconv.trace import OpenInferenceSpanKindValues
from ai_server.redis.client import RedisClient
from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.redis.session_manager import RedisSessionManager
from ai_server.redis.semantic_cache import ConversationMemoryCache
from ai_server.ai.providers.openai_provider import OpenAIChatCompletionAPI, OpenAIResponsesAPI, OpenAIEmbeddingProvider
from ai_server.schemas.message import Message, Role
from ai_server.utils.general import generate_id
from ai_server.ai.tools.tools import GetWeather, GetHoroscope, GetCompanyName, Tool
from typing import List
import asyncio
from dotenv import load_dotenv
import time
import os
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from ai_server.utils.logger import setup_logging
from ai_server.utils.tracing import spanner, async_spanner
from ai_server.constants import DATABASE, INIT

load_dotenv()

setup_logging(level="INFO")

tracer_provider = register(
    space_id = os.getenv("ARIZE_SPACE_ID"),
    api_key = os.getenv("ARIZE_API_KEY"),
    project_name = os.getenv("ARIZE_PROJECT_NAME"),
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

"""Startup initialization: group one-time embeddings and index checks in a single trace."""
# Tracer for custom spans in this module (after global provider is set)
tracer = trace.get_tracer(__name__)

with spanner(
    tracer=tracer,
    name="startup",
    kind=INIT,
    metadata={
        "redis.host": os.environ.get("REDIS_HOST"),
        "redis.port": os.environ.get("REDIS_PORT"),
    },
):
    embeddings_provider = OpenAIEmbeddingProvider()

    # Create Redis client within the async context
    redis_config = RedisClient(
        host=os.environ.get("REDIS_HOST"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        username=os.environ.get("REDIS_USERNAME"),
        password=os.environ.get("REDIS_PASSWORD"),
    )
    sync_redis_client = redis_config.get_sync_client()
    async_redis_client = redis_config.get_async_client()
        
    embedding_cache = RedisEmbeddingsCache(
        async_redis_client=async_redis_client,
        embedding_provider=embeddings_provider,
    )

    semantic_cache = ConversationMemoryCache(
        redis_client=sync_redis_client,
        embedding_cache=embedding_cache,
    )

session_id = 'a7beae66086f4116af33af72ffd6b2f8_ee8fbe625269404b95a8802f794c9a47'
user_id = 'a7beae66086f4116af33af72ffd6b2f8'

# openai_client = OpenAIResponsesAPI()
openai_client = OpenAIChatCompletionAPI()

turn_id = generate_id(8)
turn_id = f"{session_id}_{turn_id}"
system_message = Message(
    role=Role.SYSTEM,
    tool_call_id="null",
    user_id=user_id,
    session_id=session_id,
    metadata={},
    turn_id=turn_id,
    content="You are a helpful assistant.",
    function_call=None,
)

@semantic_cache.cache
async def generate_response(
        conversation_history: List[Message], 
        redis_session_manager: RedisSessionManager, 
        query: str,
        session_id: str, 
        user_id: str, 
        turn_id: str,
        tools: List[Tool] = [],
    ):
    async with async_spanner(
        tracer=tracer,
        name="GenerateResponse",
        kind=OpenInferenceSpanKindValues.CHAIN,
        session_id=session_id,
        user_id=user_id,
        turn_id=turn_id,
        input=query,
    ):
        if conversation_history[-1].role != Role.TOOL:
            semantic_conv_history = await redis_session_manager.get_relevant_messages_by_session_id(session_id, user_id, query)
            if len(semantic_conv_history) > 0:
                if conversation_history[-1].role != Role.SYSTEM:
                    conversation_history.extend(semantic_conv_history)

        # Consider Query only when previous message is not a tool call, 
        # if previous message is tool call we pass it to LLM for summarisation without user query
        # as query associated with it is already stored in conversation history
        pure_user_query = query and conversation_history[-1].role != Role.TOOL
        async with async_spanner(
            tracer=tracer,
            name="LLMGenerateResponse",
            kind=OpenInferenceSpanKindValues.CHAIN,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            input=query,
            metadata={
                "pure_user_query": bool(pure_user_query),
            },
        ):
            messages = await openai_client.generate_response(
                query=query if pure_user_query else None,
                conversation_history=conversation_history,
                user_id=user_id,
                turn_id=turn_id,
                session_id=session_id,
                tools=tools
            )
        return messages


async def fill_data():
    redis_session_manager = await RedisSessionManager.create(
        async_redis_client=async_redis_client,
        embedding_cache=embedding_cache,
    )

    conversation_history: List[Message] = [system_message]

    queries = [
        # "hi",
        # "What is the weather today at paris",
        # "what is my horoscope, am airies and what is the weather at kolkata",
        # "what is the company name",
        # "what is the company name and weather in delhi",
        # "thanks",
        "explain the working of fast api server",
        "explain the working of flask server",
        "explain how middlewares in fast api server can be configured",
        "exit"
    ]
    i = 0

    step = 1
    max_step = 3
    start_time = None

    while True:

        if step > max_step:
            raise Exception("Max step reached")
        
        if i >= len(queries):
            break
        
        # if last message was an AI message, start a new turn of conversation or if its just the starting conversation
        if conversation_history[-1].role == Role.AI or conversation_history[-1].role == Role.SYSTEM:
            print("\n")
            turn_id = generate_id(8)
            turn_id = f"{session_id}_{turn_id}"
            query = queries[i]
            start_time = time.time()
            print("Q:", query)
            if query == "exit":
                break
            
            i+=1
            step = 1
        else:
            step+=1
        
        # When LLM requests a tool call, skip semantic cache as Tool call messages are not stored in semantic cache 
        # But we want to store in cache the AI response in response to the tool call, hence we only skip check cache
        skip_semantic_cache_check_only = conversation_history[-1].role == Role.TOOL
        
        messages = await generate_response(
            conversation_history=conversation_history,
            redis_session_manager=redis_session_manager,
            query=query,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            tools=[GetWeather(), GetHoroscope(), GetCompanyName()],
            skip_semantic_cache=skip_semantic_cache_check_only,
        )
        
        await redis_session_manager.add_message(messages)

        # Only during user query we replace conv history with semantic history otherwise all messages generated by AI
        # have to be added to conv history as it is, because they need to be processed for tool calls
        conversation_history.extend(messages)
        if len(messages) >= 1:
            if messages[0].role == Role.HUMAN:
                for message in messages[1:]:
                    if message.content:
                        print(f"{message.role.value}:", message.content)
            else:
                for message in messages:
                    if message.content:
                        print(f"{message.role.value}:", message.content)



async def generate_answer(query: str, session_id: str, user_id: str, turn_id: str):
    async with async_spanner(
        tracer=tracer,
        name="GenerateAnswer",
        kind=OpenInferenceSpanKindValues.CHAIN,
        session_id=session_id,
        user_id=user_id,
        turn_id=turn_id,
        input=query,
    ):
        redis_session_manager = await RedisSessionManager.create(
            async_redis_client=async_redis_client,
            embedding_cache=embedding_cache,
        )
        
        step = 1
        max_step = 3
        conversation_history: List[Message] = [system_message]
        
        while True:
            if step > max_step:
                raise Exception("Max step reached")

            messages = await generate_response(
                conversation_history=conversation_history,
                redis_session_manager=redis_session_manager,
                query=query,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                tools=[GetWeather(), GetHoroscope(), GetCompanyName()],
            )
            
            # Add messages to long-term store under its own span
            async with async_spanner(
                tracer=tracer,
                name="RedisAddMessage",
                kind=DATABASE,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                input=messages,
                metadata={
                    "step": step,
                    "max_step": max_step,
                    "query": query,
                },
            ):
                await redis_session_manager.add_message(messages)

            # Only during user query we replace conv history with semantic history otherwise all messages generated by AI
            # have to be added to conv history as it is, because they need to be processed for tool calls
            conversation_history.extend(messages)

            if len(messages) >= 1:
                if messages[0].role == Role.HUMAN:
                    for message in messages[1:]:
                        if message.content:
                            print(f"{message.role.value}:", message.content)
                else:
                    for message in messages:
                        if message.content:
                            print(f"{message.role.value}:", message.content)
            if messages[-1].role == Role.AI:
                break
            elif messages[-1].role == Role.TOOL:
                step+=1

async def interactive_main():
    while True:
        # Read blocking input without leaving the event loop
        query = await asyncio.get_running_loop().run_in_executor(None, input, "You: ")
        if query == "exit":
            break
        turn_id = f"{session_id}_{generate_id(8)}"
        await generate_answer(query, session_id, user_id, turn_id)

if __name__ == "__main__":
    asyncio.run(interactive_main())
    # asyncio.run(fill_data())
    
# Next steps
# 4. Add tracing in embeddings and vectorizers and all
