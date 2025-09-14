from ai_server.redis.client import RedisClient
from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.redis.session_manager import RedisSessionManager
from ai_server.redis.semantic_cache import ConversationMemoryCache
from langchain_openai import OpenAIEmbeddings
from ai_server.ai.providers.openai_provider import OpenAIChatCompletionAPI, OpenAIResponsesAPI
from ai_server.schemas.message import Message, Role
from ai_server.utils.general import generate_id
from ai_server.ai.tools.tools import GetWeather, GetHoroscope, GetCompanyName
from typing import List
import asyncio
from dotenv import load_dotenv
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

embedding_model = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=embedding_model)

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
    embedding_client=embeddings,
    model_name=embedding_model,
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
        **kwargs
    ):
    if conversation_history[-1].role != Role.TOOL:
        start_time_semantic = time.time()
        semantic_conv_history = await redis_session_manager.get_relevant_messages_by_session_id(session_id, user_id, query)
        end_time_semantic = time.time()
        logger.info(f"Timer check Semantic: Turn {turn_id} took {end_time_semantic - start_time_semantic} seconds")
        if len(semantic_conv_history) > 0:
            conversation_history.extend(semantic_conv_history)
    start = time.time()
    # Consider Query only when previous message is not a tool call, 
    # if previous message is tool call we pass it to LLM for summarisation without user query
    # as query associated with it is already stored in conversation history
    pure_user_query = query and conversation_history[-1].role != Role.TOOL
    messages = await openai_client.generate_response(
        query=query if pure_user_query else None,
        conversation_history=conversation_history,
        user_id=user_id,
        turn_id=turn_id,
        session_id=session_id,
        tools=[GetWeather(), GetHoroscope(), GetCompanyName()]
    )
    end = time.time()
    logger.info(f"Timer check LLM: Turn {turn_id} took {end - start} seconds")
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
        
        start_time_openai = time.time()
        messages = await generate_response(
            conversation_history=conversation_history,
            redis_session_manager=redis_session_manager,
            query=query,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            skip_semantic_cache=skip_semantic_cache_check_only,
        )
        end_time_openai = time.time()
        logger.info(f"Timer check Cache check of Query with LLM: Turn {turn_id} took {end_time_openai - start_time_openai} seconds")
        
        start_time_redis_add = time.time()

        await redis_session_manager.add_message(messages)

        end_time_redis_add = time.time()
        logger.info(f"Timer check Redis add: Turn {turn_id} took {end_time_redis_add - start_time_redis_add} seconds")

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

        end_time = time.time()
        logger.info(f"Timer check: Turn {turn_id} took {end_time - start_time} seconds")



async def generate_answer(query: str, session_id: str, user_id: str) -> str:

    redis_session_manager = await RedisSessionManager.create(
        async_redis_client=async_redis_client,
        embedding_cache=embedding_cache,
    )
    
    step = 1
    max_step = 3
    conversation_history: List[Message] = [system_message]
    turn_id = f"{session_id}_{generate_id(8)}"
    
    while True:
        if step > max_step:
            raise Exception("Max step reached")

        # When LLM requests a tool call, skip semantic cache as Tool call messages are not stored in semantic cache
        skip_semantic_cache = conversation_history[-1].role == Role.TOOL

        start_time_openai = time.time()
        messages = await generate_response(
            conversation_history=conversation_history,
            redis_session_manager=redis_session_manager,
            query=query,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            skip_semantic_cache=skip_semantic_cache
        )
        end_time_openai = time.time()
        logger.info(f"Timer check Cache check of Query with LLM: Turn {turn_id} took {end_time_openai - start_time_openai} seconds")
        
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

if __name__ == "__main__":
    query = "who are you"
    asyncio.run(generate_answer(query, session_id, user_id))
    # asyncio.run(fill_data())
    
# Next steps
# 1. Move the code to somewhere better - the decorator: consider using Singleton pattern
# 2. Performance analysis in notebook with multiple queries and redis cloud
# 3. Threshold Optimization - semantic cache
# 4. Telemetry to see timing 
# 5. Remove logic of cache skip from here and put it as part of tool call at semantic cache