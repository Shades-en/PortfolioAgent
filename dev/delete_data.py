import asyncio
from ai_server.redis.session_manager import RedisSessionManager
from ai_server.redis.client import RedisClient
from langchain_openai import OpenAIEmbeddings
from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.redis.semantic_cache import ConversationMemoryCache
from dotenv import load_dotenv

load_dotenv()

embedding_model = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=embedding_model)

config = RedisClient(host="redis-10403.c279.us-central1-1.gce.redns.redis-cloud.com", port=10403, username="default", password="nCBJDG503L1wmPJvDoEdfEujTHeQIaMb")
async_redis_client = config.get_async_client()
sync_redis_client = config.get_sync_client()
    
embedding_cache = RedisEmbeddingsCache(
    async_redis_client=async_redis_client,
    embedding_client=embeddings,
    model_name=embedding_model,
)

conversation_memory_cache = ConversationMemoryCache(
    redis_client=sync_redis_client,
    embedding_cache=embedding_cache,
)


async def delete_data():
    session_manager = await RedisSessionManager.create(async_redis_client=async_redis_client, embedding_cache=embedding_cache)
    await session_manager.delete_conv_index_data()
    await embedding_cache.clear_cache()
    await conversation_memory_cache.clear_cache()

asyncio.run(delete_data())