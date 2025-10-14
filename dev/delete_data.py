import asyncio
import os
from ai_server.redis.session_manager import RedisSessionManager
from ai_server.redis.client import RedisClient
from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.redis.semantic_cache import ConversationMemoryCache
from ai_server.ai.providers.openai_provider import OpenAIEmbeddingProvider
from dotenv import load_dotenv

load_dotenv()

session_id = 'a7beae66086f4116af33af72ffd6b2f8_ee8fbe625269404b95a8802f794c9a47'
user_id = 'a7beae66086f4116af33af72ffd6b2f8'

embeddings_provider = OpenAIEmbeddingProvider()

# Read Redis configuration from environment variables
# Expected variables: REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD
redis_host = os.getenv("REDIS_HOST")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_username = os.getenv("REDIS_USERNAME") or None
redis_password = os.getenv("REDIS_PASSWORD") or None

config = RedisClient(host=redis_host, port=redis_port, username=redis_username, password=redis_password)
async_redis_client = config.get_async_client()
sync_redis_client = config.get_sync_client()
    
embedding_cache = RedisEmbeddingsCache(
    async_redis_client=async_redis_client,
    embedding_provider=embeddings_provider,
)

conversation_memory_cache = ConversationMemoryCache(
    redis_client=sync_redis_client,
    embedding_cache=embedding_cache,
)


async def delete_data():
    session_manager = await RedisSessionManager.create(async_redis_client=async_redis_client, embedding_cache=embedding_cache)
    await asyncio.gather(
        session_manager.delete_conv_index_data(),
        session_manager.delete_user_kv(user_id=user_id),
        embedding_cache.clear_cache(),
        conversation_memory_cache.clear_cache()
    )

asyncio.run(delete_data())