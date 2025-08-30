import asyncio
from ai_server.redis.client import RedisClient
from ai_server.schemas.redis import RedisConfig
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding_model = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=embedding_model)

config = RedisConfig(host="redis-10403.c279.us-central1-1.gce.redns.redis-cloud.com", port=10403, username="default", password="nCBJDG503L1wmPJvDoEdfEujTHeQIaMb")

async def delete_data():
    redis_client = await RedisClient.create(config=config, embedding_provider=embeddings, model_name=embedding_model)
    await redis_client.delete_conv_index_data()
    redis_client.embedding_cache.clear_cache()

asyncio.run(delete_data())