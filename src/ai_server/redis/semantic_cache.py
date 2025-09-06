from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.schemas.message import Message, Role
from ai_server.redis.langchain_vectorizer import LangchainTextVectorizer

from typing import List, Callable
import functools
import json
import logging

from redis import Redis

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag

logger = logging.getLogger(__name__)

class ConversationMemoryCache:
    def __init__(self, redis_client: Redis, embedding_cache: RedisEmbeddingsCache) -> None:
        self.redis_client: Redis = redis_client
        self.embedding_cache: RedisEmbeddingsCache = embedding_cache
        self.conv_memory_cache = SemanticCache(
            name="agent_memory_cache",
            redis_client=self.redis_client,
            distance_threshold=0.85,
            filterable_fields=[
                {"name": "user_id", "type": "tag"},
                {"name": "session_id", "type": "tag"},
            ],
            vectorizer=LangchainTextVectorizer(
                langchain_embeddings=self.embedding_cache.embedding_client,
                model=self.embedding_cache.model_name,
                cache=self.embedding_cache.cache,
            ),
        )

    async def clear_cache(self):
        await self.conv_memory_cache.aclear()

    def cache(self, func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> List[Message]:
            query = kwargs.get('query')
            session_id = kwargs.get('session_id')
            turn_id = kwargs.get('turn_id')
            user_id = kwargs.get('user_id')
            skip_semantic_cache = kwargs.get('skip_semantic_cache', False)
            if not skip_semantic_cache:
                vector_query = await self.embedding_cache.embed_query(query)
                session_id_filter = Tag("session_id") == session_id
                user_id_filter = Tag("user_id") == user_id
                filter_ = session_id_filter & user_id_filter
                if result := await self.conv_memory_cache.acheck(
                    prompt=query,
                    filter_expression=filter_,
                ):
                    logger.info(f"Semantic cache hit for query: {query}")
                    formatted_query = Message(
                        role=Role.HUMAN,
                        tool_call_id="null",
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata={},
                        content=query,
                        function_call=None,
                        embedding=vector_query,
                    )
                    ai_response = result[0].get("response", "")
                    
                    # Parse metadata from JSON string if available
                    metadata_json = result[0].get("metadata", "{}")
                    try:
                        ai_metadata = json.loads(metadata_json)
                    except (json.JSONDecodeError, TypeError):
                        ai_metadata = {}
                        
                    ai_embedding = await self.embedding_cache.embed_query(ai_response)  
                    formatted_result = Message(
                        role=Role.AI,
                        tool_call_id="null",
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata=ai_metadata,
                        content=ai_response,
                        function_call=None,
                        embedding=ai_embedding,
                    )
                    return [formatted_query, formatted_result]
                else:
                    logger.info(f"Semantic cache miss for query: {query}")
            response: List[Message] = await func(*args, **kwargs)
            if response[-1].role == Role.AI:
                # Serialize metadata to JSON string to avoid Redis dictionary error
                metadata_json = json.dumps(response[-1].metadata) if response[-1].metadata else "{}"
                
                await self.conv_memory_cache.astore(
                    prompt=query,
                    response=response[-1].content,
                    filters={
                        "metadata": metadata_json,  # Store metadata as JSON string
                        "user_id": user_id,
                        "session_id": session_id,
                    }
                )
            return response
        return wrapper

    