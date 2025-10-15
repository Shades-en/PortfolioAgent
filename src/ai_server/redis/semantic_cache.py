from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.schemas.message import Message, Role
from ai_server.redis.langchain_vectorizer import LangchainTextVectorizer
from ai_server.utils.singleton import SingletonMeta

from typing import List, Callable
import functools
import json
import logging

from redis import Redis

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class ConversationMemoryCache(metaclass=SingletonMeta):
    def __init__(self, redis_client: Redis, embedding_cache: RedisEmbeddingsCache) -> None:
        self.redis_client: Redis = redis_client
        self.embedding_cache: RedisEmbeddingsCache = embedding_cache
        self.conv_memory_cache = SemanticCache(
            name="agent_memory_cache",
            redis_client=self.redis_client,
            distance_threshold=0.85,
            overwrite=True,
            filterable_fields=[
                {"name": "user_id", "type": "tag"},
                {"name": "session_id", "type": "tag"},
            ],
            vectorizer=LangchainTextVectorizer(
                langchain_embeddings=self.embedding_cache.embedding_client,
                model=self.embedding_cache.model_name,
                cache=self.embedding_cache.cache,
                dimensions=self.embedding_cache.dims,
            ),
        )

    async def clear_cache(self):
        await self.conv_memory_cache.aclear()

    async def delete_cache(self):
        await self.conv_memory_cache.adelete()

    def cache(self, func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> List[Message]:
            query = kwargs.get('query')
            session_id = kwargs.get('session_id')
            turn_id = kwargs.get('turn_id')
            user_id = kwargs.get('user_id')
            conversation_history = kwargs.get('conversation_history')
            explicit_skip = kwargs.get('skip_semantic_cache', False)
            # When LLM requests a tool call, skip semantic cache as Tool call messages are not stored in semantic cache
            skip_semantic_cache_retrieval = (len(conversation_history) > 0 and conversation_history[-1].role == Role.TOOL)
            if not skip_semantic_cache_retrieval and not explicit_skip:
                session_id_filter = Tag("session_id") == session_id
                user_id_filter = Tag("user_id") == user_id
                filter_ = session_id_filter & user_id_filter
                with tracer.start_as_current_span(
                    "semantic_cache_check",
                    attributes={"app.query_len": len(query) if isinstance(query, str) else 0},
                ):
                    result = await self.conv_memory_cache.acheck(
                        prompt=query,
                        filter_expression=filter_,
                    )
                if result:
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
                        embedding=None,
                    )
                    ai_response = result[0].get("response", "")
                    
                    # Parse metadata from JSON string if available
                    metadata_json = result[0].get("metadata", "{}")
                    try:
                        ai_metadata = json.loads(metadata_json)
                    except (json.JSONDecodeError, TypeError):
                        ai_metadata = {}
                        
                    formatted_result = Message(
                        role=Role.AI,
                        tool_call_id="null",
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        metadata=ai_metadata,
                        content=ai_response,
                        function_call=None,
                        embedding=None,
                    )
                    return [formatted_query, formatted_result]
                else:
                    logger.info(f"Semantic cache miss for query: {query}")

            response: List[Message] = await func(*args, **kwargs)
            if response[-1].role == Role.AI and not explicit_skip:
                # Serialize metadata to JSON string to avoid Redis dictionary error
                metadata_json = json.dumps(response[-1].metadata) if response[-1].metadata else "{}"
                with tracer.start_as_current_span(
                    "semantic_cache_store",
                    attributes={
                        "app.response_len": len(response[-1].content) if response[-1].content else 0,
                    },
                ):
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

__all__ = ["ConversationMemoryCache"]