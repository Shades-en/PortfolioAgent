from ai_server.redis.embedding_cache import RedisEmbeddingsCache
from ai_server.schemas.message import Message, Role
from ai_server.redis.langchain_vectorizer import LangchainTextVectorizer

from typing import List, Callable
import functools

from redis import Redis

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag

class ConversationMemoryCache:
    def __init__(self, redis_client: Redis, embedding_cache: RedisEmbeddingsCache) -> None:
        self.redis_client: Redis = redis_client
        self.embedding_cache: RedisEmbeddingsCache = embedding_cache
        self.conv_memory_cache = SemanticCache(
            name="agent_memory_cache",
            redis_client=self.redis_client,
            distance_threshold=0.1,
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

    def clear_cache(self):
        self.conv_memory_cache.clear()

    def cache(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> List[Message]:
            query = kwargs.get('query')
            session_id = kwargs.get('session_id')
            turn_id = kwargs.get('turn_id')
            user_id = kwargs.get('user_id')
            vector_query = self.embedding_cache.embed_query(query)
            session_id_filter = Tag("session_id") == session_id
            user_id_filter = Tag("user_id") == user_id
            filter_ = session_id_filter & user_id_filter
            if result := self.conv_memory_cache.check(
                prompt=query,
                filter_expression=filter_,
            ):
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
                ai_metadata = result[0].get("metadata", {})
                formatted_result = Message(
                    role=Role.AI,
                    tool_call_id="null",
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    metadata=ai_metadata,
                    content=ai_response,
                    function_call=None,
                    embedding=self.embedding_cache.embed_query(ai_response),
                )
                return [formatted_query, formatted_result]
            else:
                response: List[Message] = func(*args, **kwargs)
                if response[-1].role == Role.AI:
                    self.conv_memory_cache.store(
                        prompt=query,
                        response=response[-1].content,
                        filters={
                            "metadata": response[-1].metadata,
                            "user_id": user_id,
                            "session_id": session_id,
                        }
                    )
                return response
        return wrapper

    