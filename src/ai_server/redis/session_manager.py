
from ai_server.api.exceptions.redis_exceptions import RedisIndexFailedException, \
    RedisMessageStoreFailedException, \
    RedisRetrievalFailedException, \
    RedisIndexDropFailedException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from ai_server.schemas.message import Message
from ai_server.schemas.message import Role
from ai_server.schemas.message import FunctionCallRequest

from ai_server.redis.embedding_cache import RedisEmbeddingsCache

from typing import List, Self, Optional
import json

from redisvl.index import AsyncSearchIndex
from redis.asyncio import Redis as AsyncRedis
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema.schema import IndexSchema
from redisvl.redis.utils import array_to_buffer


class RedisSessionManager:
    def __init__(self, async_redis_client: AsyncRedis, embedding_cache: RedisEmbeddingsCache) -> None:
        self.async_redis_client: AsyncRedis = async_redis_client
        self.embedding_cache: RedisEmbeddingsCache = embedding_cache
        self._conv_memory_index: Optional[AsyncSearchIndex] = None
        self._index_schema: dict = {
                "index": {
                    "name": "agent_memories",
                    "prefix": "memory",
                    "key_separator": ":",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "message", "type": "text"},
                    {"name": "role", "type": "tag"},
                    {"name": "tool_call_id", "type": "tag"},
                    {"name": "function_call", "type": "text"},
                    {"name": "metadata", "type": "text"},
                    {"name": "created_at", "type": "text"},
                    {"name": "user_id", "type": "tag"},
                    {"name": "turn_id", "type": "tag"},
                    {"name": "memory_id", "type": "tag"},
                    {"name": "session_id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": self.embedding_cache.dims,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        
    
    @classmethod
    async def create(cls, async_redis_client: AsyncRedis, embedding_cache: RedisEmbeddingsCache) -> Self:
        try:
            redis_client = cls(async_redis_client=async_redis_client, embedding_cache=embedding_cache)
            await redis_client.create_conv_memory_index()
            return redis_client
        except Exception as e:
            raise RedisIndexFailedException(
                message="Failed to create conversation memory index", 
                note=str(e)
            )

        
    async def create_conv_memory_index(self) -> AsyncSearchIndex:
        try:
            memory_schema = IndexSchema.from_dict(self._index_schema)
            index = AsyncSearchIndex(redis_client=self.async_redis_client, schema=memory_schema, validate_on_load=True)
            await index.create(overwrite=True)
            self._conv_memory_index = index
            return index
        except Exception as e:
            raise RedisIndexFailedException(
                message="Failed to create conversation memory index", 
                note=str(e)
            )
    
    async def add_message(self, messages: List[Message]) -> None:
        try:
            memory_data = []

            embeddings = await self.embedding_cache.embed_documents([message.content for message in messages])

            for message, embedding in zip(messages, embeddings):
                memory = {
                    "message": message.content,
                    "role": message.role.value,
                    "tool_call_id": message.tool_call_id if message.tool_call_id else "null",
                    "function_call": message.function_call.model_dump_json() 
                        if message.function_call else None,
                    "metadata": json.dumps(message.metadata),
                    "created_at": message.created_at,
                    "user_id": message.user_id,
                    "turn_id": message.turn_id,
                    "memory_id": message.message_id,
                    "session_id": message.session_id,
                    "embedding": array_to_buffer(embedding, dtype="float32"),
                }
                memory_data.append(memory)
            
            await self._conv_memory_index.load(memory_data)

        except Exception as e:
            raise RedisMessageStoreFailedException(
                message="Failed to add message to conversation memory", 
                note=str(e)
            )

    async def get_all_messages_by_session_id(self, session_id: str, user_id: str) -> List[Message]:
        try:
            session_id_filter = Tag("session_id") == session_id
            user_id_filter = Tag("user_id") == user_id
            filter_ = session_id_filter & user_id_filter
            filter_query = FilterQuery(
                filter_expression=filter_,
                num_results=1000,
                return_fields=[field["name"] for field in self._index_schema["fields"] if field["type"] != "vector"],
            ).sort_by("created_at", asc=True)
            results = await self._conv_memory_index.query(filter_query)
            messages = self._parse_messages(results)
            return messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get messages from conversation memory", 
                note=str(e)
            )

    async def get_relevant_messages_by_session_id(self, session_id: str, user_id: str, query: str, top_n_turns: int = 3) -> List[Message]:
        try:
            session_id_filter = Tag("session_id") == session_id
            user_id_filter = Tag("user_id") == user_id
            tool_call_id_filter = Tag("tool_call_id") == "null"
            filter_ = session_id_filter & user_id_filter & tool_call_id_filter

            query_embedding = await self.embedding_cache.embed_query(query)
            
            vector_query = VectorQuery(
                vector=query_embedding,
                filter_expression=filter_,
                return_fields=[field["name"] for field in self._index_schema["fields"] if field["type"] != "vector"],
                vector_field_name="embedding",
                num_results=1000
            ).sort_by("vector_distance", asc=True)
            
            results = await self._conv_memory_index.query(vector_query)
            
            # Get top n distinct turn_ids with their complete dictionaries
            seen_turn_ids = set()
            top_n_turn_ids = []
            top_n_dicts = []
            
            # First pass: collect top n distinct turn_ids and their first occurrence
            for result_dict in results:
                turn_id = result_dict.get('turn_id')
                if turn_id and turn_id not in seen_turn_ids:
                    seen_turn_ids.add(turn_id)
                    top_n_turn_ids.append(turn_id)
                    top_n_dicts.append(result_dict)
                    if len(top_n_turn_ids) == top_n_turns:
                        break
            
            # Second pass: collect ALL dictionaries that have the same turn_ids as top n
            final_results = []
            for result_dict in results:
                turn_id = result_dict.get('turn_id')
                if turn_id in top_n_turn_ids:
                    final_results.append(result_dict)
            
            # Sort by created_at in ascending order
            final_results.sort(key=lambda x: x.get('created_at', ''))
            
            messages = self._parse_messages(final_results)

            # All the semantically relevent messages in top n turns are returned
            return messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get relevant messages from conversation memory", 
                note=str(e)
            )

    def _parse_messages(self, messages_dict: List[dict]) -> List[Message]:
        try:
            messages = []
            for message in messages_dict:
                if message.get("function_call"):
                    function = json.loads(message.get("function_call", {}))
                    if function.get("name"):
                        function_call = FunctionCallRequest(
                            name=function.get("name", None),
                            arguments=function.get("arguments", None),
                        )
                else:
                    function_call = None
                messages.append(Message(
                    role=Role(message["role"]),
                    tool_call_id=message.get("tool_call_id", "null"),
                    user_id=message["user_id"],
                    session_id=message["session_id"],
                    turn_id=message["turn_id"],
                    metadata=json.loads(message["metadata"]),
                    content=message["message"],
                    function_call=function_call,
                ))
            return messages
        except Exception as e:
            raise MessageParseException(
                message="Failed to parse message from conversation memory", 
                note=str(e)
            )

    async def delete_conv_index_data(self):
        try:
            await self._conv_memory_index.delete(drop=True)
        except Exception as e:
            raise RedisIndexDropFailedException(
                message="Failed to drop conversation memory index", 
                note=str(e)
            )