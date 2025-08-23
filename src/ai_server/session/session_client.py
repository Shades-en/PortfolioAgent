
from ai_server.api.exceptions.redis_exceptions import RedisIndexFailedException, \
    RedisMessageStoreFailedException, \
    RedisRetrievalFailedException, \
    RedisIndexDropFailedException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from ai_server.schemas.message import Message
from ai_server.schemas.message import Role
from ai_server.schemas.message import FunctionCallRequest
from ai_server.schemas.redis import RedisConfig

from typing import List, Callable
import json
import functools

from langchain_core.embeddings import Embeddings

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema.schema import IndexSchema
from redisvl.extensions.cache.llm import SemanticCache


class RedisClient:
    def __init__(self, config: RedisConfig, embedding_provider: Embeddings) -> None:
        self.redis: Redis = Redis(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
        )
        self.embedding_provider: Embeddings = embedding_provider
        self._index_schema: dict = {
                "index": {
                    "name": "agent_memories",  # Index name for identification
                    "prefix": "memory",       # Redis key prefix (memory:1, memory:2, etc.)
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
                            "dims": 1536,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        self._conv_memory_index: SearchIndex = self.create_conv_memory_index()
        self.conv_memory_cache = SemanticCache(
            name="agent_memory_cache",
            redis_client=self.redis,
            distance_threshold=0.1,
            # TODO: Remove unneccessary filters and add embedding filter and figure out how retrieve it. If it doesnt allow you to retreive it then store it in custom field as well
            filterable_fields=[field["name"] for field in self._index_schema["fields"] if field["type"] != "vector"],
        )
        
    def create_conv_memory_index(self) -> SearchIndex:
        try:
            memory_schema = IndexSchema.from_dict(self._index_schema)
            index = SearchIndex(redis_client=self.redis, schema=memory_schema, validate_on_load=True)
            index.create(overwrite=True)
            return index
        except Exception as e:
            raise RedisIndexFailedException(
                message="Failed to create conversation memory index", 
                note=str(e)
            )
    
    def add_message(self, messages: List[Message]) -> None:
        try:
            memory_data = []
            for message in messages:
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
                    "embedding": message.embedding,
                }
                memory_data.append(memory)
            self._conv_memory_index.load(memory_data)
        except Exception as e:
            raise RedisMessageStoreFailedException(
                message="Failed to add message to conversation memory", 
                note=str(e)
            )

    def get_all_messages_by_session_id(self, session_id: str, user_id: str) -> List[Message]:
        try:
            session_id_filter = Tag("session_id") == session_id
            user_id_filter = Tag("user_id") == user_id
            filter_ = session_id_filter & user_id_filter
            filter_query = FilterQuery(
                filter_expression=filter_,
                num_results=1000,
                return_fields=[field["name"] for field in self._index_schema["fields"] if field["type"] != "vector"],
            ).sort_by("created_at", asc=True)
            results = self._conv_memory_index.query(filter_query)
            messages = self._parse_messages(results)
            return messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get messages from conversation memory", 
                note=str(e)
            )

    def get_relevant_messages_by_session_id(self, session_id: str, user_id: str, query: str, top_n_turns: int = 3) -> List[Message]:
        try:
            session_id_filter = Tag("session_id") == session_id
            user_id_filter = Tag("user_id") == user_id
            tool_call_id_filter = Tag("tool_call_id") == "null"
            filter_ = session_id_filter & user_id_filter & tool_call_id_filter
            query_embedding = self.embedding_provider.embed_query(query)
            vector_query = VectorQuery(
                vector=query_embedding,
                filter_expression=filter_,
                return_fields=[field["name"] for field in self._index_schema["fields"] if field["type"] != "vector"],
                vector_field_name="embedding",
                num_results=1000
            ).sort_by("vector_distance", asc=True)
            results = self._conv_memory_index.query(vector_query)
            
            # Get top 5 distinct turn_ids with their complete dictionaries
            seen_turn_ids = set()
            top_n_turn_ids = []
            top_n_dicts = []
            
            # First pass: collect top 5 distinct turn_ids and their first occurrence
            for result_dict in results:
                turn_id = result_dict.get('turn_id')
                if turn_id and turn_id not in seen_turn_ids:
                    seen_turn_ids.add(turn_id)
                    top_n_turn_ids.append(turn_id)
                    top_n_dicts.append(result_dict)
                    if len(top_n_turn_ids) == top_n_turns:
                        break
            
            # Second pass: collect ALL dictionaries that have the same turn_ids as top 5
            final_results = []
            for result_dict in results:
                turn_id = result_dict.get('turn_id')
                if turn_id in top_n_turn_ids:
                    final_results.append(result_dict)
            
            # Sort by created_at in ascending order
            final_results.sort(key=lambda x: x.get('created_at', ''))
            
            messages = self._parse_messages(final_results)
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

    def delete_conv_index_data(self):
        try:
            self.conv_memory_index.delete(drop=True)
        except Exception as e:
            raise RedisIndexDropFailedException(
                message="Failed to drop conversation memory index", 
                note=str(e)
            )
        
    def cache(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            query = kwargs.get('query')
            session_id = kwargs.get('session_id')
            turn_id = kwargs.get('turn_id')
            user_id = kwargs.get('user_id')
            vector_query = self.embedding_provider.embed_query(query)
            session_id_filter = Tag("session_id") == session_id
            user_id_filter = Tag("user_id") == user_id
            filter_ = session_id_filter & user_id_filter
            if result := self.conv_memory_cache.check(
                prompt=query,
                vector=vector_query,
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
                formatted_result = Message(
                    role=Role.AI,
                    tool_call_id="null",
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    metadata={},
                    content=result[0].get("response", ""),
                    function_call=None,
                )
                return [formatted_query, formatted_result]
            else:
                result: List[Message] = func(*args, **kwargs)
                for message in result:
                    message.embedding = self.embedding_provider.embed_query(message.content)
                if result[-1].role == Role.AI:
                    self.conv_memory_cache.store(
                        prompt=query,
                        vector=result[-1].embedding,
                        response=result[-1].content,
                        filters={
                            "metadata": result[-1].metadata,
                            "user_id": user_id,
                            "turn_id": turn_id,
                            "session_id": session_id,
                        }
                    )
                return result
        return wrapper

    