import json
from ai_server.api.schemas.redis import RedisConfig
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from ai_server.ai.providers.llm_provider import LLMProvider
from ai_server.api.exceptions.redis_exceptions import RedisIndexFailedException, RedisMessageStoreFailedException
from ai_server.api.schemas.message import Message
from typing import List

class RedisClient:
    def __init__(self, config: RedisConfig, embedding_provider: LLMProvider) -> None:
        self.redis: Redis = Redis(config.host, config.port, config.username, config.password)
        self.embedding_provider: LLMProvider = embedding_provider
        self._conv_memory_index: SearchIndex = self._create_conv_memory_index()
        
    def _create_conv_memory_index(self) -> SearchIndex:
        try:
            memory_schema = IndexSchema.from_dict({
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
                    {"name": "memory_id", "type": "tag"},
                    {"name": "session_id", "type": "tag"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": self.embedding_provider.embedder_dimension,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            })
            index = SearchIndex(redis_client=self.redis, schema=memory_schema, validate_on_load=True)
            index.create(overwrite=True)
            return index
        except Exception as e:
            raise RedisIndexFailedException(message="Failed to create conversation memory index", note=str(e))
    
    def add_message(self, messages: List[Message]) -> None:
        try:
            memory_data = []
            for message in messages:
                embedding = self.embedding_provider.embed_query(message.content)
                memory = {
                    "message": message.content,
                    "role": message.role.value,
                    "tool_call_id": message.tool_call_id,
                    "function_call": json.dumps(message.function_call),
                    "metadata": json.dumps(message.metadata),
                    "created_at": message.created_at,
                    "user_id": message.user_id,
                    "memory_id": message.message_id,
                    "session_id": message.session_id,
                    "embedding": embedding,
                }
                memory_data.append(memory)
            self._conv_memory_index.load(memory_data)
        except Exception as e:
            raise RedisMessageStoreFailedException(message="Failed to add message to conversation memory", note=str(e))

    def get_all_messages_by_session_id(self, session_id: str, user_id: str) -> List[Message]:
        try:
            return self._conv_memory_index.search(session_id, user_id) # Findout correct way to search
        except Exception as e:
            raise RedisMessageStoreFailedException(message="Failed to get messages from conversation memory", note=str(e)) # Do we need an exception?

    def get_relevant_messages_by_session_id(self, session_id: str, user_id: str) -> List[Message]:
        pass

    def get_user_sessions(self, user_id: str) -> List[str]:
        pass
        