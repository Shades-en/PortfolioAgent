
from ai_server.api.exceptions.redis_exceptions import RedisIndexFailedException, \
    RedisMessageStoreFailedException, \
    RedisRetrievalFailedException, \
    RedisIndexDropFailedException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from ai_server.constants import DATABASE, VECTOR_INDEX
from ai_server.schemas.message import Message
from ai_server.schemas.message import Role
from ai_server.schemas.message import FunctionCallRequest

from ai_server.redis.embedding_cache import RedisEmbeddingsCache

from ai_server.utils.singleton import SingletonMeta
from ai_server.utils.general import get_env_int
from ai_server.utils.tracing import async_spanner

from typing import List, Self, Optional
import json
from datetime import datetime, timezone
import asyncio
import logging
from opentelemetry import trace
import re

from redisvl.index import AsyncSearchIndex
from redis.asyncio import Redis as AsyncRedis
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema.schema import IndexSchema
from redisvl.redis.utils import array_to_buffer

# Logging and tracing
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Trace attribute keys
TRACE_SESSION_ID = "app.session_id"
TRACE_USER_ID = "app.user_id"
TRACE_TURN_ID = "app.turn_id"


class RedisSessionManager(metaclass=SingletonMeta):
    def __init__(self, async_redis_client: AsyncRedis, embedding_cache: RedisEmbeddingsCache) -> None:
        self.async_redis_client: AsyncRedis = async_redis_client
        self.embedding_cache: RedisEmbeddingsCache = embedding_cache
        self._conv_memory_index: Optional[AsyncSearchIndex] = None
        # Track background tasks to avoid premature GC and to allow cleanup if needed
        self._bg_tasks: set[asyncio.Task] = set()
        self._index_schema: dict = {
            "index": {
                "name": "agent_turns",
                "prefix": "turn",
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "created_at_epoch", "type": "numeric"},
                {"name": "user_id", "type": "tag"},
                {"name": "turn_id", "type": "tag"},
                {"name": "session_id", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": self.embedding_cache.dims,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                        "m": get_env_int("REDIS_HNSW_M", 24),
                        "ef_construction": get_env_int("REDIS_HNSW_EF_CONSTRUCTION", 200),
                    },
                },
            ],
        }
        
    
    @classmethod
    async def create(cls, async_redis_client: AsyncRedis, embedding_cache: RedisEmbeddingsCache) -> Self:
        try:
            redis_client = cls(async_redis_client=async_redis_client, embedding_cache=embedding_cache)
            if redis_client._conv_memory_index is None:
                await redis_client.create_conv_memory_index()
            return redis_client
        except Exception as e:
            raise RedisIndexFailedException(
                message="Failed to create conversation memory index", 
                note=str(e)
            )
        
    async def create_conv_memory_index(self) -> AsyncSearchIndex:
        try:
            if self._conv_memory_index is not None:
                return self._conv_memory_index
            memory_schema = IndexSchema.from_dict(self._index_schema)
            index = AsyncSearchIndex(redis_client=self.async_redis_client, schema=memory_schema, validate_on_load=True)
            await index.create(overwrite=False)
            self._conv_memory_index = index
            return index
        except Exception as e:
            raise RedisIndexFailedException(
                message="Failed to create conversation memory index", 
                note=str(e)
            )

    def _schedule_background_embed_index(
        self,
        messages: List[Message],
        session_id: str,
        user_id: str,
        turn_id: str,
        created_at: str,
        created_at_epoch: int,
        max_chars: int = 4000,
    ) -> None:
        """Prepare text and schedule background embedding + index load for a turn.

        Non-blocking. Keeps a reference to the created task to prevent GC.
        """
        # Build a turn-level text (HUMAN + AI only) for vector indexing
        natural_text_parts = [
            msg.content.strip()
            for msg in messages
            if msg.content and msg.content.strip() and msg.role in (Role.HUMAN, Role.AI)
        ]

        if not natural_text_parts:
            return

        turn_text_full = "\n".join(natural_text_parts)
        turn_text = self._truncate_text(turn_text_full, max_chars=max_chars)

        async def _bg_embed_and_index(skip_cache: bool = False):
            try:
                async with async_spanner(
                    tracer=tracer,
                    name="BackgroundEmbedIndex",
                    kind=VECTOR_INDEX,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    input=turn_text,
                    metadata={
                        "skip_cache": skip_cache,
                        "max_chars": max_chars,
                    }
                ):
                    embedding = (await self.embedding_cache.embed_documents([turn_text], skip_cache=skip_cache))[0]
                    turn_doc = {
                        "text": turn_text,
                        "created_at": created_at,
                        "created_at_epoch": created_at_epoch,
                        "user_id": user_id,
                        "turn_id": turn_id,
                        "session_id": session_id,
                        "embedding": array_to_buffer(embedding, dtype="float32"),
                    }
                    async with async_spanner(
                        tracer=tracer,
                        name="VectorIndexLoad",
                        kind=VECTOR_INDEX,
                        session_id=session_id,
                        user_id=user_id,
                        turn_id=turn_id,
                        input=turn_text,
                        metadata={
                            "document_count": 1,
                        },
                    ):
                        await self._conv_memory_index.load([turn_doc])
            except Exception:
                logger.exception("Background embed/index failed for turn %s", turn_id)

        # Fire-and-forget background task, keep a reference to avoid GC
        # We skip cache when storing embeddings here as we are sure that ai messages will always be different, hence no point of caching them
        task = asyncio.create_task(_bg_embed_and_index(skip_cache=True)) 
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
    
    async def add_message(self, messages: List[Message]) -> None:
        try:
            if not messages:
                return

            # Persist full turn (including TOOL messages) in KV store
            turn_id = messages[0].turn_id
            session_id = messages[0].session_id
            user_id = messages[0].user_id
            created_at = messages[-1].created_at
            created_at_epoch = self._to_epoch_ms(created_at)

            kv_key = f"turn:{turn_id}"
            kv_payload = {
                "turn_id": turn_id,
                "session_id": session_id,
                "user_id": user_id,
                "created_at": created_at,
                "created_at_epoch": created_at_epoch,
                "messages": [
                    {
                        "role": msg.role.value,
                        "tool_call_id": msg.tool_call_id,
                        "user_id": msg.user_id,
                        "session_id": msg.session_id,
                        "turn_id": msg.turn_id,
                        "metadata": msg.metadata,
                        "content": msg.content,
                        "function_call": (msg.function_call.model_dump() if msg.function_call else None),
                        "message_id": msg.message_id,
                        "created_at": msg.created_at,
                    }
                    for msg in messages
                ],
            }
            # Persist KV and update ZSET/SESSION SET in parallel (namespaced by user and session)
            zset_key = f"user:{user_id}:session:{session_id}:turns"

            async def _span_await(name: str, coro):
                # Convert provided span name to PascalCase for consistency
                span_name = ''.join(part.capitalize() for part in re.split(r'[^0-9a-zA-Z]+', name) if part)
                async with async_spanner(
                    tracer=tracer,
                    name=span_name,
                    kind=DATABASE,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=turn_id,
                ):
                    return await coro

            async with async_spanner(
                tracer=tracer,
                name="AddMessage",
                kind=DATABASE,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                input=json.dumps(kv_payload),
                metadata={
                    "kv_key": kv_key,
                    "zset_key": zset_key,
                    "created_at": created_at,
                    "created_at_epoch": created_at_epoch,
                }
            ):
                # Write KV/ZSET/SADD now for immediate UI visibility
                await asyncio.gather(
                    _span_await("kv_set", self.async_redis_client.set(kv_key, json.dumps(kv_payload))), # Stores turn along with all relevent data of that turn
                    _span_await("zadd", self.async_redis_client.zadd(zset_key, {turn_id: created_at_epoch})), # Stores turn_id with created_at_epoch as score
                    _span_await("sadd", self.async_redis_client.sadd(f"user:{user_id}:sessions", session_id)), # Stores session_id in user's session set
                )

                # Schedule background embed + index load for this turn
                self._schedule_background_embed_index(
                    messages=messages,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    created_at=created_at,
                    created_at_epoch=created_at_epoch,
                )

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
            # Get all turns for this session ordered by created_at_epoch (numeric)
            filter_query = FilterQuery(
                filter_expression=filter_,
                num_results=1000,
                return_fields=["turn_id", "created_at_epoch"],
            ).sort_by("created_at_epoch", asc=True)
            results = await self._conv_memory_index.query(filter_query)

            # Fetch full turn messages from KV using MGET and concatenate
            all_messages: List[Message] = []
            turn_ids = [doc.get("turn_id") for doc in results if doc.get("turn_id")]
            if turn_ids:
                kv_keys = [f"turn:{tid}" for tid in turn_ids]
                raws = await self.async_redis_client.mget(kv_keys)
                for raw in raws:
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    turn_msgs = self._parse_messages(payload.get("messages", []))
                    all_messages.extend(turn_msgs)
            return all_messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get messages from conversation memory", 
                note=str(e)
            )

    async def get_session_messages_page(
        self,
        session_id: str,
        user_id: str,
        num_turns: int = 20,
        start_ts: float | int | None = None,
        end_ts: float | int | None = None,
        offset_num_turns: int | None = None,
        newest_first: bool = True,
    ) -> List[Message]:
        """Paginate a session's messages using the session ZSET timeline.

        Modes:
        - Time window: provide start_ts and/or end_ts (epoch seconds or ms) to page by timestamp.
          Uses Z[R]RANGEBYSCORE with LIMIT 0 num_turns.
        - Offset mode: provide offset_num_turns (int) for index-based paging. Uses Z[R]RANGE.

        Returns flattened Message list for the selected turns.
        """
        try:
            # ZSET timeline key includes user_id for namespacing
            zset_key = f"user:{user_id}:session:{session_id}:turns"
            
            # Decide retrieval mode
            turn_ids: List[str] = []
            if start_ts is not None or end_ts is not None:
                # Normalize epoch to ms
                min_score = -float("inf") if start_ts is None else self._normalize_epoch_input(start_ts)
                max_score = float("inf") if end_ts is None else self._normalize_epoch_input(end_ts)
                if newest_first:
                    raw_ids = await self.async_redis_client.zrevrangebyscore(
                        zset_key, max_score, min_score, start=0, num=num_turns
                    )
                else:
                    raw_ids = await self.async_redis_client.zrangebyscore(
                        zset_key, min_score, max_score, start=0, num=num_turns
                    )
                turn_ids = [tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else tid for tid in raw_ids]
            else:
                # Offset mode (index-based). Default newest-first.
                start_idx = offset_num_turns or 0
                end_idx = start_idx + num_turns - 1
                if newest_first:
                    raw_ids = await self.async_redis_client.zrevrange(zset_key, start_idx, end_idx)
                else:
                    raw_ids = await self.async_redis_client.zrange(zset_key, start_idx, end_idx)
                turn_ids = [tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else tid for tid in raw_ids]

            # Fetch turns from KV (batch via MGET for speed)
            messages: List[Message] = []
            if turn_ids:
                kv_keys = [f"turn:{tid}" for tid in turn_ids]
                raws = await self.async_redis_client.mget(kv_keys)
                for raw in raws:
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    msgs = self._parse_messages(payload.get("messages", []))
                    messages.extend(msgs)

            return messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get paged messages from session timeline",
                note=str(e),
            )

    async def get_relevant_messages_by_session_id(
        self, 
        session_id: str, 
        user_id: str, 
        query: str, 
        top_n_turns: int = 3, 
        vector_top_k: int = 50
    ) -> List[Message]:
        try:
            async with async_spanner(
                tracer=tracer,
                name="GetRelevantMessagesBySessionId",
                kind=DATABASE,
                session_id=session_id,
                user_id=user_id,
                input=query,
                metadata={
                    "top_n_turns": top_n_turns,
                    "vector_top_k": vector_top_k,
                }
            ):
                session_id_filter = Tag("session_id") == session_id
                user_id_filter = Tag("user_id") == user_id
                filter_ = session_id_filter & user_id_filter

                query_embedding = await self.embedding_cache.embed_query(query)
                ef_runtime = get_env_int("REDIS_HNSW_EF_RUNTIME")

                async with async_spanner(
                    tracer=tracer,
                    name="VectorSearch",
                    kind=VECTOR_INDEX,
                    session_id=session_id,
                    user_id=user_id,
                    input=query,
                    metadata={
                        "ef_runtime": ef_runtime,
                        "num_results": max(vector_top_k, top_n_turns),
                        "filter": str(filter_),
                    },
                ):
                    vector_query = VectorQuery(
                        vector=query_embedding,
                        filter_expression=filter_,
                        return_fields=["turn_id"],
                        vector_field_name="embedding",
                        num_results=max(vector_top_k, top_n_turns),
                        ef_runtime=ef_runtime,
                    ).sort_by("vector_distance", asc=True)
                    results = await self._conv_memory_index.query(vector_query)
                
                # Collect top n distinct turn_ids
                seen_turn_ids = set()
                top_n_turn_ids = []
                for result_dict in results:
                    turn_id = result_dict.get('turn_id')
                    if turn_id and turn_id not in seen_turn_ids:
                        seen_turn_ids.add(turn_id)
                        top_n_turn_ids.append(turn_id)
                        if len(top_n_turn_ids) == top_n_turns:
                            break

                if not top_n_turn_ids:
                    return []

                # Fetch full turn messages from KV for the selected turns (MGET)
                relevant_messages: List[Message] = []
                kv_keys = [f"turn:{tid}" for tid in top_n_turn_ids]
                async with async_spanner(
                    tracer=tracer,
                    name="KvMgetFetch",
                    kind=DATABASE,
                    session_id=session_id,
                    user_id=user_id,
                    input=query,
                    metadata={
                        "turn_count": len(top_n_turn_ids),
                        "kv_keys": kv_keys,
                    },
                ):
                    raws = await self.async_redis_client.mget(kv_keys)
                    for raw in raws:
                        if not raw:
                            continue
                        payload = json.loads(raw)
                        msgs = self._parse_messages(payload.get("messages", []))
                        relevant_messages.extend(msgs)

                return relevant_messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get relevant messages from conversation memory", 
                note=str(e)
            )

    def _parse_messages(self, messages_dict: List[dict]) -> List[Message]:
        try:
            messages = []
            for message in messages_dict:
                function_field = message.get("function_call")
                function_call = None
                if function_field:
                    try:
                        # function_call might be a JSON string or already a dict
                        function = json.loads(function_field) if isinstance(function_field, str) else function_field
                        if function.get("name"):
                            function_call = FunctionCallRequest(
                                name=function.get("name", None),
                                arguments=function.get("arguments", None),
                            )
                    except Exception:
                        function_call = None
                messages.append(Message(
                    role=Role(message["role"]),
                    tool_call_id=message.get("tool_call_id", "null"),
                    user_id=message["user_id"],
                    session_id=message["session_id"],
                    turn_id=message["turn_id"],
                    metadata=message.get("metadata", {}),
                    content=message.get("content"),
                    function_call=function_call,
                ))
            return messages
        except Exception as e:
            raise MessageParseException(
                message="Failed to parse message from conversation memory", 
                note=str(e)
            )

    async def _delete_session_kv(self, session_id: str, user_id: str, batch_size: int = 1000) -> int:
        """Delete all KV turn payloads and the session timeline ZSET for a given session/user.

        Uses UNLINK in batches to avoid blocking Redis. Returns the number of KV keys unlinked.
        """
        try:
            zset_key = f"user:{user_id}:session:{session_id}:turns"

            # Get all turn_ids for this session
            raw_ids = await self.async_redis_client.zrange(zset_key, 0, -1)
            turn_ids: List[str] = [tid.decode("utf-8") if isinstance(tid, (bytes, bytearray)) else str(tid) for tid in raw_ids]

            # Batch UNLINK KV keys
            deleted_total = 0
            if turn_ids:
                kv_keys = [f"turn:{tid}" for tid in turn_ids]
                for i in range(0, len(kv_keys), batch_size):
                    chunk = kv_keys[i:i+batch_size]
                    try:
                        deleted_total += int(await self.async_redis_client.unlink(*chunk))
                    except Exception:
                        # Fallback to DEL if UNLINK not supported
                        deleted_total += int(await self.async_redis_client.delete(*chunk))

            # Remove timeline ZSET and session entry in user's set
            await self.async_redis_client.delete(zset_key)
            await self.async_redis_client.srem(f"user:{user_id}:sessions", session_id)

            return deleted_total
        except Exception as e:
            raise RedisIndexDropFailedException(
                message="Failed to delete session KV data",
                note=str(e),
            )

    async def delete_user_kv(self, user_id: str, batch_size: int = 1000) -> int:
        """Delete all KV turn payloads and timelines for all sessions of a user.

        Returns total number of KV keys unlinked.
        """
        try:
            sessions = await self.async_redis_client.smembers(f"user:{user_id}:sessions")
            session_ids = [sid.decode("utf-8") if isinstance(sid, (bytes, bytearray)) else str(sid) for sid in sessions]
            total_deleted = 0
            for session_id in session_ids:
                total_deleted += await self._delete_session_kv(session_id=session_id, user_id=user_id, batch_size=batch_size)
            # Finally, remove the sessions set
            await self.async_redis_client.delete(f"user:{user_id}:sessions")
            return total_deleted
        except Exception as e:
            raise RedisIndexDropFailedException(
                message="Failed to delete user KV data",
                note=str(e),
            )

    async def delete_conv_index_data(self):
        try:
            await self._conv_memory_index.delete(drop=True)
        except Exception as e:
            raise RedisIndexDropFailedException(
                message="Failed to drop conversation memory index", 
                note=str(e)
            )

    def _truncate_text(self, text: str, max_chars: int = 4000) -> str:
        """Truncate text to a maximum number of characters, preserving head and tail for context."""
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    def _to_epoch_ms(self, ts: str) -> int:
        """Convert ISO-like timestamp string to epoch milliseconds.
        Falls back to current time on parse failure.
        """
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return int(datetime.now(timezone.utc).timestamp() * 1000)

    def _normalize_epoch_input(self, value: float | int) -> int:
        """Normalize epoch seconds or milliseconds to milliseconds as integer."""
        try:
            v = float(value)
            # Heuristic: if seconds (< 1e12), convert to ms
            if v < 1e12:
                v *= 1000.0
            return int(v)
        except Exception:
            return 0