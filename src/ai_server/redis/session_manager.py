
from ai_server.api.exceptions.redis_exceptions import RedisMessageStoreFailedException, \
    RedisRetrievalFailedException
from ai_server.api.exceptions.schema_exceptions import MessageParseException

from ai_server.schemas.message import Message, Role, FunctionCallRequest
from ai_server.schemas.context import Context

from ai_server.redis.embedding_cache import RedisEmbeddingsCache

from ai_server.utils.general import get_token_count
from ai_server.utils.singleton import SingletonMeta
from ai_server.utils.tracing import async_spanner

from ai_server.constants import DATABASE
from ai_server.config import MAX_TOKEN_THRESHOLD, MAX_TURNS_TO_FETCH

from typing import List, Dict
import json
from datetime import datetime, timezone
import asyncio
import logging
from opentelemetry import trace
import re

from redis.asyncio import Redis as AsyncRedis

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

    def _prepare_turn_index_doc(
        self,
        messages: List[Message],
        session_id: str,
        user_id: str,
        turn_id: str,
        created_at: str,
        created_at_epoch: int,
        turn_number: int,
        prev_n_turn_after_last_summary: int,
        prev_total_token_count_since_last_summary: int,
        summary: str | None,
    ) -> Dict[str, str | int | None]:
        turn_token_count = 0 
        for msg in messages:
            turn_token_count += msg.token_count
        total_token_count_since_last_summary = prev_total_token_count_since_last_summary + turn_token_count if summary is None else turn_token_count 
        n_turn_after_last_summary = prev_n_turn_after_last_summary + 1 if summary is None else 1
        return {
            "created_at_epoch": created_at_epoch,
            "created_at": created_at,
            "turn_id": turn_id,
            "user_id": user_id,
            "session_id": session_id,
            "turn_token_count": turn_token_count, # token count of the turn
            "summary_for_last_n_turns": summary, # Running summary of the last n turns from the beginning of conversation
            "summary_token_count": get_token_count(summary), # token count of the summary
            "turn_number": turn_number, # nth turn in the conversation session
            "n_turn_after_last_summary": n_turn_after_last_summary, # nth turn counted from the record where last summary was attached
            # total token count of the conversation session since last summary - should not include summary token count 
            "total_token_count_since_last_summary": total_token_count_since_last_summary, 
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
    
    async def add_message(
        self, 
        messages: List[Message], 
        new_summary: str | None, 
        turn_number: int, 
        prev_n_turn_after_last_summary: int,
        prev_total_token_count_since_last_summary: int,
    ) -> None:
        """
        Store a turn's messages in Redis with associated metadata.
        
        Args:
            messages: List of messages in this turn (user query + AI response + tool calls)
            new_summary: New summary generated for this turn, or None if no summarization occurred
            turn_number: The sequential turn number in the conversation session
            prev_n_turn_after_last_summary: Number of turns since last summary (from previous turn's context)
            prev_total_token_count_since_last_summary: Token count since last summary (from previous turn's context)
        """
        try:
            if not messages:
                return

            # KV store data for each turn and its relevant data to be stored in a set structure
            turn_id = messages[0].turn_id
            session_id = messages[0].session_id
            user_id = messages[0].user_id
            created_at = messages[-1].created_at
            created_at_epoch = self._to_epoch_ms(created_at)

            turn_kv_key = f"turn:{turn_id}"
            turn_payload = self._prepare_turn_payload(
                messages=messages,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                created_at=created_at,
                created_at_epoch=created_at_epoch,
                turn_number=turn_number,
                prev_n_turn_after_last_summary=prev_n_turn_after_last_summary,
                prev_total_token_count_since_last_summary=prev_total_token_count_since_last_summary,
                summary=new_summary,
            )
            
            # Session key to store session data in a time sorted manner
            session_key = f"user:{user_id}:sessions_by_last_activity"
            
            turn_key = f"session:{session_id}:turn:{turn_id}"

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
                input=json.dumps(turn_payload),
                metadata={
                    "turn_kv_key": turn_kv_key,
                    "session_key": session_key,
                    "turn_key": turn_key,
                    "created_at": created_at,
                    "created_at_epoch": created_at_epoch,
                }
            ):
                await asyncio.gather(
                    # Stores 'turn' along with all relevent data of that turn since redis index does not allow for 
                    # storing raw json unless you use storage_type as json at the cost of performance so maintaining a kv store makes sense
                    _span_await("kv_set_turns", self.async_redis_client.set(turn_kv_key, json.dumps(turn_payload))), 

                    # Stores session_id with created_at_epoch as score for quick retreival of active sessions per user
                    _span_await("zadd_sessions", self.async_redis_client.zadd(session_key, {session_id: created_at_epoch})),

                    # Stores turn_id with created_at_epoch as score for quick retreival of turns per session
                    _span_await("zadd_turns", self.async_redis_client.zadd(turn_key, {turn_id: created_at_epoch})),
                )

        except Exception as e:
            raise RedisMessageStoreFailedException(
                message="Failed to add message to conversation memory", 
                note=str(e)
            )


    async def get_all_session_per_user(self, user_id: str) -> List[str]:
        try:
            session_key = f"user:{user_id}:sessions_by_last_activity"
            session_ids = await self.async_redis_client.zrevrange(session_key, 0, -1)
            return session_ids
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get all sessions per user", 
                note=str(e)
            )

    async def get_all_messages_per_session(self, session_id: str) -> List[Message]:
        """Get all messages for a session, most recent first."""
        try:
            turn_key = f"session:{session_id}:turns"
            # Get all turn_ids, most recent first
            turn_ids = await self.async_redis_client.zrevrange(turn_key, 0, -1)
            
            if not turn_ids:
                return []
            
            # Fetch all turn payloads in one MGET call
            turn_kv_keys = [f"turn:{tid}" for tid in turn_ids]
            raw_payloads = await self.async_redis_client.mget(turn_kv_keys)
            
            all_messages: List[Message] = []
            for raw in raw_payloads:
                if not raw:
                    continue
                payload = json.loads(raw)
                turn_msgs = self._parse_messages(payload.get("messages", []))
                all_messages.extend(turn_msgs)
            
            return all_messages
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get all messages for session",
                note=str(e)
            )

    async def get_paginated_messages_per_session(
        self, 
        session_id: str, 
        page: int = 1, 
        page_size: int = 40
    ) -> Dict[str, any]:
        """
        Get paginated messages for a session, most recent first.
        
        Args:
            session_id: The session ID
            page: Page number (1-indexed)
            page_size: Number of turns per page
            
        Returns:
            Dict with 'messages', 'page', 'page_size', 'total_turns', 'has_more'
        """
        try:
            turn_key = f"session:{session_id}:turns"
            
            # Calculate offset (0-indexed for Redis)
            start = (page - 1) * page_size
            end = start + page_size - 1
            
            total_turns, turn_ids = await asyncio.gather(
                self.async_redis_client.zcard(turn_key),
                self.async_redis_client.zrevrange(turn_key, start, end),
            )
            
            if not turn_ids:
                return {
                    "messages": [],
                    "page": page,
                    "page_size": page_size,
                    "total_turns": total_turns,
                    "has_more": False,
                }
            
            # Fetch turn payloads in one MGET call
            turn_kv_keys = [f"turn:{tid}" for tid in turn_ids]
            raw_payloads = await self.async_redis_client.mget(turn_kv_keys)
            
            all_messages: List[Message] = []
            for raw in raw_payloads:
                if not raw:
                    continue
                payload = json.loads(raw)
                turn_msgs = self._parse_messages(payload.get("messages", []))
                all_messages.extend(turn_msgs)
            
            has_more = (start + len(turn_ids)) < total_turns
            
            return {
                "messages": all_messages,
                "page": page,
                "page_size": page_size,
                "total_turns": total_turns,
                "has_more": has_more,
            }
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get paginated messages for session",
                note=str(e)
            )

    def _build_context_result(
        self,
        summary_text: str | None,
        messages: List[Message],
        total_tokens: int,
        most_recent_turn: Dict,
    ) -> Context:
        """Build the standardized context result dictionary."""
        return Context(
            summary=summary_text,
            previous_conversation=messages,
            context_token_count=total_tokens,
            turns_after_last_summary=most_recent_turn.get("n_turn_after_last_summary", 0),
            total_token_count_since_last_summary=most_recent_turn.get("total_token_count_since_last_summary", 0),
        )
    
    def _find_summary_position(self, turns: List[Dict]) -> tuple[int | None, str | None, int]:
        """Find the most recent turn with a summary. Returns (position, text, token_count)."""
        for i, turn in enumerate(turns):
            if turn.get("summary_for_last_n_turns"):
                return i, turn["summary_for_last_n_turns"], turn.get("summary_token_count", 0)
        return None, None, 0
    
    def _select_additional_turns(self, turns: List[Dict], start_idx: int, budget: int) -> List[Dict]:
        """Select older turns that fit within the remaining token budget."""
        additional = []
        remaining = budget
        for turn in turns[start_idx:]:
            turn_tokens = turn.get("turn_token_count", 0)
            if turn_tokens > remaining:
                break
            additional.append(turn)
            remaining -= turn_tokens
        return additional

    async def get_context_for_llm(
        self,
        session_id: str,
        max_token_threshold: int = MAX_TOKEN_THRESHOLD,
        max_turns_to_fetch: int = MAX_TURNS_TO_FETCH,
    ) -> Context:
        """
        Get conversation context for LLM using summarization-aware retrieval.
        
        Args:
            session_id: The session ID to retrieve context for
            max_token_threshold: Maximum token budget for context (default from config)
            max_turns_to_fetch: Maximum number of turns to retrieve from Redis (default from config)
        
        Returns:
            Context object containing summary, previous_conversation messages,
            context_token_count, turns_after_last_summary, and total_token_count_since_last_summary
        
        Algorithm:
            1. Retrieve recent N turns (default 100)
            2. Find the most recent turn with a summary (position S)
            3. Turns from S to most recent are mandatory context
            4. If token count < threshold, add older turns until threshold reached
        """
        try:
            turn_key = f"session:{session_id}:turns"
            turn_ids = await self.async_redis_client.zrevrange(turn_key, 0, max_turns_to_fetch - 1)
            
            if not turn_ids:
                return self._build_context_result(None, [], 0, {})
            
            # Fetch and parse turn payloads (index 0 = most recent)
            turn_kv_keys = [f"turn:{tid}" for tid in turn_ids]
            raw_payloads = await self.async_redis_client.mget(turn_kv_keys)
            turns = [json.loads(raw) for raw in raw_payloads if raw]
            
            if not turns:
                return self._build_context_result(None, [], 0, {})
            
            # Find summary position and determine mandatory turns
            summary_position, summary_text, summary_token_count = self._find_summary_position(turns)
            mandatory_turns = turns[:summary_position + 1] if summary_position is not None else turns
            
            # Calculate mandatory token count
            mandatory_token_count = summary_token_count + sum(t.get("turn_token_count", 0) for t in mandatory_turns)
            most_recent_turn = turns[0]
            
            # If mandatory context exceeds threshold, return it directly
            if mandatory_token_count >= max_token_threshold:
                context_messages = self._extract_messages_from_turns(mandatory_turns[::-1])
                return self._build_context_result(
                    summary_text, context_messages, mandatory_token_count, most_recent_turn
                )
            
            # Add older turns within budget
            older_turns_start = (summary_position + 1) if summary_position is not None else 0
            remaining_budget = max_token_threshold - mandatory_token_count
            additional_turns = self._select_additional_turns(turns, older_turns_start, remaining_budget)
            
            # Combine and reverse to chronological order (oldest first)
            all_context_turns = additional_turns[::-1] + mandatory_turns[::-1]
            context_messages = self._extract_messages_from_turns(all_context_turns)
            total_tokens = mandatory_token_count + sum(t.get("turn_token_count", 0) for t in additional_turns)
            
            return self._build_context_result(
                summary_text, context_messages, total_tokens, most_recent_turn
            )
            
        except Exception as e:
            raise RedisRetrievalFailedException(
                message="Failed to get context for LLM",
                note=str(e)
            )
    
    def _extract_messages_from_turns(self, turns: List[Dict]) -> List[Message]:
        """Extract and parse messages from turn payloads in order."""
        all_messages: List[Message] = []
        for turn in turns:
            turn_msgs = self._parse_messages(turn.get("messages", []))
            all_messages.extend(turn_msgs)
        return all_messages



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