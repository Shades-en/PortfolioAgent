from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from ai_server.redis.client import RedisClient
from ai_server.redis.session_manager import RedisSessionManager
from ai_server.ai.providers.openai_provider import OpenAIChatCompletionAPI, OpenAIResponsesAPI
from ai_server.types.context import Context
from ai_server.types.message import Message, Role
from ai_server.utils.general import generate_id
from ai_server.ai.tools.tools import GetWeather, GetHoroscope, GetCompanyName
import asyncio
from dotenv import load_dotenv
import os
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from ai_server.utils.logger import setup_logging
from ai_server.utils.tracing import spanner, async_spanner
from ai_server.constants import DATABASE, INIT
from ai_server.db import MongoDB
from ai_server.db.test import get_or_create_test_user
from ai_server.schemas.user import User, Session
from ai_server.types.state import State

load_dotenv()

setup_logging(level="INFO")

tracer_provider = register(
    space_id = os.getenv("ARIZE_SPACE_ID"),
    api_key = os.getenv("ARIZE_API_KEY"),
    project_name = os.getenv("ARIZE_PROJECT_NAME"),
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

"""Startup initialization: group one-time embeddings and index checks in a single trace."""
# Tracer for custom spans in this module (after global provider is set)
tracer = trace.get_tracer(__name__)

with spanner(
    tracer=tracer,
    name="startup",
    kind=INIT,
    metadata={
        "redis.host": os.environ.get("REDIS_HOST"),
        "redis.port": os.environ.get("REDIS_PORT"),
    },
):
    # Create Redis client within the async context
    redis_config = RedisClient(
        host=os.environ.get("REDIS_HOST"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        username=os.environ.get("REDIS_USERNAME"),
        password=os.environ.get("REDIS_PASSWORD"),
    )
    sync_redis_client = redis_config.get_sync_client()
    async_redis_client = redis_config.get_async_client()
    # embeddings_provider = OpenAIEmbeddingProvider()
        
    # embedding_cache = RedisEmbeddingsCache(
    #     async_redis_client=async_redis_client,
    #     embedding_provider=embeddings_provider,
    # )

    # semantic_cache = ConversationMemoryCache(
    #     redis_client=sync_redis_client,
    #     embedding_cache=embedding_cache,
    # )

    redis_session_manager = RedisSessionManager(
        async_redis_client=async_redis_client,
    )

openai_client = OpenAIResponsesAPI()
# openai_client = OpenAIChatCompletionAPI()

skip_cache = bool(os.getenv("SKIP_CACHE", False))

turn_id = generate_id(8)

async def generate_response(
    query: str,
    session_id: str, 
    user_id: str, 
    turn_id: str,
    **kwargs,
):
    async with async_spanner(
        tracer=tracer,
        name="GenerateResponse",
        kind=OpenInferenceSpanKindValues.CHAIN,
        session_id=session_id,
        user_id=user_id,
        turn_id=turn_id,
        input=query,
    ):
        turn_completed = False
        tool_call = False

        conversation_history = []
        messages = []

        summary = None
        turns_after_last_summary = 0
        total_token_count_of_context = 0

        user_query_message = Message(
            role=Role.HUMAN,
            tool_call_id="null",
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            metadata={},
            content=query,
            function_call=None,
        )

        while not turn_completed:
            if not tool_call:
                async with async_spanner(
                    tracer=tracer,
                    name="RedisGetContextForLLM",
                    kind=DATABASE,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    input=query,
                ):
                    context: Context = await redis_session_manager.get_context_for_llm(session_id)
                system_message = openai_client.build_system_message(
                    instructions="You are a helpful assistant.",
                    user_id=user_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    summary=context.summary,
                )
                conversation_history = [system_message] + context.previous_conversation
                conversation_history.append(user_query_message)
                turns_after_last_summary = context.turns_after_last_summary 
                total_token_count_of_context = context.context_token_count

            async with async_spanner(
                tracer=tracer,
                name="LLMGenerateAndSummarize",
                kind=OpenInferenceSpanKindValues.LLM,
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                input=query,
                metadata={
                    "tool_call_input": tool_call,
                    "context_token_count": total_token_count_of_context,
                    "turns_after_last_summary": turns_after_last_summary,
                },
            ):
                (messages, tool_call), summary = await asyncio.gather(
                    openai_client.generate_response(
                        conversation_history=conversation_history,
                        user_id=user_id,
                        turn_id=turn_id,
                        session_id=session_id,
                        tools=[GetWeather(), GetHoroscope(), GetCompanyName()],
                    ),
                    openai_client.generate_summary(
                        conversation_to_summarize=context.previous_conversation,
                        previous_summary=context.summary,
                        query=query,
                        turns_after_last_summary=turns_after_last_summary,
                        context_token_count=total_token_count_of_context, # Wrong -> need total count after last summary
                        tool_call=tool_call,
                    ),
                )

            # If tool call is not made then turn is completed. If tool call is made 
            # then turn will be completed once AI executes the tool call, in the next iteration.
            if tool_call:
                conversation_history.extend(messages)
            else:
                turn_completed = True
                
        return [user_query_message, *messages], summary, turns_after_last_summary


async def generate_answer(query: str, session: Session, user: User, turn_number: int, state: State):
    async with async_spanner(
        tracer=tracer,
        name="GenerateAnswer",
        kind=OpenInferenceSpanKindValues.CHAIN,
        session_id=session.id,
        user_id=user.id,
        turn_number=turn_number,
        input=query,
        metadata={
            "turn_number": turn_number,
        },
    ) as span:
        messages, \
        summary, \
        turns_after_last_summary = await generate_response(
            query=query,
            session=session,
            user=user,
            turn_number=turn_number,
            state=state,
            skip_semantic_cache=skip_cache,
        )
        
        # Add messages to long-term store under its own span
        async with async_spanner(
            tracer=tracer,
            name="RedisAddMessage",
            kind=DATABASE,
            session_id=session.id,
            user_id=user.id,
            turn_number=turn_number,
            input=query,
            metadata={
                "query": query,
                "turn_number": turn_number,
            },
        ):
            await redis_session_manager.add_message(
                messages=messages, 
                new_summary=summary, 
                turn_number=turn_number,
                prev_n_turn_after_last_summary=turns_after_last_summary, 
            )

        if len(messages) >= 1:
            if messages[0].role == Role.HUMAN:
                for message in messages[1:]:
                    if message.content:
                        print(f"{message.role.value}:", message.content)
            else:
                for message in messages:
                    if message.content:
                        print(f"{message.role.value}:", message.content)
        if messages[-1].role == Role.AI:
            # Set the output attribute with the last AI message content
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, messages[-1].content or "")

async def interactive_main():
    await MongoDB.init()

    user, session = await get_or_create_test_user()
    state = State()

    turn_number = 0
    while True:
        # Read blocking input without leaving the event loop
        query = await asyncio.get_running_loop().run_in_executor(None, input, "You: ")
        if query == "exit":
            break
        turn_number += 1
        await generate_answer(query, session, user, turn_number, state)

if __name__ == "__main__":
    asyncio.run(interactive_main())
    
# Next steps
# 1. Test with different scenarios - for turn based condition and for token based condition
# 2. Is there a way i can use semantic cache on this?

# 4. Add route for getting all chat history - make sure nothing is missed
# 5. Add route for getting all sessions
# 6. Implement the agentic Loop

# First message in session?
#     → Skip context retrieval entirely (no history exists)

# Short session (< 10 messages)?
#     → Fetch last N messages directly (no vector search, instant)

# Long session?
#     → Use vector search (accept the latency, but it's rare)