# OPENAI Models
GPT_4_1_MINI = "gpt-4.1-mini"
GPT_4_1 = "gpt-4.1"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

# PROVIDERS
OPENAI = "openai"

# SERVER
SERVER_STARTUP = "SERVER_STARTUP"
SERVER_SHUTDOWN = "SERVER_SHUTDOWN"

# Streaming (AI SDK data stream protocol)
STREAM_EVENT_START = "start"
STREAM_EVENT_TEXT_START = "text-start"
STREAM_EVENT_TEXT_DELTA = "text-delta"
STREAM_EVENT_TEXT_END = "text-end"
STREAM_EVENT_REASONING_START = "reasoning-start"
STREAM_EVENT_REASONING_DELTA = "reasoning-delta"
STREAM_EVENT_REASONING_END = "reasoning-end"
STREAM_EVENT_TOOL_INPUT_START = "tool-input-start"
STREAM_EVENT_TOOL_INPUT_DELTA = "tool-input-delta"
STREAM_EVENT_TOOL_INPUT_AVAILABLE = "tool-input-available"
STREAM_EVENT_TOOL_OUTPUT_AVAILABLE = "tool-output-available"
STREAM_EVENT_FINISH_STEP = "finish-step"
STREAM_EVENT_FINISH = "finish"
STREAM_EVENT_ERROR = "error"
STREAM_EVENT_DATA_SESSION = "data-session"
STREAM_DONE_SENTINEL = "[DONE]"
STREAM_HEADER_NAME = "x-vercel-ai-ui-message-stream"
STREAM_HEADER_VERSION = "v1"

# OpenAI streaming event types
OPENAI_EVENT_RESPONSE_CREATED = "response.created"
OPENAI_EVENT_TEXT_DELTA = "response.output_text.delta"
OPENAI_EVENT_TEXT_DONE = "response.output_text.done"
OPENAI_EVENT_REASONING_DELTA = "response.reasoning_text.delta"
OPENAI_EVENT_REASONING_DONE = "response.reasoning_text.done"
OPENAI_EVENT_FUNCTION_ARGS_DELTA = "response.function_call_arguments.delta"
OPENAI_EVENT_FUNCTION_ARGS_DONE = "response.function_call_arguments.done"
OPENAI_EVENT_COMPLETED = "response.completed"
OPENAI_EVENT_FAILED = "response.failed"
