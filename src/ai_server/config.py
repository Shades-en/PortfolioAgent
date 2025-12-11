## Configuration file for the ai-server

from ai_server.constants import GPT_4_1_MINI, TEXT_EMBEDDING_3_SMALL, OPENAI
from ai_server.utils.general import _env_flag

import os

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 4))
RELOAD = _env_flag("RELOAD", True)
BASE_PATH = os.getenv("BASE_PATH", "/api")

# Model Configuration
BASE_MODEL = os.getenv("BASE_MODEL", GPT_4_1_MINI)
BASE_EMBEDDING_MODEL = os.getenv("BASE_EMBEDDING_MODEL", TEXT_EMBEDDING_3_SMALL)

# Context Configuration
MAX_TOKEN_THRESHOLD = int(os.getenv("MAX_TOKEN_THRESHOLD", 50000))
MAX_TURNS_TO_FETCH = int(os.getenv("MAX_TURNS_TO_FETCH", 100))

# Tracing config
ENABLE_TRACING = _env_flag("ENABLE_TRACING", True)
ENABLE_INPUT_GUARDRAIL = _env_flag("ENABLE_INPUT_GUARDRAIL", True)
ENABLE_OUTPUT_GUARDRAIL = _env_flag("ENABLE_OUTPUT_GUARDRAIL", True)

# Cache
SKIP_CACHE = _env_flag("SKIP_CACHE", False)

# LLM Providers
LLM_PROVIDER = os.getenv("LLM_PROVIDER", OPENAI)
CHAT_COMPLETION = _env_flag("CHAT_COMPLETION", False)

# AGENT
MAX_STEPS = 10

# Session Configuration
DEFAULT_SESSION_NAME = os.getenv("DEFAULT_SESSION_NAME", "New Chat")
TURNS_BETWEEN_CHAT_NAME = int(os.getenv("TURNS_BETWEEN_CHAT_NAME", 10))
MAX_CHAT_NAME_LENGTH = int(os.getenv("MAX_CHAT_NAME_LENGTH", 50))
MAX_CHAT_NAME_WORDS = int(os.getenv("MAX_CHAT_NAME_WORDS", 5))

# Pagination Configuration
DEFAULT_MESSAGE_PAGE_SIZE = int(os.getenv("DEFAULT_MESSAGE_PAGE_SIZE", 50))
MAX_MESSAGE_PAGE_SIZE = int(os.getenv("MAX_MESSAGE_PAGE_SIZE", 100))
DEFAULT_SESSION_PAGE_SIZE = int(os.getenv("DEFAULT_SESSION_PAGE_SIZE", 20))
MAX_SESSION_PAGE_SIZE = int(os.getenv("MAX_SESSION_PAGE_SIZE", 50))
