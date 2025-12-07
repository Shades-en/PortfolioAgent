## Configuration file for the ai-server

from ai_server.constants import GPT_4_1_MINI, TEXT_EMBEDDING_3_SMALL

# Server config
HOST = "0.0.0.0"
PORT = 8000
WORKERS = 4
RELOAD = True

BASE_PATH = "/api"

# Model Configuration
BASE_MODEL = GPT_4_1_MINI
BASE_EMBEDDING_MODEL = TEXT_EMBEDDING_3_SMALL

# Context Configuration
MAX_TOKEN_THRESHOLD = 50000
MAX_TURNS_TO_FETCH = 100

# Tracing config
ENABLE_TRACING = True
ENABLE_INPUT_GUARDRAIL = True
ENABLE_OUTPUT_GUARDRAIL = True
