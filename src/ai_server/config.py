## Configuration file for the ai-server

from ai_server.utils.general import _env_flag

import os

# Server config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 4))
RELOAD = _env_flag("RELOAD", True)
BASE_PATH = os.getenv("BASE_PATH", "/api")

# CORS Configuration
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = _env_flag("CORS_ALLOW_CREDENTIALS", True)
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "*").split(",")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")

# Development mode - enables index dropping and other dev features
DEV_MODE = _env_flag("DEV_MODE", False)

# Note: Pagination config (DEFAULT_MESSAGE_PAGE_SIZE, etc.) is now in omniagent.config
