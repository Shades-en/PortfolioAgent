from uuid import uuid4
import os
import tiktoken
from ai_server.config import BASE_MODEL

# Cached encoder instance
_tiktoken_encoder: tiktoken.Encoding | None = None

def generate_id(length: int = 8) -> str:
    """Generate a short UUID with specified length."""
    return uuid4().hex[:length]

def get_env_int(name: str, default: int | None = None) -> int | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {v!r}")

def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment flags.

    Accepts true values: '1', 'true', 'yes', 'on' (case-insensitive). False for others or unset.
    """
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def get_token_count(text: str) -> int:
    """Count tokens in text using tiktoken encoder for BASE_MODEL."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        _tiktoken_encoder = tiktoken.encoding_for_model(BASE_MODEL)
    return len(_tiktoken_encoder.encode(text))

__all__ = [
    "generate_id", 
    "get_env_int", 
    "_env_flag",
    "get_token_count",
]
