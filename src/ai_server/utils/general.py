from uuid import uuid4
import os
import tiktoken

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
    from ai_server.config import BASE_MODEL
    
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        _tiktoken_encoder = tiktoken.encoding_for_model(BASE_MODEL)
    return len(_tiktoken_encoder.encode(text))

def safe_get_arg(args: tuple, index: int, default=None):
    """
    Safely extract an argument from args tuple by index.
    
    Args:
        args: Tuple of arguments
        index: Index to extract
        default: Default value if index is out of bounds
    
    Returns:
        Value at index or default
    
    Example:
        >>> args = ("hello", "world", 123)
        >>> safe_get_arg(args, 0)
        'hello'
        >>> safe_get_arg(args, 5, "default")
        'default'
    """
    try:
        return args[index] if index < len(args) else default
    except (IndexError, TypeError):
        return default

def generate_order(order: int, step: int) -> int:
    return step*10 + order

__all__ = [
    "generate_id", 
    "get_env_int", 
    "_env_flag",
    "get_token_count",
    "safe_get_arg",
    "generate_order",
]
