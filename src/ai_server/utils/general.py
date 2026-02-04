from uuid import uuid4
import os
import secrets
import tiktoken
from bson import ObjectId

# Cached encoder instance
_tiktoken_encoder: tiktoken.Encoding | None = None

# URL-safe alphabet for nanoid (same as AI SDK)
_NANOID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"

def _generate_nanoid(length: int = 21) -> str:
    """
    Generate a nanoid-style ID using cryptographically secure random generation.
    Uses the same URL-safe alphabet as Vercel AI SDK: A-Za-z0-9_-
    
    Args:
        length: Length of the ID (default 21, matching nanoid default)
    
    Returns:
        A URL-safe random string ID
    
    Example:
        >>> _generate_nanoid(16)
        'kAANsGIQ6xRJp4Zc'
    """
    return ''.join(secrets.choice(_NANOID_ALPHABET) for _ in range(length))

def generate_id(length: int = 8, id_type: str = "uuid") -> str:
    """
    Generate a unique ID with specified length and type.
    
    Args:
        length: Length of the ID
        id_type: Type of ID to generate:
            - "mongodb": MongoDB-compatible ObjectId (24 chars, ignores length param)
            - "nanoid": AI SDK-style nanoid using URL-safe chars (A-Za-z0-9_-)
            - "uuid": UUID hex string truncated to specified length (default)
    
    Returns:
        Generated ID string
    
    Examples:
        >>> generate_id(24, "mongodb")  # MongoDB ObjectId
        '507f1f77bcf86cd799439011'
        >>> generate_id(16, "nanoid")   # AI SDK style
        'kAANsGIQ6xRJp4Zc'
        >>> generate_id(8, "uuid")      # Regular UUID
        'a3f5c8d1'
    """
    if id_type == "mongodb":
        return str(ObjectId())
    elif id_type == "nanoid":
        return _generate_nanoid(length)
    else:  # uuid (default)
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

def generate_order(step: int, order: int) -> int:
    return step*10 + order

__all__ = [
    "generate_id", 
    "get_env_int", 
    "_env_flag",
    "get_token_count",
    "safe_get_arg",
    "generate_order",
]
