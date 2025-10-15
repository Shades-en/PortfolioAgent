from uuid import uuid4
import os

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

__all__ = [
    "generate_id", 
    "get_env_int", 
    "_env_flag"
]
