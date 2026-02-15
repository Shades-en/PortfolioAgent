import os

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
    "get_env_int", 
    "_env_flag",
]
