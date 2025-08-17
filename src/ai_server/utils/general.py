from uuid import uuid4

def generate_id(length: int = 8) -> str:
    """Generate a short UUID with specified length."""
    return uuid4().hex[:length]