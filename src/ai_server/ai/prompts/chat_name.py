CHAT_NAME_SYSTEM_PROMPT = """You are a helpful assistant that generates concise chat titles."""

CHAT_NAME_USER_PROMPT = """{context}User query: {query}

Generate a concise, meaningful chat title (max {max_words} words) that captures the essence of this conversation.
Return ONLY the title, no quotes or extra text."""
