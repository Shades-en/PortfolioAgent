from abc import ABC
from langchain_core.embeddings import Embeddings

class EmbeddingProvider(ABC):
    def __init__(self, provider: str, client: Embeddings, model_name: str, dimensions: int) -> None:
        self.provider = provider
        self.client = client
        self.dimensions = dimensions
        self.model_name = model_name