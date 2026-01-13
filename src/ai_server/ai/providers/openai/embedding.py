from ai_server.ai.providers.embedding_provider import EmbeddingProvider
from ai_server.utils.general import get_env_int
from ai_server.config import BASE_EMBEDDING_MODEL
from ai_server.constants import OPENAI

import logging
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self, 
        model_name: str = BASE_EMBEDDING_MODEL, 
        dimensions: int | None = None
    ) -> None:
        self.model_name = model_name
        env_dims = dimensions or get_env_int("OPENAI_EMBEDDING_DIMENSIONS")
        self.client = OpenAIEmbeddings(model=model_name, dimensions=env_dims)
        self.dimensions = env_dims or self._set_dimension()
        dim_source = "env" if env_dims is not None else "probe"
        logger.info(
            "OpenAIEmbeddingProvider initialized: model=%s dims=%s source=%s",
            self.model_name,
            self.dimensions,
            dim_source,
        )
        super().__init__(
            provider=OPENAI, 
            client=self.client, 
            model_name=self.model_name, 
            dimensions=self.dimensions
        )
    
    def _set_dimension(self) -> int:
        try:
            embedding = self.client.embed_query("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the LangChain embeddings: {str(ke)}")
        except Exception as e:
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
