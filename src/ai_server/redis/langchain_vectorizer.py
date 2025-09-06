"""LangChain Text Vectorizer for Redis VL integration.

This vectorizer bridges LangChain's Embeddings interface with Redis VL's BaseVectorizer,
allowing any LangChain embedding provider to be used with Redis vector operations.
"""

from typing import TYPE_CHECKING, List, Optional
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
    from langchain_core.embeddings import Embeddings

from redisvl.utils.vectorize import BaseVectorizer

# Constants for error messages
_STR_INPUT_ERROR: str = "Must pass in a str value to embed."
_LIST_STR_INPUT_ERROR: str = "Must pass in a list of str values to embed."


class LangchainTextVectorizer(BaseVectorizer):
    """LangChain Text Vectorizer for Redis VL.
    
    This vectorizer allows any LangChain embedding model to be used with Redis VL
    by implementing the BaseVectorizer interface. It acts as a bridge between
    LangChain's Embeddings interface and Redis VL's vectorization system.
    
    The vectorizer supports caching to improve performance when generating
    embeddings for repeated text inputs.
    
    Requirements:
        - A LangChain embedding model instance must be provided
        - The embedding model must implement embed_documents() and embed_query() methods
    
    Examples:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from redisvl.extensions.cache.embeddings import EmbeddingsCache
        >>> 
        >>> # Create LangChain embedding model
        >>> langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        >>> 
        >>> # Basic usage
        >>> vectorizer = LangchainTextVectorizer(
        ...     langchain_embeddings=langchain_embeddings,
        ...     model="text-embedding-3-large"
        ... )
        >>> embedding = vectorizer.embed("Hello, world!")
        >>> 
        >>> # With caching enabled
        >>> cache = EmbeddingsCache(name="langchain_embeddings_cache")
        >>> vectorizer = LangchainTextVectorizer(
        ...     langchain_embeddings=langchain_embeddings,
        ...     model="text-embedding-3-large",
        ...     cache=cache
        ... )
        >>> 
        >>> # First call will compute and cache the embedding
        >>> embedding1 = vectorizer.embed("Hello, world!")
        >>> 
        >>> # Second call will retrieve from cache
        >>> embedding2 = vectorizer.embed("Hello, world!")
        >>> 
        >>> # Batch processing
        >>> embeddings = vectorizer.embed_many(
        ...     ["Hello, world!", "How are you?"],
        ...     batch_size=2
        ... )
    """

    _langchain_embeddings: "Embeddings" = PrivateAttr()
    
    def __init__(
        self,
        langchain_embeddings: "Embeddings",
        model: str = "langchain-model",
        dtype: str = "float32",
        cache: Optional["EmbeddingsCache"] = None,
        **kwargs,
    ):
        """Initialize the LangChain text vectorizer.
        
        Args:
            langchain_embeddings: A LangChain Embeddings instance (e.g., OpenAIEmbeddings,
                HuggingFaceEmbeddings, etc.)
            model: Model identifier for caching and identification purposes.
                Defaults to 'langchain-model'.
            dtype: The default datatype to use when embedding text as byte arrays.
                Used when setting `as_buffer=True` in calls to embed() and embed_many().
                Defaults to 'float32'.
            cache: Optional EmbeddingsCache instance to cache embeddings for
                better performance with repeated texts. Defaults to None.
            **kwargs: Additional parameters (currently unused but maintained for compatibility).
        
        Raises:
            TypeError: If langchain_embeddings doesn't implement required methods.
            ValueError: If there is an error setting the embedding model dimensions.
            ValueError: If an invalid dtype is provided.
        """
        # Validate that the provided embeddings object has required methods
        if not hasattr(langchain_embeddings, 'embed_documents'):
            raise TypeError(
                "langchain_embeddings must implement embed_documents() method"
            )
        if not hasattr(langchain_embeddings, 'embed_query'):
            raise TypeError(
                "langchain_embeddings must implement embed_query() method"
            )
        
        # Calculate dimensions using the provided embeddings instance
        calculated_dims = self._calculate_dims(langchain_embeddings)
        
        # Initialize the base class with calculated dimensions
        super().__init__(model=model, dtype=dtype, dims=calculated_dims, cache=cache)
        
        # Set the private embeddings attribute AFTER BaseModel initialization
        self._langchain_embeddings = langchain_embeddings
    
    def _calculate_dims(self, embeddings: "Embeddings") -> int:
        """Calculate the dimensionality of the embedding model by making a test call.
        
        Args:
            embeddings: The LangChain embeddings instance to use for the calculation.
        
        Returns:
            int: Dimensionality of the embedding model
        
        Raises:
            ValueError: If embedding dimensions cannot be determined
        """
        try:
            # Use LangChain's embed_query method directly for dimension calculation
            embedding = embeddings.embed_query("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the LangChain embeddings: {str(ke)}")
        except Exception as e:
            raise ValueError(f"Error calculating embedding model dimensions: {str(e)}")
    
    def _embed(self, text: str, **kwargs) -> List[float]:
        """Generate a vector embedding for a single text using the LangChain model.
        
        Args:
            text: Text to embed
            **kwargs: Additional model-specific parameters (passed to LangChain)
        
        Returns:
            List[float]: Vector embedding as a list of floats
        
        Raises:
            TypeError: If the input is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError(_STR_INPUT_ERROR)
        
        try:
            # Use LangChain's embed_query method for single text embedding
            embedding = self._langchain_embeddings.embed_query(text)
            return embedding
        except Exception as e:
            raise ValueError(f"Embedding text failed: {e}")
    
    def _embed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts using the LangChain model.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (handled by LangChain)
            **kwargs: Additional model-specific parameters (passed to LangChain)
        
        Returns:
            List[List[float]]: List of vector embeddings as lists of floats
        
        Raises:
            TypeError: If the input is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError(_LIST_STR_INPUT_ERROR)
        if texts and not isinstance(texts[0], str):
            raise TypeError(_LIST_STR_INPUT_ERROR)
        
        try:
            # Use LangChain's embed_documents method for batch embedding
            # Note: LangChain handles batching internally, so we pass all texts at once
            # The batch_size parameter is maintained for interface compatibility
            embeddings = self._langchain_embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding texts failed: {e}")
    
    async def _aembed(self, text: str, **kwargs) -> List[float]:
        """Asynchronously generate a vector embedding for a single text.
        
        Args:
            text: Text to embed
            **kwargs: Additional model-specific parameters (passed to LangChain)
        
        Returns:
            List[float]: Vector embedding as a list of floats
        
        Raises:
            TypeError: If the input is not a string
            ValueError: If embedding fails
        """
        if not isinstance(text, str):
            raise TypeError(_STR_INPUT_ERROR)
        
        try:
            # Check if LangChain embeddings supports async operations
            if hasattr(self._langchain_embeddings, 'aembed_query'):
                embedding = await self._langchain_embeddings.aembed_query(text)
                return embedding
            else:
                # Fall back to sync method
                return self._embed(text, **kwargs)
        except Exception as e:
            raise ValueError(f"Async embedding text failed: {e}")
    
    async def _aembed_many(
        self, texts: List[str], batch_size: int = 10, **kwargs
    ) -> List[List[float]]:
        """Asynchronously generate vector embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (handled by LangChain)
            **kwargs: Additional model-specific parameters (passed to LangChain)
        
        Returns:
            List[List[float]]: List of vector embeddings as lists of floats
        
        Raises:
            TypeError: If the input is not a list of strings
            ValueError: If embedding fails
        """
        if not isinstance(texts, list):
            raise TypeError(_LIST_STR_INPUT_ERROR)
        if texts and not isinstance(texts[0], str):
            raise TypeError(_LIST_STR_INPUT_ERROR)
        
        try:
            # Check if LangChain embeddings supports async operations
            if hasattr(self._langchain_embeddings, 'aembed_documents'):
                embeddings = await self._langchain_embeddings.aembed_documents(texts)
                return embeddings
            else:
                # Fall back to sync method
                return self._embed_many(texts, batch_size, **kwargs)
        except Exception as e:
            raise ValueError(f"Async embedding texts failed: {e}")
    
    @property
    def type(self) -> str:
        """Return the type of vectorizer."""
        return "langchain"