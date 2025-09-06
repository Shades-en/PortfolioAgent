from redisvl.extensions.cache.embeddings import EmbeddingsCache
from langchain_core.embeddings import Embeddings
from redis.asyncio import Redis as AsyncRedis
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisEmbeddingsCache:
    def __init__(self, async_redis_client: AsyncRedis, embedding_client: Embeddings, model_name: str = None) -> None:
        self.async_redis_client: AsyncRedis = async_redis_client
        self.embedding_client: Embeddings = embedding_client
        self.model_name: str = self._get_model_name(embedding_client, model_name)
        self.cache = EmbeddingsCache(
            name="embedcache",
            async_redis_client=async_redis_client,
            ttl=None,
        )
        self.dims = self._set_dimension()
    
    def _set_dimension(self) -> int:
        try:
            embedding = self.embedding_client.embed_query("dimension check")
            return len(embedding)
        except (KeyError, IndexError) as ke:
            raise ValueError(f"Unexpected response from the LangChain embeddings: {str(ke)}")
        except Exception as e:
            raise ValueError(f"Error setting embedding model dimensions: {str(e)}")
    
    def _get_model_name(self, client: Embeddings, model_name: str = None) -> str:
        if model_name:
            return model_name
        if hasattr(client, "model"):
            return client.model
        return "unknown"

    async def embed_query(self, text: str, metadata: dict = None, skip_cache: bool = False) -> List[float]:
        if not skip_cache:
            logger.info("Checking cache for query")
            if result := await self.cache.aget(text=text, model_name=self.model_name):
                logger.info("Cached embedding found")
                return result.get("embedding")
        logger.info("New embedding computed")
        embedding = await self.embedding_client.aembed_query(text)
        if not skip_cache:
            await self.cache.aset( 
                text=text,
                model_name=self.model_name,
                embedding=embedding,
                metadata=metadata,
            )
        logger.info("New embedding computed")
        return embedding

    async def embed_documents(self, documents: List[str], metadata: dict = None, skip_cache: bool = False) -> List[float]:
        if not skip_cache:
            logger.info(f"Checking cache for {len(documents)} documents")
            cached_embedded_docs = await self.cache.amget(texts=documents, model_name=self.model_name)
            not_cached_documents = []
            index = 0
            
            # Filter out documents that are not cached along with their index
            for document, embedding_result in zip(documents, cached_embedded_docs):
                if not embedding_result:
                    not_cached_documents.append({
                        "text": document,
                        "index": index,
                    })
                index += 1
            
            # If there are documents that are not cached, compute their embeddings and add them to the cache
            if len(not_cached_documents) > 0:
                not_cached_embeddings = await self.embedding_client.aembed_documents(
                    texts=[document["text"] for document in not_cached_documents],
                )
                documents = [
                    {
                        "text": document["text"],
                        "embedding": embedding,
                        "model_name": self.model_name,
                        "metadata": metadata,
                    }
                    for document, embedding in zip(not_cached_documents, not_cached_embeddings)
                ]
                await self.cache.amset(documents)
                logger.info(f"{len(not_cached_documents)} Embeddings needed to be computed out of {len(cached_embedded_docs)}")

                # Update the cached embedded documents with the new embeddings at their original index
                for document, embedding in zip(not_cached_documents, not_cached_embeddings):
                    cached_embedded_docs[document["index"]] = {
                        "text": document["text"],
                        "embedding": embedding,
                        "model_name": self.model_name,
                        "metadata": metadata,
                    }
            else:
                logger.info("No new embeddings need to becomputed")
            return [doc["embedding"] for doc in cached_embedded_docs]
            
        embeddings = await self.embedding_client.aembed_documents(documents)
        documents = [
            {
                "text": document,
                "embedding": embedding,
                "model_name": self.model_name,
                "metadata": metadata,
            }
            for document, embedding in zip(documents, embeddings)
        ]
        logger.info(f"{len(documents)} Embeddings computed without storing in cache")
        return embeddings

    async def clear_cache(self):
        await self.cache.aclear()