"""Embedding engine using OpenAI Compatible API."""
import logging
import ssl
from typing import List, Optional
from openai import OpenAI
import time
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedding:
    """Embedding client for OpenAI Compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "bge-m3",
        batch_size: int = 32,
        verify_ssl: bool = True
    ):
        """Initialize embedding client.

        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            batch_size: Batch size for embedding requests
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.verify_ssl = verify_ssl

        # Initialize client with SSL configuration
        if verify_ssl:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            # Disable SSL verification
            http_client = httpx.Client(verify=False)
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=http_client
            )
            logger.warning("SSL verification is disabled. This is not recommended for production.")

        logger.info(f"Initialized embedding client with model: {model}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Small delay to avoid rate limiting
                if batch_num < total_batches:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error embedding batch {batch_num}: {e}")
                raise

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
