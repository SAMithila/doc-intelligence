"""
OpenAI embedding provider.

Uses OpenAI's embedding API for high-quality embeddings.
"""
from typing import Sequence
import openai

from docint.embeddings.base import BaseEmbedder, EmbeddingResult


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding provider.
    
    Models:
    - text-embedding-3-small: 1536 dimensions, fast, cheap
    - text-embedding-3-large: 3072 dimensions, higher quality
    - text-embedding-ada-002: 1536 dimensions, legacy
    """
    
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        self._model = model
        self._batch_size = batch_size
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        
        self._client = openai.OpenAI(api_key=api_key)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self._model,
                dimension=self._dimension,
                total_tokens=0,
            )
        
        # Process in batches
        all_embeddings: list[list[float]] = []
        total_tokens = 0
        
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i:i + self._batch_size])
            
            # Clean texts (remove empty strings)
            cleaned_batch = [t if t.strip() else " " for t in batch]
            
            response = self._client.embeddings.create(
                model=self._model,
                input=cleaned_batch,
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            total_tokens += response.usage.total_tokens
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self._model,
            dimension=self._dimension,
            total_tokens=total_tokens,
        )
