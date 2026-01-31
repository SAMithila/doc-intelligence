"""
Base embedding interface.

Defines abstract interface for embedding providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: list[list[float]]
    model: str
    dimension: int
    total_tokens: int = 0
    
    def __len__(self) -> int:
        return len(self.embeddings)


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass
    
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Sequence of texts to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass
    
    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = self.embed([text])
        return result.embeddings[0]
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string.
        
        Some models use different embeddings for queries vs documents.
        Override this method if your model supports it.
        """
        return self.embed_single(query)
