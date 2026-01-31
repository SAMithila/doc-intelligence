"""
Base vector store interface.

Defines abstract interface for vector databases.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Single search result from vector store."""
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]
    
    def __repr__(self) -> str:
        return f"SearchResult(id={self.chunk_id}, score={self.score:.4f})"


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """
        Add documents to the store.
        
        Args:
            ids: Unique identifiers for each document
            embeddings: Vector embeddings
            contents: Original text content
            metadatas: Optional metadata for each document
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult ordered by relevance
        """
        pass
    
    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of documents."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all documents."""
        pass
