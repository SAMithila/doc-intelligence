"""
Retrieval implementations.

Baseline: Simple semantic search retriever.
"""
from dataclasses import dataclass

from docint.embeddings.base import BaseEmbedder
from docint.store.base import BaseVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Result of retrieval operation."""
    results: list[SearchResult]
    query: str
    retriever_type: str = "simple"
    
    @property
    def contexts(self) -> list[str]:
        """Get just the content strings."""
        return [r.content for r in self.results]
    
    def __len__(self) -> int:
        return len(self.results)


class SimpleRetriever:
    """
    Simple semantic search retriever.
    
    Baseline implementation:
    1. Embed query
    2. Search vector store
    3. Return top-k results
    
    No fancy features - just pure similarity search.
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            top_k: Override default top_k
            filter_metadata: Optional metadata filter
            
        Returns:
            RetrievalResult with ranked documents
        """
        k = top_k or self.top_k
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter_metadata=filter_metadata,
        )
        
        return RetrievalResult(
            results=results,
            query=query,
            retriever_type="simple",
        )
