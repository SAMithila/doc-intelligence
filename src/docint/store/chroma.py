"""
ChromaDB vector store implementation.

ChromaDB is a lightweight, embedded vector database.
Good for development and small-to-medium scale deployments.
"""
from typing import Any
import chromadb
from chromadb.config import Settings

from docint.store.base import BaseVectorStore, SearchResult


class ChromaStore(BaseVectorStore):
    """
    ChromaDB vector store implementation.
    
    Features:
    - Embedded mode (no separate server needed)
    - Persistent storage
    - Metadata filtering
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
    ):
        self._collection_name = collection_name
        
        # Initialize client
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
    
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents to ChromaDB."""
        if not ids:
            return
        
        # Ensure metadatas is a list
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        # ChromaDB doesn't allow None values in metadata
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {
                k: v for k, v in meta.items() 
                if v is not None and isinstance(v, (str, int, float, bool))
            }
            clean_metadatas.append(clean_meta)
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=clean_metadatas,
        )
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar documents."""
        # Build query
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        results = self._collection.query(**query_params)
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns distance, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance
                
                search_results.append(SearchResult(
                    chunk_id=results["ids"][0][i],
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))
        
        return search_results
    
    def delete(self, ids: list[str]) -> None:
        """Delete documents from ChromaDB."""
        if ids:
            self._collection.delete(ids=ids)
    
    def count(self) -> int:
        """Return total document count."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all documents."""
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
