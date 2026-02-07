"""
Hybrid retrieval combining semantic search and BM25.

Uses Reciprocal Rank Fusion (RRF) to combine rankings.
"""
from dataclasses import dataclass

from docint.embeddings.base import BaseEmbedder
from docint.store.base import BaseVectorStore, SearchResult
from docint.retrieval.bm25 import BM25Index


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    chunk_id: str
    content: str
    score: float
    semantic_rank: int | None
    bm25_rank: int | None
    metadata: dict


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings:
    RRF(d) = Σ 1 / (k + rank(d))

    Where k is a constant (default 60) that reduces the impact of high rankings.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        bm25_index: BM25Index,
        top_k: int = 5,
        rrf_k: int = 60,
        semantic_weight: float = 0.5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        """
        Retrieve documents using hybrid search.

        1. Get top results from semantic search
        2. Get top results from BM25
        3. Combine using RRF
        """
        k = top_k or self.top_k

        # Get more candidates than needed for fusion
        candidate_k = k * 3

        # Semantic search
        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        # BM25 search
        bm25_results = self.bm25_index.search(query, top_k=candidate_k)

        # Build rank dictionaries
        semantic_ranks: dict[str, int] = {
            r.chunk_id: i + 1 for i, r in enumerate(semantic_results)
        }
        bm25_ranks: dict[str, int] = {
            r.chunk_id: i + 1 for i, r in enumerate(bm25_results)
        }

        # Get all unique document IDs
        all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        for doc_id in all_ids:
            score = 0.0

            # Semantic contribution
            if doc_id in semantic_ranks:
                score += self.semantic_weight / \
                    (self.rrf_k + semantic_ranks[doc_id])

            # BM25 contribution
            if doc_id in bm25_ranks:
                score += (1 - self.semantic_weight) / \
                    (self.rrf_k + bm25_ranks[doc_id])

            rrf_scores[doc_id] = score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(),
                            key=lambda x: rrf_scores[x], reverse=True)[:k]

        # Build results
        results = []

        # Create content lookup from both result sets
        content_lookup: dict[str, str] = {}
        metadata_lookup: dict[str, dict] = {}

        for r in semantic_results:
            content_lookup[r.chunk_id] = r.content
            metadata_lookup[r.chunk_id] = r.metadata

        for r in bm25_results:
            if r.chunk_id not in content_lookup:
                content_lookup[r.chunk_id] = r.content
                metadata_lookup[r.chunk_id] = r.metadata

        for doc_id in sorted_ids:
            results.append(HybridResult(
                chunk_id=doc_id,
                content=content_lookup.get(doc_id, ""),
                score=rrf_scores[doc_id],
                semantic_rank=semantic_ranks.get(doc_id),
                bm25_rank=bm25_ranks.get(doc_id),
                metadata=metadata_lookup.get(doc_id, {}),
            ))

        return results
