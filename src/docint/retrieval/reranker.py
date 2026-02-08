"""
Cross-encoder reranking for improved retrieval accuracy.

Cross-encoders process query-document pairs together,
providing more accurate relevance scores than bi-encoders.
"""
from dataclasses import dataclass
import openai


@dataclass
class RerankResult:
    """Single reranked result."""
    chunk_id: str
    content: str
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int
    metadata: dict


class LLMReranker:
    """
    LLM-based reranker using GPT for relevance scoring.

    Uses the LLM to score query-document relevance on a 0-10 scale.
    More expensive but doesn't require additional models.

    For production, consider using:
    - sentence-transformers cross-encoders (free, local)
    - Cohere Rerank API
    - Jina Reranker
    """

    RERANK_PROMPT = """Rate the relevance of the following document to the query.
Score from 0-10 where:
- 0: Completely irrelevant
- 5: Somewhat relevant, mentions related topics
- 10: Highly relevant, directly answers the query

Query: {query}

Document: {document}

Respond with ONLY a number from 0-10, nothing else."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": self.RERANK_PROMPT.format(
                        query=query,
                        document=document[:1000],  # Limit document length
                    )
                }],
                temperature=0,
                max_tokens=5,
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return min(max(score, 0), 10) / 10  # Normalize to 0-1

        except (ValueError, TypeError):
            return 0.5  # Default score on error

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Rerank results using LLM scoring.

        Args:
            query: The search query
            results: List of dicts with 'chunk_id', 'content', 'score', 'metadata'
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Score each result
        scored_results = []
        for i, result in enumerate(results):
            rerank_score = self._score_pair(query, result['content'])
            scored_results.append({
                **result,
                'original_rank': i + 1,
                'original_score': result.get('score', 0),
                'rerank_score': rerank_score,
            })

        # Sort by rerank score
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Build final results
        final_results = []
        for i, r in enumerate(scored_results[:top_k]):
            final_results.append(RerankResult(
                chunk_id=r['chunk_id'],
                content=r['content'],
                original_score=r['original_score'],
                rerank_score=r['rerank_score'],
                original_rank=r['original_rank'],
                new_rank=i + 1,
                metadata=r.get('metadata', {}),
            ))

        return final_results


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Requires: pip install sentence-transformers

    Uses local model - free and fast after initial download.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            self._available = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Using fallback.")
            self._available = False
            self._model = None

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        if not self._available:
            # Fallback: return original order
            return [
                RerankResult(
                    chunk_id=r['chunk_id'],
                    content=r['content'],
                    original_score=r.get('score', 0),
                    rerank_score=r.get('score', 0),
                    original_rank=i + 1,
                    new_rank=i + 1,
                    metadata=r.get('metadata', {}),
                )
                for i, r in enumerate(results[:top_k])
            ]

        # Create query-document pairs
        pairs = [(query, r['content']) for r in results]

        # Get scores from cross-encoder
        scores = self._model.predict(pairs)

        # Combine with original results
        scored_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            scored_results.append({
                **result,
                'original_rank': i + 1,
                'original_score': result.get('score', 0),
                'rerank_score': float(score),
            })

        # Sort by rerank score
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Build final results
        final_results = []
        for i, r in enumerate(scored_results[:top_k]):
            final_results.append(RerankResult(
                chunk_id=r['chunk_id'],
                content=r['content'],
                original_score=r['original_score'],
                rerank_score=r['rerank_score'],
                original_rank=r['original_rank'],
                new_rank=i + 1,
                metadata=r.get('metadata', {}),
            ))

        return final_results
