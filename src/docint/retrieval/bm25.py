"""
BM25 keyword-based retrieval.

BM25 (Best Matching 25) is a ranking function used for keyword search.
It considers term frequency, document length, and inverse document frequency.
"""
import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class BM25Result:
    """Single BM25 search result."""
    chunk_id: str
    content: str
    score: float
    metadata: dict


class BM25Index:
    """
    BM25 index for keyword-based retrieval.

    BM25 scoring formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
    - f(qi,D) = frequency of term qi in document D
    - |D| = length of document D
    - avgdl = average document length
    - k1, b = tuning parameters (typically k1=1.5, b=0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Index storage
        self._documents: dict[str, str] = {}  # id -> content
        self._metadata: dict[str, dict] = {}  # id -> metadata
        self._doc_lengths: dict[str, int] = {}  # id -> word count
        self._avgdl: float = 0.0  # average document length

        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)

        # IDF cache
        self._idf_cache: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Basic tokenization - can be improved with proper NLP
        text = text.lower()
        # Remove punctuation and split
        tokens = []
        current_token = []
        for char in text:
            if char.isalnum():
                current_token.append(char)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
        if current_token:
            tokens.append(''.join(current_token))
        return tokens

    def add(
        self,
        ids: list[str],
        contents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents to the index."""
        if metadatas is None:
            metadatas = [{} for _ in ids]

        for doc_id, content, metadata in zip(ids, contents, metadatas):
            # Store document
            self._documents[doc_id] = content
            self._metadata[doc_id] = metadata

            # Tokenize
            tokens = self._tokenize(content)
            self._doc_lengths[doc_id] = len(tokens)

            # Build inverted index
            term_freq: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                self._inverted_index[term][doc_id] = freq

        # Update average document length
        if self._doc_lengths:
            self._avgdl = sum(self._doc_lengths.values()) / \
                len(self._doc_lengths)

        # Clear IDF cache (needs recalculation)
        self._idf_cache.clear()

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]

        n = len(self._documents)  # total documents
        # documents containing term
        df = len(self._inverted_index.get(term, {}))

        # IDF formula: log((N - df + 0.5) / (df + 0.5))
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        self._idf_cache[term] = idf
        return idf

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        """Search for documents matching the query."""
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 scores for all documents
        scores: dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._inverted_index:
                continue

            idf = self._idf(term)

            for doc_id, tf in self._inverted_index[term].items():
                doc_len = self._doc_lengths[doc_id]

                # BM25 scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * \
                    (1 - self.b + self.b * doc_len / self._avgdl)
                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score and return top_k
        sorted_docs = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            results.append(BM25Result(
                chunk_id=doc_id,
                content=self._documents[doc_id],
                score=score,
                metadata=self._metadata[doc_id],
            ))

        return results

    def count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear the index."""
        self._documents.clear()
        self._metadata.clear()
        self._doc_lengths.clear()
        self._inverted_index.clear()
        self._idf_cache.clear()
        self._avgdl = 0.0
