"""Unit tests for retrieval functionality."""
import pytest
from docint.retrieval.bm25 import BM25Index


class TestBM25Index:
    """Tests for BM25 keyword search."""

    def test_basic_search(self):
        """Should find documents with matching keywords."""
        index = BM25Index()
        index.add(
            ids=["doc1", "doc2", "doc3"],
            contents=[
                "The quick brown fox jumps",
                "A lazy dog sleeps",
                "The brown dog runs fast"
            ],
            metadatas=[{}, {}, {}]
        )

        results = index.search("brown dog", top_k=2)

        # doc3 has both "brown" and "dog"
        assert results[0].chunk_id == "doc3"

    def test_exact_match_ranking(self):
        """Documents with more keyword matches should rank higher."""
        index = BM25Index()
        index.add(
            ids=["doc1", "doc2"],
            contents=[
                "revenue growth in Q3",
                "Q3 revenue growth was strong revenue increase"
            ],
            metadatas=[{}, {}]
        )

        results = index.search("revenue", top_k=2)

        # doc2 has "revenue" twice
        assert results[0].chunk_id == "doc2"

    def test_no_match(self):
        """Should return empty when no keywords match."""
        index = BM25Index()
        index.add(
            ids=["doc1"],
            contents=["The quick brown fox"],
            metadatas=[{}]
        )

        results = index.search("elephant zebra", top_k=5)

        assert len(results) == 0

    def test_case_insensitive(self):
        """Search should be case insensitive."""
        index = BM25Index()
        index.add(
            ids=["doc1"],
            contents=["REVENUE growth STRONG"],
            metadatas=[{}]
        )

        results = index.search("revenue strong", top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "doc1"

    def test_empty_query(self):
        """Should handle empty query."""
        index = BM25Index()
        index.add(
            ids=["doc1"],
            contents=["Some content"],
            metadatas=[{}]
        )

        results = index.search("", top_k=5)

        assert len(results) == 0

    def test_count(self):
        """Should track document count."""
        index = BM25Index()

        assert index.count() == 0

        index.add(
            ids=["doc1", "doc2"],
            contents=["content 1", "content 2"],
            metadatas=[{}, {}]
        )

        assert index.count() == 2

    def test_clear(self):
        """Should clear all documents."""
        index = BM25Index()
        index.add(
            ids=["doc1"],
            contents=["content"],
            metadatas=[{}]
        )

        index.clear()

        assert index.count() == 0


class TestBM25EdgeCases:
    """Edge cases for BM25."""

    def test_single_word_document(self):
        """Should handle single-word documents."""
        index = BM25Index()
        index.add(
            ids=["doc1"],
            contents=["revenue"],
            metadatas=[{}]
        )

        results = index.search("revenue", top_k=1)

        assert len(results) == 1

    def test_duplicate_ids(self):
        """Adding same ID should overwrite."""
        index = BM25Index()
        index.add(ids=["doc1"], contents=["old content"], metadatas=[{}])
        index.add(ids=["doc1"], contents=["new content"], metadatas=[{}])

        results = index.search("new", top_k=1)

        # Should find "new content"
        assert len(results) == 1

    def test_special_characters(self):
        """Should handle special characters in content."""
        index = BM25Index()
        index.add(
            ids=["doc1"],
            contents=["Revenue: $1.5B (15% growth)"],
            metadatas=[{}]
        )

        results = index.search("revenue growth", top_k=1)

        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
