"""Unit tests for chunking functionality."""
import pytest
from docint.ingest.loaders import Document
from docint.ingest.chunkers import FixedChunker, RecursiveChunker


class TestFixedChunker:
    """Tests for fixed-size chunker."""
    
    def test_basic_chunking(self):
        """Should split text into fixed-size chunks."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(content="a" * 250, metadata={"source": "test"})
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
    
    def test_overlap(self):
        """Chunks should overlap by specified amount."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        doc = Document(content="word " * 50, metadata={})
        
        chunks = list(chunker.chunk(doc))
        
        assert chunks[0].content[-20:] in chunks[1].content[:30]
    
    def test_empty_document(self):
        """Should handle empty documents."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(content="", metadata={})
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 0
    
    def test_small_document(self):
        """Document smaller than chunk size should return one chunk."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(content="small text", metadata={})
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 1
        assert chunks[0].content == "small text"


class TestRecursiveChunker:
    """Tests for recursive chunker."""
    
    def test_splits_on_paragraphs(self):
        """Should split on paragraph boundaries when content exceeds chunk size."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        doc = Document(
            content="First paragraph is here with more text.\n\nSecond paragraph has content too.",
            metadata={}
        )
        
        chunks = list(chunker.chunk(doc))
        
        # Should split because total > 50 chars
        assert len(chunks) >= 2
        assert "First" in chunks[0].content
    
    def test_keeps_sections_together(self):
        """Section header should stay with content when under chunk size."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        doc = Document(
            content="## Q3 2024\n\nRevenue was $1.15 billion.",
            metadata={}
        )
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 1
        assert "Q3 2024" in chunks[0].content
        assert "1.15 billion" in chunks[0].content
    
    def test_falls_back_to_sentences(self):
        """Should split on sentences if paragraph is too long."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        doc = Document(
            content="First sentence here. Second sentence here. Third sentence here.",
            metadata={}
        )
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) >= 2
    
    def test_metadata_preserved(self):
        """Chunk should inherit document metadata."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt", "author": "me"}
        )
        
        chunks = list(chunker.chunk(doc))
        
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "me"


class TestChunkerEdgeCases:
    """Edge cases and regression tests."""
    
    def test_unicode_handling(self):
        """Should handle unicode characters."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        doc = Document(content="Hello 你好 مرحبا", metadata={})
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 1
        assert "你好" in chunks[0].content
    
    def test_only_whitespace(self):
        """Should handle whitespace-only documents."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        doc = Document(content="   \n\n   \t  ", metadata={})
        
        chunks = list(chunker.chunk(doc))
        
        assert len(chunks) == 0
    
    def test_chunk_ids_exist(self):
        """Each chunk should have an ID."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        doc = Document(content="Some text here. More text there.", metadata={})
        
        chunks = list(chunker.chunk(doc))
        
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert len(chunk.chunk_id) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
