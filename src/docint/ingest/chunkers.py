"""
Document chunking strategies.

Baseline: Fixed-size chunking with overlap.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
import hashlib

from docint.ingest.loaders import Document


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        """Generate unique ID from content hash."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def __len__(self) -> int:
        return len(self.content)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: Document) -> Iterator[Chunk]:
        """Split document into chunks."""
        pass
    
    def chunk_many(self, documents: list[Document]) -> Iterator[Chunk]:
        """Chunk multiple documents."""
        for doc in documents:
            yield from self.chunk(doc)


class FixedChunker(BaseChunker):
    """
    Fixed-size chunking with overlap.
    
    Simple baseline strategy that splits text by character count.
    
    Pros:
    - Simple and predictable
    - Consistent chunk sizes
    
    Cons:
    - Can split mid-word or mid-sentence
    - No awareness of document structure
    """
    
    def __init__(
        self, 
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, document: Document) -> Iterator[Chunk]:
        """Split document into fixed-size chunks."""
        text = document.content
        
        if not text.strip():
            return
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Skip empty chunks
            if chunk_text.strip():
                yield Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunk_start": start,
                        "chunk_end": min(end, len(text)),
                        "chunker": "fixed",
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    }
                )
                chunk_index += 1
            
            # Move to next position (with overlap)
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start <= 0 and end >= len(text):
                break
