"""
Document chunking strategies.

Day 1 Baseline: Fixed-size chunking with overlap.
Day 2 Addition: Recursive chunking that respects document structure.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
import hashlib
import re

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


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking that respects document structure.

    Splits text hierarchically:
    1. First try to split on double newlines (paragraphs/sections)
    2. If still too big, split on single newlines
    3. If still too big, split on sentences (. ! ?)
    4. Last resort: split on spaces (words)

    Pros:
    - Respects document structure
    - Keeps related content together
    - Doesn't split mid-sentence (usually)

    Cons:
    - Variable chunk sizes
    - More complex logic
    - May create very small chunks
    """

    # Separators in order of preference (most structure to least)
    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraphs / sections
        "\n",        # Lines
        ". ",        # Sentences
        "? ",        # Questions
        "! ",        # Exclamations
        "; ",        # Clauses
        ", ",        # Phrases
        " ",         # Words
        "",          # Characters (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """
        Recursively split text using separators.

        Try each separator in order until chunks are small enough.
        """
        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # No more separators - force split
        if not separators:
            # Split at chunk_size as last resort
            chunks = []
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means character-by-character
            splits = list(text)

        # If separator didn't help, try next one
        if len(splits) == 1:
            return self._split_text(text, remaining_separators)

        # Merge splits into chunks of appropriate size
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            piece = split + (separator if i < len(splits) - 1 else "")

            # If adding this piece would exceed chunk size
            if current_chunk and len(current_chunk) + len(piece) > self.chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = ""

            # If single piece is too big, recursively split it
            if len(piece) > self.chunk_size:
                # Save any accumulated content first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Recursively split the large piece
                sub_chunks = self._split_text(piece, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                current_chunk += piece

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between chunks for context continuity."""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no prefix, add suffix from next
                suffix = chunks[i + 1][:self.chunk_overlap] if i + \
                    1 < len(chunks) else ""
                overlapped.append(chunk + (" " + suffix if suffix else ""))
            elif i == len(chunks) - 1:
                # Last chunk: add prefix from previous, no suffix
                prefix = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append((prefix + " " if prefix else "") + chunk)
            else:
                # Middle chunks: add both prefix and suffix
                prefix = chunks[i - 1][-self.chunk_overlap:]
                suffix = chunks[i + 1][:self.chunk_overlap]
                new_chunk = (prefix + " " if prefix else "") + \
                    chunk + (" " + suffix if suffix else "")
                overlapped.append(new_chunk)

        return overlapped

    def chunk(self, document: Document) -> Iterator[Chunk]:
        """Split document into chunks respecting structure."""
        text = document.content

        if not text.strip():
            return

        # Split recursively
        chunks = self._split_text(text, self.separators)

        # Add overlap for context
        # Note: Disabling overlap for now to keep chunks cleaner
        # chunks = self._add_overlap(chunks)

        # Yield chunks with metadata
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                yield Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "chunker": "recursive",
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    }
                )


# Factory function to create chunker by name
def create_chunker(
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> BaseChunker:
    """
    Create a chunker by strategy name.

    Args:
        strategy: "fixed" or "recursive"
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        Configured chunker instance
    """
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
