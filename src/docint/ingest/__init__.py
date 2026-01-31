"""Document ingestion components."""
from docint.ingest.loaders import TextLoader, Document
from docint.ingest.chunkers import FixedChunker, Chunk

__all__ = ["TextLoader", "Document", "FixedChunker", "Chunk"]
