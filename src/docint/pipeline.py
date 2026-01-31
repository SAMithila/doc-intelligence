"""
End-to-end RAG pipeline.

Orchestrates ingestion, retrieval, and generation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import time

from docint.config import Config
from docint.ingest.loaders import TextLoader, Document
from docint.ingest.chunkers import FixedChunker, Chunk
from docint.embeddings.openai import OpenAIEmbedder
from docint.store.chroma import ChromaStore
from docint.retrieval.retriever import SimpleRetriever, RetrievalResult
from docint.generation.generator import Generator, GenerationResult


@dataclass 
class QueryResult:
    """Complete result of a RAG query."""
    question: str
    answer: str
    retrieval: RetrievalResult
    generation: GenerationResult
    latency_ms: dict = field(default_factory=dict)
    
    @property
    def sources(self) -> list[dict]:
        """Get source documents with metadata."""
        return [
            {
                "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in self.retrieval.results
        ]


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    
    Baseline v0.1:
    - Fixed chunking
    - OpenAI embeddings
    - ChromaDB vector store
    - Simple semantic retrieval
    - OpenAI generation
    """
    
    VERSION = "0.1.0"
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self._init_components()
        
        # Track ingested documents
        self._chunk_count = 0
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Loader (only text for baseline)
        self.loader = TextLoader()
        
        # Chunker
        self.chunker = FixedChunker(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        
        # Embedder
        self.embedder = OpenAIEmbedder(
            api_key=self.config.openai_api_key,
            model=self.config.embedding.model,
        )
        
        # Vector store
        self.vector_store = ChromaStore(
            collection_name=self.config.vector_store.collection_name,
            persist_directory=self.config.vector_store.persist_directory,
        )
        
        # Retriever
        self.retriever = SimpleRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.config.retrieval.top_k,
        )
        
        # Generator
        self.generator = Generator(
            api_key=self.config.openai_api_key,
            model=self.config.generation.model,
            temperature=self.config.generation.temperature,
            max_tokens=self.config.generation.max_tokens,
        )
    
    def ingest_document(self, path: str | Path) -> int:
        """
        Ingest a single document.
        
        Returns number of chunks created.
        """
        # Load
        document = self.loader.load(path)
        
        # Chunk
        chunks = list(self.chunker.chunk(document))
        
        if not chunks:
            return 0
        
        # Embed
        contents = [c.content for c in chunks]
        embeddings = self.embedder.embed(contents)
        
        # Store
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        self.vector_store.add(
            ids=ids,
            embeddings=embeddings.embeddings,
            contents=contents,
            metadatas=metadatas,
        )
        
        self._chunk_count += len(chunks)
        return len(chunks)
    
    def ingest_directory(
        self, 
        path: str | Path,
        pattern: str = "*",
    ) -> dict:
        """
        Ingest all documents from directory.
        
        Returns stats about ingestion.
        """
        path = Path(path)
        stats = {
            "documents": 0,
            "chunks": 0,
            "errors": [],
        }
        
        for doc in self.loader.load_directory(path, pattern):
            try:
                chunks = list(self.chunker.chunk(doc))
                
                if chunks:
                    contents = [c.content for c in chunks]
                    embeddings = self.embedder.embed(contents)
                    
                    self.vector_store.add(
                        ids=[c.chunk_id for c in chunks],
                        embeddings=embeddings.embeddings,
                        contents=contents,
                        metadatas=[c.metadata for c in chunks],
                    )
                    
                    stats["documents"] += 1
                    stats["chunks"] += len(chunks)
                    
            except Exception as e:
                stats["errors"].append({
                    "source": doc.metadata.get("source", "unknown"),
                    "error": str(e),
                })
        
        self._chunk_count = self.vector_store.count()
        return stats
    
    def query(
        self,
        question: str,
        top_k: int | None = None,
        include_citations: bool = False,
    ) -> QueryResult:
        """
        Run RAG query.
        
        Args:
            question: User question
            top_k: Override default number of chunks to retrieve
            include_citations: Ask LLM to include citations
            
        Returns:
            QueryResult with answer and metadata
        """
        latency = {}
        
        # Retrieve
        start = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )
        latency["retrieval_ms"] = (time.perf_counter() - start) * 1000
        
        # Generate
        start = time.perf_counter()
        generation = self.generator.generate(
            question=question,
            context_chunks=retrieval.contexts,
            include_citations=include_citations,
        )
        latency["generation_ms"] = (time.perf_counter() - start) * 1000
        
        latency["total_ms"] = latency["retrieval_ms"] + latency["generation_ms"]
        
        return QueryResult(
            question=question,
            answer=generation.answer,
            retrieval=retrieval,
            generation=generation,
            latency_ms=latency,
        )
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "version": self.VERSION,
            "chunk_count": self.vector_store.count(),
            "chunker": {
                "strategy": "fixed",
                "chunk_size": self.config.chunking.chunk_size,
                "chunk_overlap": self.config.chunking.chunk_overlap,
            },
            "embedder": {
                "model": self.config.embedding.model,
                "dimension": self.embedder.dimension,
            },
            "retriever": {
                "type": "simple",
                "top_k": self.config.retrieval.top_k,
            },
            "generator": {
                "model": self.config.generation.model,
            },
        }
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.vector_store.clear()
        self._chunk_count = 0
