"""
FastAPI REST API for the RAG system.

Run: uvicorn docint.api.main:app --reload
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from docint.config import load_config
from docint.pipeline import RAGPipeline
from docint.verification.groundedness import GroundednessChecker


# Global pipeline instance
pipeline = None
groundedness_checker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline, groundedness_checker

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    config = load_config('configs/default.yaml')
    config.openai_api_key = api_key

    pipeline = RAGPipeline(config)
    pipeline.ingest_directory('eval_data/documents')

    groundedness_checker = GroundednessChecker(api_key=api_key)

    print("✓ Pipeline initialized")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Doc Intelligence API",
    description="RAG system for document Q&A",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    verify_groundedness: bool = False


class SourceChunk(BaseModel):
    content: str
    score: float
    filename: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_time_ms: float
    generation_time_ms: float
    is_grounded: bool | None = None
    groundedness_confidence: float | None = None


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int


# Endpoints
@app.get("/", response_model=dict)
async def root():
    """API info."""
    return {
        "name": "Doc Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check API health and stats."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_stats()
    return HealthResponse(
        status="healthy",
        documents_loaded=stats.get("document_count", 0),
        chunks_indexed=stats.get("chunk_count", 0),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.

    - **question**: Your question about the documents
    - **verify_groundedness**: Check if answer is grounded (adds latency)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Run query
    result = pipeline.query(request.question)

    # Build sources
    sources = [
        SourceChunk(
            content=r.content[:200] +
            "..." if len(r.content) > 200 else r.content,
            score=r.score,
            filename=r.metadata.get("filename"),
        )
        for r in result.retrieval.results[:3]
    ]

    # Optional groundedness check
    is_grounded = None
    confidence = None

    if request.verify_groundedness and groundedness_checker:
        check = groundedness_checker.check(
            question=request.question,
            answer=result.answer,
            context_chunks=result.retrieval.contexts,
        )
        is_grounded = check.is_grounded
        confidence = check.confidence

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        retrieval_time_ms=result.latency_ms.get("retrieval_ms", 0),
        generation_time_ms=result.latency_ms.get("generation_ms", 0),
        is_grounded=is_grounded,
        groundedness_confidence=confidence,
    )


@app.get("/stats")
async def stats():
    """Get pipeline statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.get_stats()
