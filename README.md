# Doc Intelligence

A production-oriented RAG system built with experiment-driven development. Focus on measured improvements, documented trade-offs, and engineering judgment.

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 83.3% (25/30 queries) |
| **Indexing time** | 2.67s |
| **Query latency (P50)** | 1,824ms |
| **Query latency (P95)** | 6,282ms |
| **Retrieval (mean)** | 269ms |
| **Generation (mean)** | 1,733ms |

Bottleneck is LLM generation (~1.7s). Retrieval is fast (~270ms).

## What I Built & Learned

### Chunking Strategy
Fixed-size chunking broke queries by splitting headers from content. A chunk literally started with "ervices" (mid-word cut). Recursive chunking respects paragraph boundaries and fixed this.

### Hybrid Retrieval
Combined BM25 keyword search with semantic search using Reciprocal Rank Fusion. Result: +6% accuracy on hard queries. Minimal latency overhead.

### Query Expansion (HyDE)
Short queries like "Q3 revenue" improved from 50% → 70% accuracy. Adds 1.5s latency, so it's conditional—only triggers for queries ≤3 words.

### Reranking (Rejected)
Built LLM-based reranker. Moved correct chunk to #1. Then measured: **5 seconds per query**. Killed it. Not worth the latency.

### Hallucination Detection
Verifies answers are grounded in retrieved context. Catches when LLM makes up facts not in the source documents.

## Accuracy by Query Type

| Approach | Overall | Easy | Vague | Hard |
|----------|---------|------|-------|------|
| Semantic only | 77% | 90% | 50% | 90% |
| + Hybrid | 83% | 90% | 60% | 100% |
| + HyDE (conditional) | 83% | 90% | 70% | 90% |

Evaluation uses 30 labeled queries: 10 easy, 10 vague, 10 hard/multi-hop.

## Architecture
```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           FastAPI / Streamlit           │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌─────────┐
│  HyDE  │  │ Semantic │  │  BM25   │
│(if short)│ │  Search  │  │ Search  │
└────┬───┘  └────┬─────┘  └────┬────┘
     └───────────┼─────────────┘
                 ▼
         ┌─────────────┐
         │ Rank Fusion │
         └──────┬──────┘
                ▼
         ┌─────────────┐
         │  Generator  │
         │ (GPT-4o-mini)│
         └──────┬──────┘
                ▼
         ┌─────────────┐
         │ Groundedness│
         │   Check     │
         └──────┬──────┘
                ▼
            Response
```

## Scaling Strategy

### Current Scale
- 2 documents, 17 chunks
- Single-node ChromaDB
- Synchronous embedding

### 10K Documents
- ChromaDB handles this fine
- Batch embedding calls
- Add query result caching

### 100K Documents
- Switch to FAISS with IVF index
- Async embedding pipeline
- Redis cache for frequent queries

### 1M+ Documents
- Shard vector index by document type/date
- Distributed embedding workers
- Consider managed service (Pinecone/Weaviate)
- Approximate nearest neighbor (ANN)

### High Traffic (100+ QPS)
- Horizontal API scaling
- GPU for local embeddings (cost trade-off)
- Response caching with TTL
- Rate limiting per client

## Failure Handling

| Failure | Impact | Mitigation |
|---------|--------|------------|
| OpenAI API timeout | Query fails | Retry with backoff, fallback to cached |
| ChromaDB corruption | Index lost | Periodic snapshots, rebuild from source |
| Empty retrieval | Bad answer | Return "not found" instead of hallucinating |
| Embedding drift | Quality degradation | Version embeddings, reindex on model change |

## Quick Start
```bash
git clone https://github.com/SAMithila/doc-intelligence.git
cd doc-intelligence
pip install -e .

export OPENAI_API_KEY=your_key

# Run evaluation
python tests/test_evaluation.py

# Run benchmarks
python benchmarks/benchmark_retrieval.py

# Start UI
streamlit run app/streamlit_app.py

# Start API
python run_api.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + stats |
| `/query` | POST | Ask a question |
| `/docs` | GET | Swagger documentation |
```bash
# Example query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Q3 revenue?", "verify_groundedness": true}'
```

## Project Structure
```
src/docint/
├── api/           # FastAPI endpoints
├── ingest/        # Document loading, chunking
├── retrieval/     # Semantic, BM25, hybrid, HyDE
├── generation/    # LLM response generation
├── verification/  # Hallucination detection
└── evaluation/    # Metrics, test framework

experiments/       # Documented experiments with decisions
docs/              # Architecture decisions, system design
benchmarks/        # Performance measurement
docker/            # Deployment configuration
```

## Experiments

| # | Experiment | Result | Decision |
|---|------------|--------|----------|
| 001 | Baseline | 80% accuracy, Q3 failed | Found chunking bug |
| 002 | Recursive chunking | Fixed retrieval | ✅ Keep |
| 003 | Hybrid search | +6% accuracy | ✅ Keep |
| 004 | Reranking | +5s latency | ❌ Rejected |
| 005 | HyDE | +20% on vague | ✅ Conditional |
| 006 | Evaluation framework | 30-query test set | ✅ Essential |
| 007 | Hallucination detection | Catches fabrications | ✅ Keep |

See `docs/architecture_decisions.md` for detailed reasoning.

## Testing
```bash
# Unit tests (21 tests)
pytest tests/test_chunker.py tests/test_retriever.py -v

# Full evaluation (30 queries)
python tests/test_evaluation.py

# Hallucination detection
python tests/test_hallucination.py
```

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector Store | ChromaDB | Local persistence, simple API, sufficient for <100K docs |
| Chunk Size | 512 chars | Tested 256/512/1024. 512 keeps sections intact without noise |
| Embedding | text-embedding-3-small | Good quality, reasonable cost. Would test local models for scale |
| LLM | gpt-4o-mini | Fast, cheap, good enough. Would use gpt-4o for complex queries |

## What I'd Do Differently

1. **Start with hard test cases** — Easy queries always pass, teach nothing
2. **Measure latency from day 1** — Would've caught reranking issue earlier  
3. **Build evaluation first** — Before adding features
   
