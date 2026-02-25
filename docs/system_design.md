# System Design

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI / Streamlit                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  HyDE    │   │ Semantic │   │   BM25   │
    │(optional)│   │  Search  │   │  Search  │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Rank Fusion    │
              │  (RRF)          │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Generator     │
              │   (GPT-4o-mini) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Groundedness   │
              │  Check (opt)    │
              └────────┬────────┘
                       │
                       ▼
                   Response
```

## Data Flow

1. **Ingestion**: Documents → Chunker → Embedder → ChromaDB
2. **Query**: User query → (HyDE) → Embed → Search → Rank → Generate → Verify

## Current Scale

| Metric | Value |
|--------|-------|
| Documents | 2 |
| Chunks | 17 |
| Indexing time | ~2s |
| Query latency P50 | ~2000ms |
| Query latency P95 | ~2500ms |

## Bottlenecks

1. **Generation** (~1800ms) - LLM API call is the slowest step
2. **Embedding** (~300ms) - Could batch for bulk ingestion
3. **Retrieval** (~200ms) - Fast enough for current scale

## Scaling Strategy

### 10K Documents
- Current architecture works fine
- ChromaDB handles this scale

### 100K Documents
- Switch to FAISS with IVF index
- Add embedding cache
- Batch embedding calls

### 1M+ Documents
- Shard vector index by document type/date
- Use approximate nearest neighbor (ANN)
- Add Redis cache for frequent queries
- Consider managed service (Pinecone)

### High Traffic (100+ QPS)
- Horizontal scaling of API servers
- GPU for local embeddings
- Async embedding pipeline
- Query result caching

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|------------|
| OpenAI API down | No queries work | Fallback to cached answers |
| ChromaDB corruption | Index lost | Periodic backups |
| Bad chunks | Poor retrieval | Validation during ingestion |
| Hallucination | Wrong answers | Groundedness checking |

## Future Improvements

- [ ] Streaming responses
- [ ] Embedding versioning
- [ ] A/B testing retrieval strategies
- [ ] Query analytics dashboard