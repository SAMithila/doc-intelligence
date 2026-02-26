# Doc Intelligence

RAG system for document Q&A. Built this to learn what actually works in retrieval systems vs. what's just hype.

## What I Learned

- **Chunking matters more than I expected.** Fixed-size chunking broke my Q3 revenue query by splitting "## Q3 2024" from "$1.15 billion". Spent 2 hours debugging before I realized the chunk literally started with "ervices" (cut mid-word). Recursive chunking fixed it.

- **Hybrid search helps, but not dramatically.** Added BM25 keyword matching alongside semantic search. Got +6% accuracy. Worth keeping for the minimal overhead.

- **Reranking is too slow.** Built an LLM-based reranker, got excited when it moved the correct chunk to #1. Then measured: 5 seconds per query. Killed it.

- **HyDE helps vague queries.** Short queries like "Q3 revenue" went from 50% to 70% accuracy with query expansion. But adds 1.5s latency, so I only use it conditionally.

- **Easy tests are worthless.** For 5 days I kept getting 100% accuracy and feeling good. Then I built proper evaluation with vague and multi-hop queries. Real accuracy was 77%. Always test on hard cases.

## Results

Tested on 30 queries (easy, vague, hard):

| Approach | Overall | Easy | Vague | Hard |
|----------|---------|------|-------|------|
| Semantic only | 77% | 90% | 50% | 90% |
| + Hybrid search | 83% | 90% | 60% | 100% |
| + HyDE (vague only) | 83% | 90% | 70% | 90% |

## Performance

| Metric | Value |
|--------|-------|
| Indexing time | 2.1s |
| Query latency (P50) | 1,850ms |
| Query latency (P95) | 2,400ms |
| Retrieval accuracy | 83% |

Bottleneck is LLM generation (~1,800ms). Retrieval is fast (~300ms).

## Quick Start
```bash
git clone https://github.com/SAMithila/doc-intelligence.git
cd doc-intelligence
pip install -e .

export OPENAI_API_KEY=your_key

# Run evaluation
python tests/test_evaluation.py

# Or try the UI
streamlit run app/streamlit_app.py
```

## Features

- **Hybrid retrieval** - BM25 + semantic search with reciprocal rank fusion
- **Query expansion** - HyDE for vague queries
- **Hallucination detection** - Verifies answers are grounded in context
- **FastAPI + Streamlit** - REST API and demo UI

## Project Structure
```
src/docint/
├── ingest/        # Chunking (fixed, recursive)
├── retrieval/     # Semantic, BM25, hybrid, reranking, HyDE
├── generation/    # LLM answer generation
├── verification/  # Hallucination detection
├── evaluation/    # Metrics and testing
└── api/           # FastAPI endpoints

experiments/       # What I tried, what worked, what didn't
app/               # Streamlit demo
```

## Experiments

Each folder in `experiments/` documents what I tried:

| Experiment | Result | Decision |
|------------|--------|----------|
| 001_baseline | 80% accuracy, Q3 query failed | Found chunking bug |
| 002_chunking | Recursive fixed the failure | Keep |
| 003_hybrid | +6% accuracy | Keep |
| 004_reranking | +5s latency, no accuracy gain | Rejected |
| 005_hyde | +20% on vague queries | Keep (conditional) |
| 006_evaluation | Built proper test framework | Essential |
| 007_hallucination | Catches made-up facts | Keep |

See `docs/architecture_decisions.md` for why I made specific choices (ChromaDB vs FAISS, chunk size, etc.)

## What I'd Do Differently

1. **Start with hard test cases** - Not easy ones that always pass
2. **Measure latency from day 1** - Would've caught reranking issue earlier
3. **Build evaluation first** - Before adding features

## Screenshots

FastAPI docs at `/docs`:
- Query endpoint with groundedness verification
- Health check and stats

Streamlit UI:
- Ask questions, see sources
- Groundedness confidence score
- Latency metrics
