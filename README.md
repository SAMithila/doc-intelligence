# Document Intelligence Platform

A production-grade RAG (Retrieval Augmented Generation) system built with experiment-driven development.

## 🎯 Project Philosophy

This isn't just another RAG tutorial. This project demonstrates **engineering judgment** through:

- **Measured improvements**: Every feature is evaluated against baseline metrics
- **Documented trade-offs**: Design decisions explained with data
- **Intentional removals**: Features that didn't help were removed and documented
- **Production practices**: Proper configuration, testing, and observability

## 📊 Performance Journey

| Version | Change | Recall@5 | MRR | Latency | Decision |
|---------|--------|----------|-----|---------|----------|
| v0.1 | Baseline (fixed chunking, semantic search) | TBD | TBD | TBD | Baseline |

*Table updated as experiments are completed*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline v0.1                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INGEST          RETRIEVE         GENERATE                  │
│  ┌─────────┐    ┌─────────┐      ┌─────────┐               │
│  │ Loader  │ →  │ Embed   │  →   │ Context │               │
│  │ Chunker │    │ Search  │      │ + LLM   │               │
│  │ Embed   │    │         │      │         │               │
│  │ Store   │    │         │      │         │               │
│  └─────────┘    └─────────┘      └─────────┘               │
│       │              │                                      │
│       └──────────────┘                                      │
│              │                                              │
│       ┌──────▼──────┐                                       │
│       │  ChromaDB   │                                       │
│       └─────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/doc-intelligence.git
cd doc-intelligence
pip install -e .

# Set API key
export OPENAI_API_KEY=your_key

# Run test
python -m tests.test_baseline
```

## 📁 Project Structure

```
doc-intelligence/
├── src/docint/
│   ├── config.py           # Typed configuration
│   ├── pipeline.py         # Main RAG pipeline
│   ├── ingest/             # Document loading & chunking
│   ├── embeddings/         # Embedding providers
│   ├── store/              # Vector store implementations
│   ├── retrieval/          # Retrieval strategies
│   ├── generation/         # LLM generation
│   └── evaluation/         # Metrics & evaluation
├── experiments/            # Documented experiments
│   └── 001_baseline/       # Baseline metrics
├── configs/                # YAML configurations
├── eval_data/              # Test documents & Q&A pairs
└── tests/                  # Test suite
```

## 🔬 Experiments

Each experiment follows a rigorous process:
1. **Hypothesis**: What we expect to improve
2. **Implementation**: Code changes
3. **Measurement**: Metrics before/after
4. **Decision**: Keep, modify, or remove

See [`experiments/`](experiments/) for full documentation.

## 📈 Evaluation Metrics

### Retrieval Metrics
- **Recall@K**: Did we find all relevant documents?
- **Precision@K**: How much noise in results?
- **MRR**: Where does first relevant doc appear?
- **NDCG@K**: How well are relevant docs ranked?

### Generation Metrics (planned)
- **Faithfulness**: Is answer grounded in context?
- **Relevance**: Does answer address the question?

## 🛠️ Configuration

```yaml
# configs/default.yaml
chunking:
  strategy: fixed      # fixed | recursive | semantic
  chunk_size: 512
  chunk_overlap: 50

embedding:
  model: text-embedding-3-small

retrieval:
  top_k: 5

generation:
  model: gpt-4o-mini
```

## 📝 Design Decisions

### Why Fixed Chunking for Baseline?
Simple and predictable. Establishes clear baseline before trying complex strategies.
We'll compare against recursive and semantic chunking in Experiment 002.

### Why ChromaDB?
- Embedded (no separate server)
- Good for development and small-to-medium scale
- Easy to swap out later if needed

### Why OpenAI Embeddings?
- High quality out of the box
- Consistent API
- Will compare against local embeddings for cost/latency trade-off

## 🎯 Roadmap

- [x] v0.1: Baseline RAG pipeline
- [ ] v0.2: Chunking strategy comparison
- [ ] v0.3: Hybrid search (BM25 + semantic)
- [ ] v0.4: Cross-encoder reranking
- [ ] v0.5: HyDE query expansion
- [ ] v0.6: Hallucination detection
- [ ] v1.0: Production-ready with full evaluation

## 📄 License

MIT
