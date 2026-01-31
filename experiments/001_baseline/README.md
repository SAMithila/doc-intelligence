# Experiment 001: Baseline Metrics

## Objective

Establish baseline performance metrics for the simplest RAG implementation. 
This provides a reference point for measuring improvements from future enhancements.

## Baseline Configuration

```yaml
chunking:
  strategy: fixed
  chunk_size: 512
  chunk_overlap: 50

embedding:
  model: text-embedding-3-small

retrieval:
  type: simple_semantic
  top_k: 5

generation:
  model: gpt-4o-mini
```

## Methodology

### Test Corpus
- 2 documents (TechCorp annual report, CloudScale documentation)
- ~4,500 words total
- Mix of financial data and technical documentation

### Evaluation Approach
For baseline, we use manual evaluation with 10 test questions:
- 5 factual questions with single correct answers
- 3 questions requiring synthesis across chunks
- 2 questions where answer is NOT in documents (tests hallucination)

## Results

*To be filled after running evaluation*

| Metric | Value | Notes |
|--------|-------|-------|
| Recall@5 | TBD | Fraction of relevant chunks retrieved |
| Precision@5 | TBD | Fraction of retrieved chunks that are relevant |
| MRR | TBD | Where does first relevant chunk appear |
| Avg Latency | TBD | End-to-end query time |
| Retrieval Latency | TBD | Time for embedding + search |
| Generation Latency | TBD | Time for LLM response |

### Qualitative Observations

*To be filled after testing*

- Chunking issues observed:
- Retrieval failures:
- Generation quality notes:

## Known Limitations

1. **Fixed chunking splits content arbitrarily**
   - Tables may be split mid-row
   - Sentences cut in middle
   - No awareness of document structure

2. **No keyword matching**
   - Pure semantic search
   - May miss exact term matches

3. **No reranking**
   - Relies entirely on embedding similarity
   - No second-pass relevance scoring

## Next Steps

Based on baseline results, prioritize experiments:
1. [ ] Experiment 002: Compare chunking strategies
2. [ ] Experiment 003: Add BM25 hybrid search
3. [ ] Experiment 004: Test reranking impact

## Reproduction

```bash
# Set up environment
cd doc-intelligence
pip install -e .
export OPENAI_API_KEY=your_key

# Run baseline test
python -m tests.test_baseline
```
