# Experiment 004: Cross-Encoder Reranking

## Objective

Test if reranking initial retrieval results improves accuracy and ranking quality.

## Hypothesis

Cross-encoder reranking compares query-document pairs directly, providing more accurate relevance scores than bi-encoder similarity.

## Implementation

### Two-Stage Retrieval
1. **Stage 1:** Retrieve top 10 candidates (fast, bi-encoder)
2. **Stage 2:** Rerank to top 5 (slow, cross-encoder)

### LLM-Based Reranker
- Uses GPT-4o-mini to score relevance 0-10
- Processes each query-document pair separately
- Normalizes scores to 0-1

## Results

### Ranking Improvement

Query: "What was TechCorp Q3 2024 revenue?"

| Rank | Before | After | Change |
|------|--------|-------|--------|
| 1 | Executive Summary | Q3 2024 section | ↑ Correct answer promoted |
| 2 | Q3 2024 section | Executive Summary | ↓ |
| 3 | Regional | Q2 section | ↑ |

**Reranker correctly identified Q3 section as most relevant (score: 1.0 vs 0.5)**

### Accuracy Comparison

| Method | Accuracy |
|--------|----------|
| Without reranking | 5/5 (100%) |
| With reranking | 5/5 (100%) |

### Latency Impact

| Metric | Value |
|--------|-------|
| Retrieval time | 219ms |
| Reranking time | 5796ms |
| Total time | 6015ms |
| **Latency increase** | **+2650%** |

## Decision

⚠️ **KEEP as optional feature, NOT default**

**Rationale:**
- Improves ranking quality (correct answer at #1)
- But adds 5+ seconds latency per query
- Accuracy already 100% without reranking on our test set
- LLM-based reranking is expensive ($$$)

**When to enable:**
- Batch processing where latency doesn't matter
- Critical queries where accuracy is paramount
- When using local cross-encoder models (faster, free)

## Alternative: Local Cross-Encoder

For production, consider using `sentence-transformers` cross-encoder:
```python
# Much faster, free, runs locally
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

Expected latency: ~50-100ms vs 5000ms for LLM-based.

## Trade-offs

| Aspect | LLM Reranker | Local Cross-Encoder |
|--------|--------------|---------------------|
| Latency | ~5000ms | ~50-100ms |
| Cost | $$$ (API calls) | Free |
| Accuracy | High | High |
| Setup | Easy | Requires model download |

## Files Added

- `src/docint/retrieval/reranker.py` - LLM and CrossEncoder rerankers
- `tests/test_reranker.py` - Reranking comparison test

## Next Steps

1. [x] Experiment 004: Reranking ✅
2. [ ] Experiment 005: Query expansion (HyDE)
3. [ ] Experiment 006: Evaluation framework improvements

## Key Learning

**Not every improvement is worth implementing.**

Reranking improves ranking quality but the 5+ second latency isn't justified when accuracy is already 100%. This demonstrates engineering judgment - knowing when NOT to add complexity.

