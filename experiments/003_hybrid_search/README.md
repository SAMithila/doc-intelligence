# Experiment 003: Hybrid Search (BM25 + Semantic)

## Objective

Test if combining BM25 keyword search with semantic search improves retrieval quality.

## Hypothesis

Pure semantic search may miss exact keyword matches. Adding BM25 will:
1. Catch exact term matches that semantic search misses
2. Improve ranking for queries with specific keywords

## Implementation

### BM25 Index
- Standard BM25 with k1=1.5, b=0.75
- Simple whitespace tokenization
- Inverted index for fast lookup

### Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = Σ 1/(k + rank(d))
```
- k = 60 (default)
- Combines semantic and BM25 rankings
- 50/50 weight between both methods

## Results

### Accuracy Comparison

| Method | Accuracy |
|--------|----------|
| Semantic only | 5/5 (100%) |
| Hybrid (BM25 + Semantic) | 5/5 (100%) |

### Ranking Comparison

Query: "What was TechCorp Q3 2024 revenue?"

| Rank | Semantic | BM25 | Content |
|------|----------|------|---------|
| 1 | 2 | **1** | Q3 2024 section (correct answer) |
| 2 | 1 | 2 | Executive Summary |
| 3 | 4 | 3 | Q1 section |
| 4 | 3 | 6 | Regional performance |
| 5 | 6 | 4 | Q2 section |

**Key insight:** BM25 ranked the Q3 chunk #1 due to exact keyword match "Q3 2024 revenue".

## Decision

✅ **KEEP hybrid search as an option**

**Rationale:**
- Equal accuracy on current test set
- Better ranking for keyword-specific queries
- BM25 adds minimal latency (in-memory index)
- Provides fallback when semantic search fails

**Trade-offs:**
- More complexity (two indexes to maintain)
- Slightly higher memory usage
- May not help for all query types

## When to Use

| Query Type | Best Approach |
|------------|---------------|
| Exact terms (e.g., "SOC 2") | Hybrid |
| Natural questions | Semantic |
| Technical queries | Hybrid |
| Exploratory queries | Semantic |

## Files Added

- `src/docint/retrieval/bm25.py` - BM25 index implementation
- `src/docint/retrieval/hybrid.py` - Hybrid retriever with RRF
- `tests/test_hybridsearch.py` - Comparison test

## Next Steps

1. [x] Experiment 003: Hybrid search ✅
2. [ ] Experiment 004: Cross-encoder reranking
3. [ ] Experiment 005: Query expansion (HyDE)