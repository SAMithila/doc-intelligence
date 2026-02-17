# Experiment 005: HyDE Query Expansion

## Objective

Test if HyDE (Hypothetical Document Embeddings) improves retrieval for vague or short queries.

## Hypothesis

Short queries like "Q3 revenue" may not embed well to match detailed documents. Generating a hypothetical answer first creates a richer embedding that better matches real documents.

## Implementation

### HyDE Process
1. Take user query: "Q3 revenue"
2. LLM generates hypothetical answer:
   "The Q3 revenue for the fiscal year totaled $2.5 billion..."
3. Embed the hypothetical document (not the query)
4. Search for real documents similar to the hypothetical

### Multi-Query Expansion
Also tested generating query variations:
- "Q3 revenue"
- "What is the revenue for the third quarter?"
- "Can you provide the Q3 revenue figures?"
- "How much revenue was generated in the third quarter?"

## Results

### Retrieval Score Improvement

Query: "Q3 revenue" (vague)

| Method | Q3 Chunk Score | Improvement |
|--------|----------------|-------------|
| Normal | 0.619 | - |
| HyDE | 0.668 | **+8%** |

### Accuracy Comparison

| Query Type | Normal | HyDE |
|------------|--------|------|
| Specific queries | 3/3 ✅ | 3/3 ✅ |
| Vague queries | 3/3 ✅ | 3/3 ✅ |
| **Total** | 6/6 (100%) | 6/6 (100%) |

### Latency Impact

| Metric | Value |
|--------|-------|
| Normal search | ~200ms |
| HyDE expansion | ~1500ms |
| **Additional latency** | **+1500ms** |

## Decision

⚠️ **KEEP as optional feature for vague queries**

**Rationale:**
- Improves retrieval scores (+8% for vague queries)
- But adds 1.5 seconds latency
- Accuracy already 100% without HyDE on our test set
- Most helpful when retrieval is failing

**When to enable:**
- Query is very short (1-3 words)
- Initial retrieval scores are low (<0.5)
- User feedback indicates poor results

**When to skip:**
- Query is already detailed
- Retrieval scores are high
- Latency is critical

## Implementation Note

HyDE works best with a query classifier:
```python
def should_use_hyde(query: str, initial_scores: list[float]) -> bool:
    # Short query
    if len(query.split()) <= 3:
        return True
    # Low retrieval scores
    if max(initial_scores) < 0.5:
        return True
    return False
```

## Files Added

- `src/docint/retrieval/hyde.py` - HyDE and MultiQuery expanders
- `tests/test_hyde.py` - Expansion comparison test

## Next Steps

1. [x] Experiment 005: HyDE query expansion ✅
2. [ ] Experiment 006: Hallucination detection
3. [ ] Build evaluation dataset with ground truth

## Key Learning

**HyDE improves retrieval quality but at a latency cost.**

The decision to use HyDE should be dynamic - enable it for vague queries or when initial retrieval looks poor. This conditional approach balances quality and speed.

Talking point:
> "I implemented HyDE for query expansion and found it improved retrieval scores by 8% for vague queries. However, it adds 1.5 seconds latency, so I made it conditional - only triggered for short queries or low-confidence retrievals."