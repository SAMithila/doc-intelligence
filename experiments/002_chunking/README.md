# Experiment 002: Chunking Strategy Comparison

## Objective

Fix the Q3 revenue retrieval failure from baseline by implementing recursive chunking that respects document structure.

## Hypothesis

Fixed chunking at 512 characters splits sections arbitrarily, separating headers from content. Recursive chunking that splits on paragraph/sentence boundaries will keep related content together and improve retrieval accuracy.

## Configuration

### Baseline (Fixed Chunking)
```yaml
chunking:
  strategy: fixed
  chunk_size: 512
  chunk_overlap: 50
```

### Experiment (Recursive Chunking)
```yaml
chunking:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 50
  separators: ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
```

## Results

### Accuracy

| Query | Fixed | Recursive |
|-------|-------|-----------|
| Q3 2024 revenue | ❌ Failed | ✅ $1.15 billion |
| CloudScale storage cost | ✅ | ✅ |
| Employee count | ✅ | ✅ |
| Security certifications | ✅ | ✅ |
| SecureNet acquisition | ✅ | ✅ |
| **Total** | **4/5 (80%)** | **5/5 (100%)** |

### Performance

| Metric | Fixed | Recursive | Change |
|--------|-------|-----------|--------|
| Chunks created | 15 | 17 | +13% |
| Avg latency | 2450ms | 1165ms | -52% |
| Retrieval latency | ~530ms | ~400ms | -25% |

### Chunk Quality Analysis

**Fixed Chunking Problem:**
```
Chunk 3: "ervices revenue hit $580 million..."
         ↑ Cut mid-word! Header separated from data.
```

**Recursive Chunking Solution:**
```
Chunk 4: "Q3 2024 was our strongest quarter, with revenue reaching $1.15 billion..."
         ↑ Complete section with header and data together.
```

## Root Cause Confirmed

The Q3 revenue query failed in baseline because:
1. Fixed chunking split at character 512 regardless of content
2. The section header "## Q3 2024 Performance" was in one chunk
3. The revenue figure "$1.15 billion" was in another chunk
4. Semantic search couldn't connect "Q3 revenue" query to the split content

Recursive chunking splits on `\n\n` (paragraph boundaries) first, keeping each section intact.

## Decision

✅ **KEEP recursive chunking as default**

**Justification:**
- Fixed critical retrieval failure (+20% accuracy)
- Improved latency (-52%)
- Minimal overhead (+2 chunks)
- Better chunk quality (no mid-word splits)

## Trade-offs

| Aspect | Fixed | Recursive |
|--------|-------|-----------|
| Simplicity | ✅ Simpler | More complex |
| Predictability | ✅ Consistent sizes | Variable sizes |
| Quality | Poor boundaries | ✅ Respects structure |
| Accuracy | 80% | ✅ 100% |

## Next Steps

1. [x] Experiment 002: Recursive chunking ✅
2. [ ] Experiment 003: Add BM25 hybrid search
3. [ ] Experiment 004: Test reranking impact

## Reproduction
```bash
# Test fixed chunking (baseline)
python -m tests.test_baseline

# Test recursive chunking
python -c "
from docint.ingest.chunkers import FixedChunker, RecursiveChunker
from docint.ingest.loaders import TextLoader

loader = TextLoader()
doc = loader.load('eval_data/documents/techcorp_annual_report.txt')

fixed = list(FixedChunker(512, 50).chunk(doc))
recursive = list(RecursiveChunker(512, 50).chunk(doc))

print(f'Fixed: {len(fixed)} chunks')
print(f'Recursive: {len(recursive)} chunks')
"
```