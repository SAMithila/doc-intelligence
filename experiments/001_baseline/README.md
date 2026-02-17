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
Manual evaluation with 5 factual questions with verifiable answers.

## Results
 
**Documents:** 2  
**Total Chunks:** 15

### Query Performance

| Query                                              | Correct? | Answer |
|----------------------------------------------------|----------|--------|
| What was TechCorp's Q3 2024 revenue?               | ❌ No    | "Context does not provide information" |
| How much does CloudScale object storage cost?      | ✅ Yes   | $0.023/GB standard, $0.0125/GB infrequent |
| What is the employee count at TechCorp?            | ✅ Yes   | 12,500 |
| What security certifications does CloudScale have? | ✅ Yes   | SOC 2 Type II, ISO 27001, HIPAA, PCI DSS |
| What was the SecureNet acquisition price?          | ✅ Yes   | $320 million |

**Accuracy: 4/5 (80%)**

### Retrieval Metrics (Simulated Test)

| Metric    | @1    | @3    | @5    |
|-----------|-------|-------|-------|
| Recall    | 0.333 | 0.667 | 0.667 |
| Precision | 1.000 | 0.667 | 0.400 |
| NDCG      | 1.000 | 0.704 | 0.704 |

**MRR: 1.000**

*Note: These metrics are from the simulated test case (3 relevant docs, 5 retrieved). For the actual Q3 revenue query, the relevant chunk was NOT in top 5, meaning Recall@5 = 0 for that query.*

### Latency

| Metric         | Value   |
|----------------|---------|
| Avg Retrieval  | 530ms   |
| Avg Generation | 1900ms  |
| Avg Total      | 2450ms  |

### Pipeline Statistics

| Setting             | Value     |
|---------------------|-----------|
| Chunk Count         | 15        |
| Chunk Size          | 512 chars |
| Chunk Overlap       | 50 chars  |
| Embedding Model     | text-embedding-3-small |
| Embedding Dimension | 1536      |
| Top K               | 5         |
| LLM                 | gpt-4o-mini |

## Root Cause Analysis: Q3 Revenue Query Failure

**Query:** "What was TechCorp's Q3 2024 revenue?"  
**Expected:** $1.15 billion  
**Got:** "Context does not provide information"

### Retrieved Chunks Analysis

| Rank | Score | Contains Q3 Data? | Issue             |
|------|-------|-------------------|-------------------|
| 1    | 0.768 | ❌ No             | Executive summary (annual totals) |
| 2    | 0.719 | ❌ No             | Employee section  |
| 3    | 0.704 | ⚠️ Partial        | Has Q3 details but header cut off |
| 4    | 0.622 | ❌ No             | Q1 section        |
| 5    | 0.587 | ❌ No             | Risk factors      |

### Root Cause

Fixed chunking at 512 characters split the Q3 section header from its content.

Chunk 3 starts mid-word: `"ervices revenue hit $580 million..."` instead of `"services"`.

The section header `## Q3 2024 Performance` and the revenue figure `$1.15 billion` were separated into different chunks, breaking the semantic connection.

## Known Limitations Confirmed

1. **Fixed chunking splits content arbitrarily** ✅ Confirmed
   - Q3 section split mid-word
   - Header separated from content

2. **No keyword matching**
   - Pure semantic search
   - "Q3" keyword not weighted

3. **No reranking**
   - Relies entirely on embedding similarity

## Next Steps

Based on baseline results, prioritize experiments:
1. [x] Experiment 001: Baseline metrics ✅
2. [ ] Experiment 002: Recursive chunking (fix Q3 failure)
3. [ ] Experiment 003: Add BM25 hybrid search
4. [ ] Experiment 004: Test reranking impact

## Reproduction
```bash
# Set up environment
cd doc-intelligence
pip install -e .
export OPENAI_API_KEY=your_key

# Run baseline test
python -m tests.test_baseline

# Debug specific query
python -c "
from docint.config import Config
from docint.pipeline import RAGPipeline
import os

config = Config()
config.openai_api_key = os.environ['OPENAI_API_KEY']
config.vector_store.persist_directory = None

pipeline = RAGPipeline(config)
pipeline.ingest_directory('eval_data/documents')
result = pipeline.query('What was TechCorp Q3 2024 revenue?')

for i, r in enumerate(result.retrieval.results):
    print(f'[{i+1}] Score: {r.score:.4f}')
    print(f'Content: {r.content[:200]}...\n')
"
```