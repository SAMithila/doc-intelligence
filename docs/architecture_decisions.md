# Architecture Decisions 
Why I built things the way I did.

## Vector Store: ChromaDB over FAISS
### Decision: Use ChromaDB

#### Alternatives considered:
1. FAISS - faster for large scale, but requires manual persistence
2. Pinecone - managed service, but adds external dependency
3. Weaviate - good but overkill for this scale

#### Why ChromaDB:
1. Built-in persistence (just set a path)
2. Works locally without setup
3. Good enough performance for <100k documents
4. Simple API that matches my mental model

#### Tradeoff: 
If I needed to scale to millions of docs, I'd switch to FAISS with a custom persistence layer.


## Chunk Size: 512 Characters
### Decision: 512 chars with 50 char overlap

#### What I tried:
1. 256 chars: Too fragmented. Sentences got split. Context lost.
2. 512 chars: Good balance. Most paragraphs fit.
3. 1024 chars: Too big. Retrieved chunks had too much irrelevant content.

#### The test that decided it:
Query: "What was Q3 revenue?"

1. At 256: Answer was split across 3 chunks, none had full context
2. At 512: Answer fit in one chunk
3. At 1024: Chunk had Q3 + Q4 + other stuff, confused the LLM

#### Tradeoff: 
512 means more chunks (17 vs 8 at 1024), slightly higher embedding cost. Worth it for retrieval quality.


## Recursive over Fixed Chunking
### Decision: Use recursive text splitter

#### The bug that forced this:
Fixed chunking at 512 chars split "## Q3 2024 Performance" header from the "$1.15 billion" revenue figure. Query failed because semantic search couldn't connect "Q3 revenue" to a chunk that started with "ervices revenue hit..."

The word "services" literally got cut in half.

#### Why recursive works:
1. First tries to split on "\n\n" (section breaks)
2. Then "\n" (paragraphs)
3. Then ". " (sentences)
4. Only splits mid-word as last resort

#### Tradeoff: 
Variable chunk sizes. Some chunks are 200 chars, some are 500. But section integrity is preserved.


## SimpleRetriever over MMR
### Decision: Top-k similarity, no diversity reranking

#### Why not MMR (Maximal Marginal Relevance):
1. MMR optimizes for diversity - useful when you want varied results
2. My use case wants the MOST relevant chunks, even if similar
3. For Q&A, I'd rather have 5 chunks all about Q3 revenue than 5 diverse chunks

#### When I'd use MMR: 
If building a research assistant where users want to explore different aspects of a topic.


## Hybrid Search: Why 50/50 Weighting
### Decision: Equal weight to BM25 and semantic search

#### What I tried:
1. 70/30 semantic: Slight improvement on specific queries
2. 50/50: Best overall
3. 30/70 BM25: Worse on conceptual queries

#### The query that showed the value:
"HIPAA compliance" - semantic search found "security certifications" chunks. BM25 found chunks with literal "HIPAA" keyword. Combined ranking got the right chunk to #1.

#### Tradeoff: 
Two indexes to maintain (vector + inverted). Minimal memory overhead for my doc size.


## No Reranking by Default
#### Decision: Skip cross-encoder reranking
#### What I built: LLM-based reranker that scores each retrieved chunk 0-10.

#### Why I'm not using it:
1. 5 seconds latency per query (10 LLM calls)
2. Accuracy was already 83% without it
3. Only improved ranking position, not final answer quality

#### When I'd enable it:
1. Batch processing where latency doesn't matter
2. High-stakes queries where accuracy is critical
3. If I switch to local cross-encoder (would be ~100ms instead of 5s)


## HyDE: Conditional, Not Default
#### Decision: Only use HyDE for short/vague queries

#### The data:
| Query type | Semantic | HyDE |
|------------|----------|------|
| Specific   | 90%      | 90%  |
| Vague      | 50%      | 70%  |

HyDE helps vague queries (+20%) but doesn't change much for specific ones.

#### Latency
Adds ~1.5s for the LLM call to generate the hypothetical.

#### Decision
Using conditionally - only for short queries (3 words or less) or when retrieval scores look low.
```python
if len(query.split()) <= 3:
    use_hyde()
```


## Embedding Model: text-embedding-3-small
#### Decision: Use OpenAI's small model, not large

#### Why not text-embedding-3-large:
1. 2x the dimension (3072 vs 1536)
2. 2x the storage
3. Marginal quality improvement for my use case

#### My reasoning: 
For short documents with clear factual content, the small model captures enough semantic meaning. I'd upgrade if working with nuanced legal or medical text.

## No Streaming by Default
#### Decision: Wait for full response

#### Why: 
For evaluation, I need the complete answer to check correctness. Streaming complicates the evaluation loop.
When I'd add it: Building a chat UI where perceived latency matters.

## What I'd Do Differently
Start with evaluation dataset - I built features for 5 days before having hard test cases. Wasted time proving things worked when they already worked.
Measure latency earlier - Reranking looked promising until I measured. Should measure on day 1.
Smaller iteration cycles - Spent too long on architecture before validating basic retrieval worked.
