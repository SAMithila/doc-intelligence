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

#### Tradeoff: Two indexes to maintain (vector + inverted). Minimal memory overhead for my doc size.

