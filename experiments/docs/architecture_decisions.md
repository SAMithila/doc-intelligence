# Architecture Decisions 
Why I built things the way I did.

## Vector Store: ChromaDB over FAISS
#### Decision: Use ChromaDB

### Alternatives considered:
1. FAISS - faster for large scale, but requires manual persistence
2. Pinecone - managed service, but adds external dependency
3. Weaviate - good but overkill for this scale

### Why ChromaDB:
1. Built-in persistence (just set a path)
2. Works locally without setup
3. Good enough performance for <100k documents
4. Simple API that matches my mental model

### Tradeoff: 
If I needed to scale to millions of docs, I'd switch to FAISS with a custom persistence layer.


# Chunk Size: 512 Characters
#### Decision: 512 chars with 50 char overlap

### What I tried:
1. 256 chars: Too fragmented. Sentences got split. Context lost.
2. 512 chars: Good balance. Most paragraphs fit.
3. 1024 chars: Too big. Retrieved chunks had too much irrelevant content.

### The test that decided it:
Query: "What was Q3 revenue?"

1. At 256: Answer was split across 3 chunks, none had full context
2. At 512: Answer fit in one chunk
3. At 1024: Chunk had Q3 + Q4 + other stuff, confused the LLM

### Tradeoff: 
512 means more chunks (17 vs 8 at 1024), slightly higher embedding cost. Worth it for retrieval quality.