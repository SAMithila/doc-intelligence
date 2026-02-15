# Architecture Decisions 
Why I built things the way I did.

## Vector Store: ChromaDB over FAISS
Decision: Use ChromaDB

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