Architecture Decisions
Why I built things the way I did.
Vector Store: ChromaDB over FAISS
Decision: Use ChromaDB
Alternatives considered:

FAISS - faster for large scale, but requires manual persistence
Pinecone - managed service, but adds external dependency
Weaviate - good but overkill for this scale

Why ChromaDB:

Built-in persistence (just set a path)
Works locally without setup
Good enough performance for <100k documents
Simple API that matches my mental model

Tradeoff: If I needed to scale to millions of docs, I'd switch to FAISS with a custom persistence layer.