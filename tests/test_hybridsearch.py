"""Test hybrid search retriever against simple semantic search retriever"""
import os
from docint.ingest.loaders import TextLoader
from docint.ingest.chunkers import RecursiveChunker
from docint.embeddings.openai import OpenAIEmbedder
from docint.store.chroma import ChromaStore
from docint.retrieval.retriever import SimpleRetriever
from docint.retrieval.bm25 import BM25Index
from docint.retrieval.hybrid import HybridRetriever
from docint.generation.generator import Generator


def test_hybrid_search():
    """Compare semantic vs hybrid retrieval."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('DAY 3: HYBRID SEARCH TEST')
    print('=' * 70)

    # Setup
    print('\n1. Loading and chunking documents...')
    loader = TextLoader()
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)

    docs = list(loader.load_directory('eval_data/documents'))
    chunks = []
    for doc in docs:
        chunks.extend(list(chunker.chunk(doc)))
    print(f'   ✓ {len(chunks)} chunks created')

    # Embeddings
    print('\n2. Creating embeddings...')
    embedder = OpenAIEmbedder(api_key=api_key)
    contents = [c.content for c in chunks]
    embeddings = embedder.embed(contents)
    print(f'   ✓ {len(embeddings)} embeddings created')

    # Vector store
    print('\n3. Setting up vector store...')
    vector_store = ChromaStore(collection_name='hybrid_test')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )
    print(f'   ✓ {vector_store.count()} docs in vector store')

    # BM25 index
    print('\n4. Setting up BM25 index...')
    bm25_index = BM25Index()
    bm25_index.add(
        ids=[c.chunk_id for c in chunks],
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )
    print(f'   ✓ {bm25_index.count()} docs in BM25 index')

    # Retrievers
    semantic_retriever = SimpleRetriever(embedder, vector_store, top_k=5)
    hybrid_retriever = HybridRetriever(
        embedder, vector_store, bm25_index, top_k=5)

    # Generator
    generator = Generator(api_key=api_key)

    # Test queries
    queries = [
        'What was TechCorp Q3 2024 revenue?',
        'How much does CloudScale object storage cost?',
        'What is the employee count at TechCorp?',
        'What security certifications does CloudScale have?',
        'What was the SecureNet acquisition price?',
    ]

    print('\n' + '=' * 70)
    print('5. COMPARING SEMANTIC vs HYBRID')
    print('=' * 70)

    semantic_correct = 0
    hybrid_correct = 0

    for q in queries:
        # Semantic retrieval
        sem_results = semantic_retriever.retrieve(q)
        sem_answer = generator.generate(q, sem_results.contexts)
        sem_ok = 'not provide' not in sem_answer.answer.lower()
        if sem_ok:
            semantic_correct += 1

        # Hybrid retrieval
        hyb_results = hybrid_retriever.retrieve(q)
        hyb_contexts = [r.content for r in hyb_results]
        hyb_answer = generator.generate(q, hyb_contexts)
        hyb_ok = 'not provide' not in hyb_answer.answer.lower()
        if hyb_ok:
            hybrid_correct += 1

        sem_status = '✅' if sem_ok else '❌'
        hyb_status = '✅' if hyb_ok else '❌'

        print(f'\nQ: {q}')
        print(f'   Semantic {sem_status}: {sem_answer.answer[:70]}...')
        print(f'   Hybrid   {hyb_status}: {hyb_answer.answer[:70]}...')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(
        f'Semantic accuracy: {semantic_correct}/{len(queries)} ({100*semantic_correct/len(queries):.0f}%)')
    print(
        f'Hybrid accuracy:   {hybrid_correct}/{len(queries)} ({100*hybrid_correct/len(queries):.0f}%)')

    if hybrid_correct >= semantic_correct:
        print('\n✅ Hybrid search performs equal or better!')
    else:
        print('\n⚠️ Semantic search performed better on this test set')


def test_ranking_comparison():
    """Show how BM25 and semantic rank differently."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('RANKING COMPARISON: SEMANTIC vs BM25')
    print('=' * 70)

    # Setup
    loader = TextLoader()
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)

    docs = list(loader.load_directory('eval_data/documents'))
    chunks = []
    for doc in docs:
        chunks.extend(list(chunker.chunk(doc)))

    embedder = OpenAIEmbedder(api_key=api_key)
    contents = [c.content for c in chunks]
    embeddings = embedder.embed(contents)

    vector_store = ChromaStore(collection_name='ranking_test')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    bm25_index = BM25Index()
    bm25_index.add(
        ids=[c.chunk_id for c in chunks],
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    hybrid = HybridRetriever(embedder, vector_store, bm25_index, top_k=5)

    query = 'What was TechCorp Q3 2024 revenue?'
    print(f'\nQuery: "{query}"\n')

    results = hybrid.retrieve(query)

    print(f'{"Rank":<6} {"Semantic":<10} {"BM25":<10} {"Content Preview":<50}')
    print('-' * 76)

    for i, r in enumerate(results, 1):
        sem = r.semantic_rank if r.semantic_rank else '-'
        bm25 = r.bm25_rank if r.bm25_rank else '-'
        content = r.content[:45].replace('\n', ' ')
        print(f'{i:<6} {sem:<10} {bm25:<10} {content}...')


if __name__ == '__main__':
    test_hybrid_search()
    test_ranking_comparison()
