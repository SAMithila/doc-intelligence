"""Test reranking"""
import os
import time
from docint.ingest.loaders import TextLoader
from docint.ingest.chunkers import RecursiveChunker
from docint.embeddings.openai import OpenAIEmbedder
from docint.store.chroma import ChromaStore
from docint.retrieval.retriever import SimpleRetriever
from docint.retrieval.reranker import LLMReranker
from docint.generation.generator import Generator


def test_reranking():
    """Test LLM-based reranking."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('DAY 4: RERANKING TEST')
    print('=' * 70)

    # Setup
    print('\n1. Setting up pipeline...')
    loader = TextLoader()
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)

    docs = list(loader.load_directory('eval_data/documents'))
    chunks = []
    for doc in docs:
        chunks.extend(list(chunker.chunk(doc)))

    embedder = OpenAIEmbedder(api_key=api_key)
    contents = [c.content for c in chunks]
    embeddings = embedder.embed(contents)

    vector_store = ChromaStore(collection_name='rerank_test')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    retriever = SimpleRetriever(
        embedder, vector_store, top_k=10)  # Get more for reranking
    reranker = LLMReranker(api_key=api_key)
    generator = Generator(api_key=api_key)

    print(f'   ✓ {len(chunks)} chunks indexed')

    # Test query
    query = 'What was TechCorp Q3 2024 revenue?'

    print(f'\n2. Testing query: "{query}"')
    print('=' * 70)

    # Stage 1: Initial retrieval
    print('\n📊 STAGE 1: Initial Retrieval (top 10)')
    start = time.time()
    initial_results = retriever.retrieve(query, top_k=10)
    retrieval_time = (time.time() - start) * 1000

    print(f'   Time: {retrieval_time:.0f}ms')
    print(f'\n   {"Rank":<6} {"Score":<10} {"Content Preview":<50}')
    print('   ' + '-' * 66)
    for i, r in enumerate(initial_results.results[:5], 1):
        content = r.content[:45].replace('\n', ' ')
        print(f'   {i:<6} {r.score:<10.4f} {content}...')

    # Stage 2: Reranking
    print('\n📊 STAGE 2: After Reranking (top 5)')

    # Convert to format reranker expects
    results_for_rerank = [
        {
            'chunk_id': r.chunk_id,
            'content': r.content,
            'score': r.score,
            'metadata': r.metadata,
        }
        for r in initial_results.results
    ]

    start = time.time()
    reranked = reranker.rerank(query, results_for_rerank, top_k=5)
    rerank_time = (time.time() - start) * 1000

    print(f'   Time: {rerank_time:.0f}ms')
    print(f'\n   {"New":<5} {"Old":<5} {"Rerank":<10} {"Content Preview":<45}')
    print('   ' + '-' * 66)
    for r in reranked:
        content = r.content[:40].replace('\n', ' ')
        moved = '↑' if r.new_rank < r.original_rank else (
            '↓' if r.new_rank > r.original_rank else '=')
        print(
            f'   {r.new_rank:<5} {r.original_rank}{moved:<4} {r.rerank_score:<10.2f} {content}...')

    # Generate answers
    print('\n' + '=' * 70)
    print('3. ANSWER COMPARISON')
    print('=' * 70)

    # Without reranking (top 5 from initial)
    no_rerank_contexts = [r.content for r in initial_results.results[:5]]
    no_rerank_answer = generator.generate(query, no_rerank_contexts)

    # With reranking
    rerank_contexts = [r.content for r in reranked]
    rerank_answer = generator.generate(query, rerank_contexts)

    print(f'\n   Without reranking: {no_rerank_answer.answer[:70]}...')
    print(f'   With reranking:    {rerank_answer.answer[:70]}...')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'   Retrieval time:  {retrieval_time:.0f}ms')
    print(f'   Reranking time:  {rerank_time:.0f}ms')
    print(f'   Total time:      {retrieval_time + rerank_time:.0f}ms')
    print(f'\n   Reranking moved chunks:')
    for r in reranked:
        if r.new_rank != r.original_rank:
            direction = '↑' if r.new_rank < r.original_rank else '↓'
            print(f'      Rank {r.original_rank} → {r.new_rank} {direction}')


def test_all_queries_with_reranking():
    """Test all queries with and without reranking."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('FULL COMPARISON: WITH vs WITHOUT RERANKING')
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

    vector_store = ChromaStore(collection_name='rerank_full_test')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    retriever = SimpleRetriever(embedder, vector_store, top_k=10)
    reranker = LLMReranker(api_key=api_key)
    generator = Generator(api_key=api_key)

    queries = [
        'What was TechCorp Q3 2024 revenue?',
        'How much does CloudScale object storage cost?',
        'What is the employee count at TechCorp?',
        'What security certifications does CloudScale have?',
        'What was the SecureNet acquisition price?',
    ]

    no_rerank_correct = 0
    rerank_correct = 0
    total_rerank_time = 0

    for q in queries:
        # Without reranking
        initial = retriever.retrieve(q, top_k=5)
        no_rerank_answer = generator.generate(q, initial.contexts)
        no_rerank_ok = 'not provide' not in no_rerank_answer.answer.lower()
        if no_rerank_ok:
            no_rerank_correct += 1

        # With reranking
        initial_10 = retriever.retrieve(q, top_k=10)
        results_for_rerank = [
            {'chunk_id': r.chunk_id, 'content': r.content,
                'score': r.score, 'metadata': r.metadata}
            for r in initial_10.results
        ]

        start = time.time()
        reranked = reranker.rerank(q, results_for_rerank, top_k=5)
        total_rerank_time += (time.time() - start) * 1000

        rerank_contexts = [r.content for r in reranked]
        rerank_answer = generator.generate(q, rerank_contexts)
        rerank_ok = 'not provide' not in rerank_answer.answer.lower()
        if rerank_ok:
            rerank_correct += 1

        no_status = '✅' if no_rerank_ok else '❌'
        re_status = '✅' if rerank_ok else '❌'

        print(f'\nQ: {q}')
        print(f'   No rerank {no_status}: {no_rerank_answer.answer[:60]}...')
        print(f'   Reranked  {re_status}: {rerank_answer.answer[:60]}...')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(
        f'Without reranking: {no_rerank_correct}/{len(queries)} ({100*no_rerank_correct/len(queries):.0f}%)')
    print(
        f'With reranking:    {rerank_correct}/{len(queries)} ({100*rerank_correct/len(queries):.0f}%)')
    print(
        f'Avg rerank time:   {total_rerank_time/len(queries):.0f}ms per query')


if __name__ == '__main__':
    test_reranking()
    test_all_queries_with_reranking()
