"""Test HyDE query expansion from Day 5."""
import os
import time
from docint.ingest.loaders import TextLoader
from docint.ingest.chunkers import RecursiveChunker
from docint.embeddings.openai import OpenAIEmbedder
from docint.store.chroma import ChromaStore
from docint.retrieval.hyde import HyDEExpander, MultiQueryExpander
from docint.generation.generator import Generator


def test_hyde():
    """Test HyDE query expansion."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('DAY 5: HyDE QUERY EXPANSION TEST')
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

    vector_store = ChromaStore(collection_name='hyde_test')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    hyde = HyDEExpander(api_key=api_key, embedder=embedder)
    generator = Generator(api_key=api_key)

    print(f'   ✓ {len(chunks)} chunks indexed')

    # Test query
    query = 'Q3 revenue'  # Intentionally vague/short

    print(f'\n2. Testing vague query: "{query}"')
    print('=' * 70)

    # Normal search
    print('\n📊 NORMAL SEARCH (embed query directly)')
    start = time.time()
    query_embedding = embedder.embed_single(query)
    normal_results = vector_store.search(query_embedding, top_k=5)
    normal_time = (time.time() - start) * 1000

    print(f'   Time: {normal_time:.0f}ms')
    print(f'\n   {"Rank":<6} {"Score":<10} {"Content Preview":<50}')
    print('   ' + '-' * 66)
    for i, r in enumerate(normal_results, 1):
        content = r.content[:45].replace('\n', ' ')
        print(f'   {i:<6} {r.score:<10.4f} {content}...')

    # HyDE search
    print('\n📊 HyDE SEARCH (embed hypothetical document)')
    start = time.time()
    hyde_result = hyde.expand(query)
    hyde_results = vector_store.search(hyde_result.embedding, top_k=5)
    hyde_time = (time.time() - start) * 1000

    print(f'   Time: {hyde_time:.0f}ms')
    print(f'\n   Hypothetical document generated:')
    print(f'   "{hyde_result.hypothetical_document[:100]}..."')
    print(f'\n   {"Rank":<6} {"Score":<10} {"Content Preview":<50}')
    print('   ' + '-' * 66)
    for i, r in enumerate(hyde_results, 1):
        content = r.content[:45].replace('\n', ' ')
        print(f'   {i:<6} {r.score:<10.4f} {content}...')

    # Compare answers
    print('\n' + '=' * 70)
    print('3. ANSWER COMPARISON')
    print('=' * 70)

    normal_contexts = [r.content for r in normal_results]
    normal_answer = generator.generate(query, normal_contexts)

    hyde_contexts = [r.content for r in hyde_results]
    hyde_answer = generator.generate(query, hyde_contexts)

    print(f'\n   Normal search: {normal_answer.answer[:70]}...')
    print(f'   HyDE search:   {hyde_answer.answer[:70]}...')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'   Normal search time: {normal_time:.0f}ms')
    print(f'   HyDE search time:   {hyde_time:.0f}ms')
    print(f'   Additional latency: +{hyde_time - normal_time:.0f}ms')


def test_hyde_vs_normal():
    """Compare HyDE vs normal search on all queries."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('FULL COMPARISON: NORMAL vs HyDE')
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

    vector_store = ChromaStore(collection_name='hyde_comparison')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    hyde = HyDEExpander(api_key=api_key, embedder=embedder)
    generator = Generator(api_key=api_key)

    # Test with both specific and vague queries
    queries = [
        ('specific', 'What was TechCorp Q3 2024 revenue?'),
        ('vague', 'Q3 revenue'),
        ('specific', 'How much does CloudScale object storage cost?'),
        ('vague', 'storage pricing'),
        ('specific', 'What is the employee count at TechCorp?'),
        ('vague', 'employees'),
    ]

    normal_correct = 0
    hyde_correct = 0
    total_hyde_time = 0

    for query_type, query in queries:
        # Normal search
        query_embedding = embedder.embed_single(query)
        normal_results = vector_store.search(query_embedding, top_k=5)
        normal_contexts = [r.content for r in normal_results]
        normal_answer = generator.generate(query, normal_contexts)
        normal_ok = 'not provide' not in normal_answer.answer.lower()
        if normal_ok:
            normal_correct += 1

        # HyDE search
        start = time.time()
        hyde_result = hyde.expand(query)
        total_hyde_time += (time.time() - start) * 1000

        hyde_results = vector_store.search(hyde_result.embedding, top_k=5)
        hyde_contexts = [r.content for r in hyde_results]
        hyde_answer = generator.generate(query, hyde_contexts)
        hyde_ok = 'not provide' not in hyde_answer.answer.lower()
        if hyde_ok:
            hyde_correct += 1

        normal_status = '✅' if normal_ok else '❌'
        hyde_status = '✅' if hyde_ok else '❌'

        print(f'\n[{query_type.upper()}] Q: {query}')
        print(f'   Normal {normal_status}: {normal_answer.answer[:60]}...')
        print(f'   HyDE   {hyde_status}: {hyde_answer.answer[:60]}...')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(
        f'Normal search: {normal_correct}/{len(queries)} ({100*normal_correct/len(queries):.0f}%)')
    print(
        f'HyDE search:   {hyde_correct}/{len(queries)} ({100*hyde_correct/len(queries):.0f}%)')
    print(f'Avg HyDE expansion time: {total_hyde_time/len(queries):.0f}ms')


def test_multi_query():
    """Test multi-query expansion."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('MULTI-QUERY EXPANSION TEST')
    print('=' * 70)

    expander = MultiQueryExpander(api_key=api_key)

    query = 'Q3 revenue'
    print(f'\nOriginal query: "{query}"')

    variations = expander.expand(query)

    print(f'\nGenerated variations:')
    for i, v in enumerate(variations):
        print(f'   {i+1}. {v}')


if __name__ == '__main__':
    test_hyde()
    test_hyde_vs_normal()
    test_multi_query()
