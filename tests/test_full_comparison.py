"""Compare all retrieval approaches on the evaluation dataset."""
import os
import json
from pathlib import Path

from docint.config import load_config
from docint.pipeline import RAGPipeline
from docint.ingest.loaders import TextLoader
from docint.ingest.chunkers import RecursiveChunker
from docint.embeddings.openai import OpenAIEmbedder
from docint.store.chroma import ChromaStore
from docint.retrieval.retriever import SimpleRetriever
from docint.retrieval.bm25 import BM25Index
from docint.retrieval.hybrid import HybridRetriever
from docint.retrieval.hyde import HyDEExpander
from docint.generation.generator import Generator
from docint.evaluation.evaluator import RAGEvaluator, EvaluationReport


def setup_components(api_key: str):
    """Set up shared components."""
    loader = TextLoader()
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)

    docs = list(loader.load_directory('eval_data/documents'))
    chunks = []
    for doc in docs:
        chunks.extend(list(chunker.chunk(doc)))

    embedder = OpenAIEmbedder(api_key=api_key)
    contents = [c.content for c in chunks]
    embeddings = embedder.embed(contents)

    # Vector store
    vector_store = ChromaStore(collection_name='comparison_test')
    vector_store.clear()
    vector_store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embeddings,
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    # BM25 index
    bm25_index = BM25Index()
    bm25_index.add(
        ids=[c.chunk_id for c in chunks],
        contents=contents,
        metadatas=[c.metadata for c in chunks],
    )

    generator = Generator(api_key=api_key)

    return {
        'chunks': chunks,
        'embedder': embedder,
        'vector_store': vector_store,
        'bm25_index': bm25_index,
        'generator': generator,
    }


def run_full_comparison():
    """Run comparison of all approaches."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('FULL COMPARISON: ALL RETRIEVAL APPROACHES')
    print('=' * 70)

    # Load dataset
    with open('eval_data/evaluation_dataset.json') as f:
        dataset = json.load(f)
    queries = dataset['queries']

    print(f'\nDataset: {len(queries)} queries')
    print('Setting up components...')

    components = setup_components(api_key)
    embedder = components['embedder']
    vector_store = components['vector_store']
    bm25_index = components['bm25_index']
    generator = components['generator']

    # Create retrievers
    semantic_retriever = SimpleRetriever(embedder, vector_store, top_k=5)
    hybrid_retriever = HybridRetriever(
        embedder, vector_store, bm25_index, top_k=5)
    hyde_expander = HyDEExpander(api_key=api_key, embedder=embedder)

    results = {
        'semantic': {'easy': 0, 'vague': 0, 'hard': 0, 'total': 0},
        'hybrid': {'easy': 0, 'vague': 0, 'hard': 0, 'total': 0},
        'hyde': {'easy': 0, 'vague': 0, 'hard': 0, 'total': 0},
    }

    category_counts = {'easy': 0, 'vague': 0, 'hard': 0}

    print('\nRunning evaluation...\n')

    for i, q in enumerate(queries):
        query = q['query']
        keywords = q.get('keywords', [])
        category = q['category']
        category_counts[category] += 1

        def check_correct(answer: str) -> bool:
            answer_lower = answer.lower()
            if 'not provide' in answer_lower or 'cannot find' in answer_lower:
                return False
            return any(kw.lower() in answer_lower for kw in keywords)

        # 1. Semantic search
        sem_results = semantic_retriever.retrieve(query)
        sem_answer = generator.generate(query, sem_results.contexts)
        if check_correct(sem_answer.answer):
            results['semantic'][category] += 1
            results['semantic']['total'] += 1

        # 2. Hybrid search
        hyb_results = hybrid_retriever.retrieve(query)
        hyb_contexts = [r.content for r in hyb_results]
        hyb_answer = generator.generate(query, hyb_contexts)
        if check_correct(hyb_answer.answer):
            results['hybrid'][category] += 1
            results['hybrid']['total'] += 1

        # 3. HyDE (only for vague queries to save API calls)
        if category == 'vague':
            hyde_result = hyde_expander.expand(query)
            hyde_search = vector_store.search(hyde_result.embedding, top_k=5)
            hyde_contexts = [r.content for r in hyde_search]
            hyde_answer = generator.generate(query, hyde_contexts)
            if check_correct(hyde_answer.answer):
                results['hyde']['vague'] += 1
                results['hyde']['total'] += 1
        else:
            # For non-vague, use semantic as baseline
            if check_correct(sem_answer.answer):
                results['hyde'][category] += 1
                results['hyde']['total'] += 1

        # Progress
        sem_ok = '✓' if check_correct(sem_answer.answer) else '✗'
        hyb_ok = '✓' if check_correct(hyb_answer.answer) else '✗'
        print(
            f'[{i+1}/{len(queries)}] [{category}] Sem:{sem_ok} Hyb:{hyb_ok} - {query[:40]}...')

    # Print summary
    print('\n' + '=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)

    print(
        f'\n{"Approach":<15} {"Overall":<12} {"Easy":<12} {"Vague":<12} {"Hard":<12}')
    print('-' * 63)

    for approach in ['semantic', 'hybrid', 'hyde']:
        r = results[approach]
        total_pct = f"{r['total']}/{len(queries)} ({100*r['total']/len(queries):.0f}%)"
        easy_pct = f"{r['easy']}/{category_counts['easy']} ({100*r['easy']/category_counts['easy']:.0f}%)"
        vague_pct = f"{r['vague']}/{category_counts['vague']} ({100*r['vague']/category_counts['vague']:.0f}%)"
        hard_pct = f"{r['hard']}/{category_counts['hard']} ({100*r['hard']/category_counts['hard']:.0f}%)"

        print(
            f'{approach.upper():<15} {total_pct:<12} {easy_pct:<12} {vague_pct:<12} {hard_pct:<12}')

    # Key findings
    print('\n' + '=' * 70)
    print('KEY FINDINGS')
    print('=' * 70)

    sem_vague = results['semantic']['vague']
    hyb_vague = results['hybrid']['vague']
    hyde_vague = results['hyde']['vague']

    if hyb_vague > sem_vague:
        print(f'\n✅ Hybrid improved vague queries: {sem_vague} → {hyb_vague}')
    else:
        print(
            f'\n➖ Hybrid same as semantic for vague queries: {sem_vague} vs {hyb_vague}')

    if hyde_vague > sem_vague:
        print(f'✅ HyDE improved vague queries: {sem_vague} → {hyde_vague}')
    else:
        print(
            f'➖ HyDE same as semantic for vague queries: {sem_vague} vs {hyde_vague}')


if __name__ == '__main__':
    run_full_comparison()
