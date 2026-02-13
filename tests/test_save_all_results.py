"""Save evaluation results for all approaches."""
import os
import json
from pathlib import Path

from docint.ingest.loaders import TextLoader
from docint.ingest.chunkers import RecursiveChunker
from docint.embeddings.openai import OpenAIEmbedder
from docint.store.chroma import ChromaStore
from docint.retrieval.retriever import SimpleRetriever
from docint.retrieval.bm25 import BM25Index
from docint.retrieval.hybrid import HybridRetriever
from docint.retrieval.hyde import HyDEExpander
from docint.generation.generator import Generator


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

    vector_store = ChromaStore(collection_name='eval_all')
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

    return {
        'embedder': embedder,
        'vector_store': vector_store,
        'bm25_index': bm25_index,
        'generator': Generator(api_key=api_key),
    }


def evaluate_approach(name, retriever_func, components, queries, generator):
    """Evaluate a single approach and return results."""
    results = []
    correct = {'easy': 0, 'vague': 0, 'hard': 0, 'total': 0}
    counts = {'easy': 0, 'vague': 0, 'hard': 0}

    for q in queries:
        query = q['query']
        keywords = q.get('keywords', [])
        category = q['category']
        counts[category] += 1

        # Get contexts using the retriever function
        contexts = retriever_func(query)

        # Generate answer
        answer = generator.generate(query, contexts)

        # Check correctness
        answer_lower = answer.answer.lower()
        is_correct = False
        if 'not provide' not in answer_lower and 'cannot find' not in answer_lower:
            is_correct = any(kw.lower() in answer_lower for kw in keywords)

        if is_correct:
            correct[category] += 1
            correct['total'] += 1

        results.append({
            'query_id': q['id'],
            'category': category,
            'query': query,
            'expected_answer': q['expected_answer'],
            'actual_answer': answer.answer,
            'correct': is_correct,
        })

    return {
        'name': name,
        'overall_accuracy': correct['total'] / len(queries),
        'easy_accuracy': correct['easy'] / counts['easy'] if counts['easy'] else 0,
        'vague_accuracy': correct['vague'] / counts['vague'] if counts['vague'] else 0,
        'hard_accuracy': correct['hard'] / counts['hard'] if counts['hard'] else 0,
        'correct': correct,
        'counts': counts,
        'results': results,
    }


def run_and_save_all():
    """Run evaluation for all approaches and save results."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('EVALUATING ALL APPROACHES')
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

    # Define retriever functions
    def semantic_retrieve(query):
        return semantic_retriever.retrieve(query).contexts

    def hybrid_retrieve(query):
        results = hybrid_retriever.retrieve(query)
        return [r.content for r in results]

    def hyde_retrieve(query):
        hyde_result = hyde_expander.expand(query)
        results = vector_store.search(hyde_result.embedding, top_k=5)
        return [r.content for r in results]

    # Output directory
    output_dir = Path('experiments/006_evaluation/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 1. Semantic
    print('\n[1/3] Evaluating SEMANTIC...')
    semantic_results = evaluate_approach(
        'semantic', semantic_retrieve, components, queries, generator
    )
    all_results.append(semantic_results)
    with open(output_dir / 'semantic_evaluation.json', 'w') as f:
        json.dump(semantic_results, f, indent=2)
    print(f'   ✓ Saved: semantic_evaluation.json')

    # 2. Hybrid
    print('\n[2/3] Evaluating HYBRID...')
    hybrid_results = evaluate_approach(
        'hybrid', hybrid_retrieve, components, queries, generator
    )
    all_results.append(hybrid_results)
    with open(output_dir / 'hybrid_evaluation.json', 'w') as f:
        json.dump(hybrid_results, f, indent=2)
    print(f'   ✓ Saved: hybrid_evaluation.json')

    # 3. HyDE
    print('\n[3/3] Evaluating HYDE...')
    hyde_results = evaluate_approach(
        'hyde', hyde_retrieve, components, queries, generator
    )
    all_results.append(hyde_results)
    with open(output_dir / 'hyde_evaluation.json', 'w') as f:
        json.dump(hyde_results, f, indent=2)
    print(f'   ✓ Saved: hyde_evaluation.json')

    # Print comparison table
    print('\n' + '=' * 70)
    print('RESULTS COMPARISON')
    print('=' * 70)

    print(
        f'\n{"Approach":<15} {"Overall":<12} {"Easy":<12} {"Vague":<12} {"Hard":<12}')
    print('-' * 63)

    for r in all_results:
        name = r['name'].upper()
        overall = f"{r['overall_accuracy']:.1%}"
        easy = f"{r['easy_accuracy']:.1%}"
        vague = f"{r['vague_accuracy']:.1%}"
        hard = f"{r['hard_accuracy']:.1%}"
        print(f'{name:<15} {overall:<12} {easy:<12} {vague:<12} {hard:<12}')

    # Save summary
    summary = {
        'approaches': [
            {
                'name': r['name'],
                'overall': r['overall_accuracy'],
                'easy': r['easy_accuracy'],
                'vague': r['vague_accuracy'],
                'hard': r['hard_accuracy'],
            }
            for r in all_results
        ]
    }
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n✓ All results saved to: {output_dir}')


if __name__ == '__main__':
    run_and_save_all()
