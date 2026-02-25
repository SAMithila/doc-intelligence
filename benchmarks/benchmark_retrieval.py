"""Benchmark retrieval performance."""
import os
import time
import statistics
from docint.config import load_config
from docint.pipeline import RAGPipeline


def run_benchmark():
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('-' * 60)
    print('RETRIEVAL BENCHMARK')
    print('-' * 60)

    # Setup
    config = load_config('configs/default.yaml')
    config.openai_api_key = api_key

    print('\n📦 Indexing...')
    start = time.time()
    pipeline = RAGPipeline(config)
    pipeline.ingest_directory('eval_data/documents')
    indexing_time = time.time() - start

    stats = pipeline.get_stats()
    print(f'   Documents: {stats.get("document_count", 0)}')
    print(f'   Chunks: {stats.get("chunk_count", 0)}')
    print(f'   Indexing time: {indexing_time:.2f}s')

    # Queries
    queries = [
        "What was TechCorp Q3 2024 revenue?",
        "How many employees does TechCorp have?",
        "What security certifications does CloudScale have?",
        "How much does CloudScale storage cost?",
        "What was the SecureNet acquisition price?",
    ]

    print(f'\n🔍 Running {len(queries)} queries...')

    latencies = []
    retrieval_times = []
    generation_times = []

    for q in queries:
        start = time.time()
        result = pipeline.query(q)
        total = (time.time() - start) * 1000

        latencies.append(total)
        retrieval_times.append(result.latency_ms.get('retrieval_ms', 0))
        generation_times.append(result.latency_ms.get('generation_ms', 0))

    # Results
    print('\n' + '-' * 60)
    print('RESULTS')
    print('-' * 60)

    print(f'\n📊 Indexing')
    print(f'   Time: {indexing_time:.2f}s')
    print(f'   Chunks: {stats.get("chunk_count", 0)}')

    print(f'\n⏱️  Query Latency (n={len(queries)})')
    print(f'   P50: {statistics.median(latencies):.0f}ms')
    print(f'   P95: {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms')
    print(f'   Mean: {statistics.mean(latencies):.0f}ms')

    print(f'\n📈 Breakdown')
    print(f'   Retrieval (mean): {statistics.mean(retrieval_times):.0f}ms')
    print(f'   Generation (mean): {statistics.mean(generation_times):.0f}ms')

    print('\n' + '=' * 60)

    return {
        'indexing_time_s': indexing_time,
        'chunks': stats.get('chunk_count', 0),
        'latency_p50_ms': statistics.median(latencies),
        'latency_p95_ms': sorted(latencies)[int(len(latencies)*0.95)],
        'retrieval_mean_ms': statistics.mean(retrieval_times),
        'generation_mean_ms': statistics.mean(generation_times),
    }


if __name__ == '__main__':
    run_benchmark()
