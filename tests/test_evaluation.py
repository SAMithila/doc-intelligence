"""Day 6: Run comprehensive evaluation."""
import os
import json
from pathlib import Path

from docint.config import load_config
from docint.pipeline import RAGPipeline
from docint.evaluation.evaluator import RAGEvaluator


def run_evaluation():
    """Run evaluation on the full dataset."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('DAY 6: COMPREHENSIVE RAG EVALUATION')
    print('=' * 70)

    # Setup pipeline
    print('\n1. Setting up pipeline...')
    config = load_config('configs/default.yaml')
    config.openai_api_key = api_key
    config.vector_store.persist_directory = None

    pipeline = RAGPipeline(config)
    pipeline.ingest_directory('eval_data/documents')
    print(f'   ✓ Pipeline ready')

    # Run evaluation
    print('\n2. Running evaluation...')
    evaluator = RAGEvaluator(pipeline, config_name='recursive_semantic')

    dataset_path = 'eval_data/evaluation_dataset.json'
    report = evaluator.evaluate(dataset_path)

    # Print summary
    report.print_summary()

    # Save report
    output_dir = Path('experiments/006_evaluation/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.save_report(
        report,
        str(output_dir / 'baseline_evaluation.json')
    )

    return report


def compare_approaches():
    """Compare different retrieval approaches."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('COMPARING RETRIEVAL APPROACHES')
    print('=' * 70)

    # This will be expanded to compare:
    # - Semantic only
    # - Hybrid (BM25 + Semantic)
    # - With reranking
    # - With HyDE

    # For now, just run baseline
    report = run_evaluation()

    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    print(f'\n{"Approach":<25} {"Overall":<10} {"Easy":<10} {"Vague":<10} {"Hard":<10} {"Latency":<10}')
    print('-' * 75)
    print(f'{"Recursive + Semantic":<25} {report.overall_accuracy:<10.1%} {report.easy_accuracy:<10.1%} {report.vague_accuracy:<10.1%} {report.hard_accuracy:<10.1%} {report.avg_total_time_ms:<10.0f}ms')


if __name__ == '__main__':
    compare_approaches()
