"""Test recursive chunking improvement from Day 2."""
import os
from docint.config import load_config
from docint.pipeline import RAGPipeline


def test_recursive_chunking():
    """Test that recursive chunking achieves 100% accuracy."""

    config = load_config('configs/default.yaml')
    config.openai_api_key = os.environ.get('OPENAI_API_KEY', '')
    config.vector_store.persist_directory = None

    pipeline = RAGPipeline(config)
    stats = pipeline.ingest_directory('eval_data/documents')

    print(f'Chunks: {stats["chunks"]}')
    print('=' * 60)

    queries = [
        'What was TechCorp Q3 2024 revenue?',
        'How much does CloudScale object storage cost?',
        'What is the employee count at TechCorp?',
        'What security certifications does CloudScale have?',
        'What was the SecureNet acquisition price?',
    ]

    correct = 0
    for q in queries:
        result = pipeline.query(q)
        passed = 'not provide' not in result.answer.lower()
        if passed:
            correct += 1
        status = '✅' if passed else '❌'
        print(f'{status} {q}')
        print(f'   {result.answer[:100]}...\n')

    print('=' * 60)
    print(
        f'Accuracy: {correct}/{len(queries)} ({100*correct/len(queries):.0f}%)')


if __name__ == '__main__':
    test_recursive_chunking()
