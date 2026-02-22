"""Test hallucination detection from Day 7."""
import os
from docint.config import load_config
from docint.pipeline import RAGPipeline
from docint.verification.groundedness import GroundednessChecker, CitationExtractor


def test_groundedness_checker():
    """Test the groundedness checker on known examples."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('=' * 70)
    print('DAY 7: HALLUCINATION DETECTION TEST')
    print('=' * 70)

    checker = GroundednessChecker(api_key=api_key)

    # Test 1: Grounded answer
    print('\n📋 TEST 1: Grounded Answer')
    print('-' * 40)

    context = ["TechCorp Q3 2024 revenue was $1.15 billion, up 15% from Q2."]
    question = "What was TechCorp's Q3 2024 revenue?"
    answer = "TechCorp's Q3 2024 revenue was $1.15 billion."

    result = checker.check(question, answer, context)

    print(f'Answer: {answer}')
    print(f'Grounded: {result.is_grounded}')
    print(f'Confidence: {result.confidence}')
    print(f'Explanation: {result.explanation}')

    assert result.is_grounded, "Should be grounded"
    print('✅ Passed')

    # Test 2: Hallucinated answer
    print('\n📋 TEST 2: Hallucinated Answer')
    print('-' * 40)

    context = ["TechCorp Q3 2024 revenue was $1.15 billion."]
    question = "What was TechCorp's Q3 2024 profit margin?"
    answer = "TechCorp's Q3 2024 profit margin was 23%."  # Not in context!

    result = checker.check(question, answer, context)

    print(f'Answer: {answer}')
    print(f'Grounded: {result.is_grounded}')
    print(f'Confidence: {result.confidence}')
    print(f'Unsupported claims: {result.unsupported_claims}')
    print(f'Explanation: {result.explanation}')

    assert not result.is_grounded, "Should NOT be grounded (hallucination)"
    print('✅ Passed - Hallucination detected!')

    # Test 3: Partial hallucination
    print('\n📋 TEST 3: Partial Hallucination')
    print('-' * 40)

    context = ["TechCorp has 12,500 employees globally."]
    question = "Tell me about TechCorp's workforce."
    answer = "TechCorp has 12,500 employees globally and plans to hire 2,000 more next year."

    result = checker.check(question, answer, context)

    print(f'Answer: {answer}')
    print(f'Grounded: {result.is_grounded}')
    print(f'Supported: {result.supported_claims}')
    print(f'Unsupported: {result.unsupported_claims}')

    print('✅ Partial hallucination detected')


def test_with_real_pipeline():
    """Test hallucination detection with actual RAG pipeline."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('TESTING WITH REAL RAG PIPELINE')
    print('=' * 70)

    # Setup pipeline
    config = load_config('configs/default.yaml')
    config.openai_api_key = api_key
    config.vector_store.persist_directory = None

    pipeline = RAGPipeline(config)
    pipeline.ingest_directory('eval_data/documents')

    checker = GroundednessChecker(api_key=api_key)

    # Test queries
    queries = [
        "What was TechCorp's Q3 2024 revenue?",
        "What is the CEO's name at TechCorp?",  # Not in docs - might hallucinate
        "How much does CloudScale storage cost?",
    ]

    for query in queries:
        print(f'\n🔍 Query: {query}')
        print('-' * 50)

        # Get RAG answer
        result = pipeline.query(query)

        # Check groundedness
        groundedness = checker.check(
            question=query,
            answer=result.answer,
            context_chunks=result.retrieval.contexts,
        )

        status = '✅ GROUNDED' if groundedness.is_grounded else '⚠️ POSSIBLY HALLUCINATED'

        print(f'Answer: {result.answer[:100]}...')
        print(f'Status: {status}')
        print(f'Confidence: {groundedness.confidence}')

        if groundedness.unsupported_claims:
            print(f'⚠️ Unsupported claims: {groundedness.unsupported_claims}')


def test_citation_extraction():
    """Test citation extraction."""

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('❌ OPENAI_API_KEY not set')
        return

    print('\n' + '=' * 70)
    print('CITATION EXTRACTION TEST')
    print('=' * 70)

    extractor = CitationExtractor(api_key=api_key)

    chunks = [
        "TechCorp Q3 2024 revenue was $1.15 billion.",
        "The company has 12,500 employees globally.",
        "CloudScale storage costs $0.023 per GB.",
    ]

    answer = "TechCorp had revenue of $1.15 billion in Q3 and employs 12,500 people."

    citations = extractor.extract(answer, chunks)

    print(f'\nAnswer: {answer}')
    print(f'\nCitations:')
    for c in citations:
        source = f"Chunk {c['source_chunk']+1}" if c['supported'] else "UNSUPPORTED"
        print(f'  - "{c["claim"]}" → {source}')


if __name__ == '__main__':
    test_groundedness_checker()
    test_with_real_pipeline()
    test_citation_extraction()
