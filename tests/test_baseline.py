"""
Test script for baseline RAG pipeline.

Run: python -m tests.test_baseline
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docint.config import Config
from docint.pipeline import RAGPipeline
from docint.evaluation.metrics import evaluate_retrieval, aggregate_metrics


def test_baseline_rag():
    """Test the baseline RAG pipeline."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping integration test.")
        return False
    
    print("=" * 60)
    print("Testing Baseline RAG Pipeline (v0.1)")
    print("=" * 60)
    
    # Create config
    config = Config()
    config.openai_api_key = api_key
    config.vector_store.persist_directory = None  # In-memory for testing
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = RAGPipeline(config)
    print(f"   ✓ Pipeline initialized")
    
    # Ingest documents
    print("\n2. Ingesting documents...")
    docs_path = Path(__file__).parent.parent / "eval_data" / "documents"
    
    if not docs_path.exists():
        print(f"   ❌ Documents directory not found: {docs_path}")
        return False
    
    stats = pipeline.ingest_directory(docs_path)
    print(f"   ✓ Ingested {stats['documents']} documents, {stats['chunks']} chunks")
    
    if stats['errors']:
        print(f"   ⚠ Errors: {stats['errors']}")
    
    # Test queries
    print("\n3. Testing queries...")
    
    test_questions = [
        "What was TechCorp's Q3 2024 revenue?",
        "How much does CloudScale object storage cost?",
        "What is the employee count at TechCorp?",
        "What security certifications does CloudScale have?",
        "What was the SecureNet acquisition price?",
    ]
    
    for question in test_questions:
        print(f"\n   Q: {question}")
        result = pipeline.query(question)
        print(f"   A: {result.answer[:200]}...")
        print(f"   ⏱ Latency: {result.latency_ms['total_ms']:.0f}ms "
              f"(retrieval: {result.latency_ms['retrieval_ms']:.0f}ms, "
              f"generation: {result.latency_ms['generation_ms']:.0f}ms)")
        print(f"   📚 Sources: {len(result.retrieval.results)} chunks retrieved")
    
    # Print stats
    print("\n4. Pipeline Statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ Baseline test completed successfully!")
    print("=" * 60)
    
    return True


def test_metrics():
    """Test retrieval metrics calculation."""
    print("\n" + "=" * 60)
    print("Testing Retrieval Metrics")
    print("=" * 60)
    
    # Simulated retrieval results
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc6"}  # doc6 not retrieved
    
    metrics = evaluate_retrieval(retrieved, relevant, k_values=[1, 3, 5])
    
    print(f"\nRetrieved: {retrieved}")
    print(f"Relevant:  {relevant}")
    print(f"\n{metrics}")
    
    # Expected values
    expected = {
        "recall@1": 1/3,    # Found 1 of 3 relevant in top 1
        "recall@3": 2/3,    # Found 2 of 3 relevant in top 3
        "precision@1": 1.0, # 1 of 1 is relevant
        "precision@3": 2/3, # 2 of 3 are relevant
        "mrr": 1.0,         # First relevant at position 1
    }
    
    print("\nValidation:")
    print(f"   Recall@1:    {metrics.recall[1]:.4f} (expected: {expected['recall@1']:.4f})")
    print(f"   Recall@3:    {metrics.recall[3]:.4f} (expected: {expected['recall@3']:.4f})")
    print(f"   Precision@1: {metrics.precision[1]:.4f} (expected: {expected['precision@1']:.4f})")
    print(f"   Precision@3: {metrics.precision[3]:.4f} (expected: {expected['precision@3']:.4f})")
    print(f"   MRR:         {metrics.mrr:.4f} (expected: {expected['mrr']:.4f})")
    
    print("\n✓ Metrics test passed!")
    return True


if __name__ == "__main__":
    test_metrics()
    print()
    test_baseline_rag()
