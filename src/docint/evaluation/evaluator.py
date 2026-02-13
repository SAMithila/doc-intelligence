"""
RAG System Evaluator.

Runs comprehensive evaluation with retrieval and answer metrics.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from docint.evaluation.metrics import (
    evaluate_retrieval,
    RetrievalMetrics,
)


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query_id: str
    category: str
    query: str
    expected_answer: str
    actual_answer: str
    keywords: list[str]
    keyword_matches: int
    keyword_total: int
    answer_correct: bool
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    retrieved_chunks: list[str]
    top_chunk_score: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    config_name: str
    total_queries: int
    results: list[QueryResult] = field(default_factory=list)

    # Aggregate metrics
    overall_accuracy: float = 0.0
    easy_accuracy: float = 0.0
    vague_accuracy: float = 0.0
    hard_accuracy: float = 0.0

    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0

    avg_keyword_match_rate: float = 0.0
    avg_top_chunk_score: float = 0.0

    def calculate_aggregates(self):
        """Calculate aggregate metrics from results."""
        if not self.results:
            return

        # Accuracy by category
        easy = [r for r in self.results if r.category == 'easy']
        vague = [r for r in self.results if r.category == 'vague']
        hard = [r for r in self.results if r.category == 'hard']

        self.overall_accuracy = sum(
            r.answer_correct for r in self.results) / len(self.results)
        self.easy_accuracy = sum(
            r.answer_correct for r in easy) / len(easy) if easy else 0
        self.vague_accuracy = sum(
            r.answer_correct for r in vague) / len(vague) if vague else 0
        self.hard_accuracy = sum(
            r.answer_correct for r in hard) / len(hard) if hard else 0

        # Latency
        self.avg_retrieval_time_ms = sum(
            r.retrieval_time_ms for r in self.results) / len(self.results)
        self.avg_generation_time_ms = sum(
            r.generation_time_ms for r in self.results) / len(self.results)
        self.avg_total_time_ms = sum(
            r.total_time_ms for r in self.results) / len(self.results)

        # Quality
        self.avg_keyword_match_rate = sum(
            r.keyword_matches / r.keyword_total for r in self.results if r.keyword_total > 0) / len(self.results)
        self.avg_top_chunk_score = sum(
            r.top_chunk_score for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'config_name': self.config_name,
            'total_queries': self.total_queries,
            'overall_accuracy': self.overall_accuracy,
            'easy_accuracy': self.easy_accuracy,
            'vague_accuracy': self.vague_accuracy,
            'hard_accuracy': self.hard_accuracy,
            'avg_retrieval_time_ms': self.avg_retrieval_time_ms,
            'avg_generation_time_ms': self.avg_generation_time_ms,
            'avg_total_time_ms': self.avg_total_time_ms,
            'avg_keyword_match_rate': self.avg_keyword_match_rate,
            'avg_top_chunk_score': self.avg_top_chunk_score,
            'results': [
                {
                    'query_id': r.query_id,
                    'category': r.category,
                    'query': r.query,
                    'expected_answer': r.expected_answer,
                    'actual_answer': r.actual_answer,
                    'answer_correct': r.answer_correct,
                    'keyword_matches': r.keyword_matches,
                    'keyword_total': r.keyword_total,
                    'retrieval_time_ms': r.retrieval_time_ms,
                    'generation_time_ms': r.generation_time_ms,
                    'top_chunk_score': r.top_chunk_score,
                }
                for r in self.results
            ]
        }

    def print_summary(self):
        """Print summary to console."""
        print('\n' + '=' * 70)
        print(f'EVALUATION REPORT: {self.config_name}')
        print('=' * 70)

        print(f'\n📊 ACCURACY')
        print(
            f'   Overall:  {self.overall_accuracy:.1%} ({sum(r.answer_correct for r in self.results)}/{len(self.results)})')
        print(f'   Easy:     {self.easy_accuracy:.1%}')
        print(f'   Vague:    {self.vague_accuracy:.1%}')
        print(f'   Hard:     {self.hard_accuracy:.1%}')

        print(f'\n⏱️  LATENCY')
        print(f'   Retrieval:   {self.avg_retrieval_time_ms:.0f}ms')
        print(f'   Generation:  {self.avg_generation_time_ms:.0f}ms')
        print(f'   Total:       {self.avg_total_time_ms:.0f}ms')

        print(f'\n📈 QUALITY')
        print(f'   Keyword match rate:  {self.avg_keyword_match_rate:.1%}')
        print(f'   Avg top chunk score: {self.avg_top_chunk_score:.3f}')

        # Show failures
        failures = [r for r in self.results if not r.answer_correct]
        if failures:
            print(f'\n❌ FAILURES ({len(failures)})')
            for f in failures[:5]:  # Show first 5
                print(f'   [{f.category}] {f.query}')
                print(f'      Expected: {f.expected_answer[:50]}')
                print(f'      Got: {f.actual_answer[:50]}')


class RAGEvaluator:
    """
    Evaluates RAG pipeline against labeled dataset.
    """

    def __init__(self, pipeline, config_name: str = "default"):
        self.pipeline = pipeline
        self.config_name = config_name

    def _check_keywords(self, answer: str, keywords: list[str]) -> tuple[int, int]:
        """Check how many keywords appear in answer."""
        answer_lower = answer.lower()
        matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return matches, len(keywords)

    def _is_correct(self, answer: str, keywords: list[str]) -> bool:
        """Determine if answer is correct based on keyword presence."""
        if not keywords:
            return True

        answer_lower = answer.lower()

        # Check for "don't know" / "not provided" responses
        negative_phrases = ['not provide', 'don\'t have',
                            'cannot find', 'no information']
        if any(phrase in answer_lower for phrase in negative_phrases):
            return False

        # At least one keyword must match
        matches, total = self._check_keywords(answer, keywords)
        return matches >= 1

    def evaluate(self, dataset_path: str) -> EvaluationReport:
        """
        Run evaluation on dataset.

        Args:
            dataset_path: Path to evaluation JSON file

        Returns:
            EvaluationReport with all metrics
        """
        # Load dataset
        with open(dataset_path) as f:
            dataset = json.load(f)

        queries = dataset['queries']

        report = EvaluationReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            config_name=self.config_name,
            total_queries=len(queries),
        )

        print(f'\nEvaluating {len(queries)} queries...\n')

        for i, q in enumerate(queries):
            # Run query
            start = time.time()
            result = self.pipeline.query(q['query'])
            total_time = (time.time() - start) * 1000

            # Extract metrics
            retrieval_time = result.latency_ms.get('retrieval_ms', 0)
            generation_time = result.latency_ms.get('generation_ms', 0)

            # Check correctness
            keywords = q.get('keywords', [])
            matches, total = self._check_keywords(result.answer, keywords)
            correct = self._is_correct(result.answer, keywords)

            # Top chunk score
            top_score = result.retrieval.results[0].score if result.retrieval.results else 0

            # Create result
            query_result = QueryResult(
                query_id=q['id'],
                category=q['category'],
                query=q['query'],
                expected_answer=q['expected_answer'],
                actual_answer=result.answer,
                keywords=keywords,
                keyword_matches=matches,
                keyword_total=total,
                answer_correct=correct,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                retrieved_chunks=[r.content[:50]
                                  for r in result.retrieval.results[:3]],
                top_chunk_score=top_score,
            )

            report.results.append(query_result)

            # Progress
            status = '✅' if correct else '❌'
            print(
                f'   [{i+1}/{len(queries)}] {status} [{q["category"]}] {q["query"][:40]}...')

        # Calculate aggregates
        report.calculate_aggregates()

        return report

    def save_report(self, report: EvaluationReport, output_path: str):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f'\nReport saved to: {output_path}')
