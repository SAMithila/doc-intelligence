"""
Retrieval evaluation metrics.

These metrics measure how well the retriever finds relevant documents.

Key Metrics:
- Recall@K: Fraction of relevant docs found in top K results
- Precision@K: Fraction of top K results that are relevant
- MRR: Mean Reciprocal Rank - where does first relevant doc appear?
- NDCG@K: Normalized Discounted Cumulative Gain - rank-weighted relevance
"""
from dataclasses import dataclass
import math


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Recall@K: What fraction of relevant documents did we find in top K?
    
    Formula: |retrieved ∩ relevant| / |relevant|
    
    Good for measuring: "Did we find all the relevant documents?"
    Range: [0, 1], higher is better
    """
    if not relevant_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    found = top_k & relevant_ids
    
    return len(found) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Precision@K: What fraction of top K results are relevant?
    
    Formula: |retrieved ∩ relevant| / K
    
    Good for measuring: "How much noise is in our results?"
    Range: [0, 1], higher is better
    """
    if k == 0:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    found = top_k & relevant_ids
    
    return len(found) / k


def mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Mean Reciprocal Rank: Where does the first relevant document appear?
    
    Formula: 1 / rank_of_first_relevant
    
    Good for measuring: "How quickly do we find something useful?"
    Range: [0, 1], higher is better
    
    If first relevant doc is at position 1 → MRR = 1.0
    If first relevant doc is at position 2 → MRR = 0.5
    If first relevant doc is at position 10 → MRR = 0.1
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at K.
    
    Relevance of each document is discounted by its position.
    Earlier positions matter more.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            # Binary relevance: 1 if relevant, 0 if not
            # Discount by log2(position + 1)
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
    return dcg


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Normalized DCG at K.
    
    DCG normalized by ideal DCG (if all relevant docs were at top).
    
    Good for measuring: "How well are relevant docs ranked?"
    Range: [0, 1], higher is better
    """
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    
    # Ideal DCG: all relevant docs at top positions
    ideal_relevant = min(len(relevant_ids), k)
    ideal_ids = list(relevant_ids)[:ideal_relevant]
    idcg = dcg_at_k(ideal_ids, relevant_ids, ideal_relevant)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    recall: dict[int, float]  # recall@k for each k
    precision: dict[int, float]  # precision@k for each k
    mrr: float
    ndcg: dict[int, float]  # ndcg@k for each k
    
    def __repr__(self) -> str:
        lines = ["RetrievalMetrics:"]
        lines.append(f"  MRR: {self.mrr:.4f}")
        for k in sorted(self.recall.keys()):
            lines.append(
                f"  @{k}: Recall={self.recall[k]:.4f}, "
                f"Precision={self.precision[k]:.4f}, "
                f"NDCG={self.ndcg[k]:.4f}"
            )
        return "\n".join(lines)


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] = [1, 3, 5, 10],
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics.
    
    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: Set of ground-truth relevant document IDs
        k_values: List of K values to compute metrics for
        
    Returns:
        RetrievalMetrics with all computed metrics
    """
    return RetrievalMetrics(
        recall={k: recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        precision={k: precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg={k: ndcg_at_k(retrieved_ids, relevant_ids, k) for k in k_values},
    )


def aggregate_metrics(
    all_metrics: list[RetrievalMetrics],
) -> RetrievalMetrics:
    """
    Aggregate metrics across multiple queries.
    
    Computes mean of each metric.
    """
    if not all_metrics:
        return RetrievalMetrics(
            recall={}, precision={}, mrr=0.0, ndcg={}
        )
    
    # Get all k values
    k_values = list(all_metrics[0].recall.keys())
    
    # Compute means
    mean_recall = {
        k: sum(m.recall[k] for m in all_metrics) / len(all_metrics)
        for k in k_values
    }
    mean_precision = {
        k: sum(m.precision[k] for m in all_metrics) / len(all_metrics)
        for k in k_values
    }
    mean_mrr = sum(m.mrr for m in all_metrics) / len(all_metrics)
    mean_ndcg = {
        k: sum(m.ndcg[k] for m in all_metrics) / len(all_metrics)
        for k in k_values
    }
    
    return RetrievalMetrics(
        recall=mean_recall,
        precision=mean_precision,
        mrr=mean_mrr,
        ndcg=mean_ndcg,
    )
