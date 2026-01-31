"""Evaluation framework for RAG system."""
from docint.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    evaluate_retrieval,
)
from docint.evaluation.dataset import EvalQuestion, EvalDataset

__all__ = [
    "recall_at_k",
    "precision_at_k", 
    "mrr",
    "ndcg_at_k",
    "evaluate_retrieval",
    "EvalQuestion",
    "EvalDataset",
]
