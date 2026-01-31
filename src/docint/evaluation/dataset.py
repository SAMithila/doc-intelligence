"""
Evaluation dataset management.

Handles Q&A pairs for evaluating RAG performance.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import json


@dataclass
class EvalQuestion:
    """
    Single evaluation question with ground truth.
    
    Attributes:
        question: The query text
        relevant_chunk_ids: IDs of chunks that should be retrieved
        expected_answer: Optional expected answer text
        metadata: Additional info (difficulty, category, etc.)
    """
    question: str
    relevant_chunk_ids: set[str]
    expected_answer: str | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "expected_answer": self.expected_answer,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvalQuestion":
        return cls(
            question=data["question"],
            relevant_chunk_ids=set(data["relevant_chunk_ids"]),
            expected_answer=data.get("expected_answer"),
            metadata=data.get("metadata", {}),
        )


class EvalDataset:
    """
    Collection of evaluation questions.
    
    Supports loading from JSON and iterating over questions.
    """
    
    def __init__(self, questions: list[EvalQuestion] | None = None):
        self._questions = questions or []
    
    def add(self, question: EvalQuestion) -> None:
        """Add a question to the dataset."""
        self._questions.append(question)
    
    def __len__(self) -> int:
        return len(self._questions)
    
    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self._questions)
    
    def __getitem__(self, idx: int) -> EvalQuestion:
        return self._questions[idx]
    
    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "version": "1.0",
            "questions": [q.to_dict() for q in self._questions],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "EvalDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        questions = [
            EvalQuestion.from_dict(q) 
            for q in data.get("questions", [])
        ]
        
        return cls(questions)
    
    @classmethod
    def from_simple_format(
        cls, 
        qa_pairs: list[dict],
    ) -> "EvalDataset":
        """
        Create dataset from simple Q&A format.
        
        Expected format:
        [
            {
                "question": "What is X?",
                "chunk_ids": ["id1", "id2"],
                "answer": "X is..."  # optional
            }
        ]
        """
        questions = []
        for item in qa_pairs:
            questions.append(EvalQuestion(
                question=item["question"],
                relevant_chunk_ids=set(item.get("chunk_ids", [])),
                expected_answer=item.get("answer"),
            ))
        return cls(questions)
