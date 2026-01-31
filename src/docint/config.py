"""
Configuration for Document Intelligence RAG System.

Uses typed dataclasses for configuration with YAML loading support.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os
import yaml


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: Literal["fixed", "recursive", "semantic"] = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 50
    

@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    provider: Literal["openai", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    provider: Literal["chroma"] = "chroma"
    collection_name: str = "documents"
    persist_directory: str = "./data/chroma"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 5
    score_threshold: float = 0.0  # Minimum similarity score
    

@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    provider: Literal["openai"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: list[str] = field(default_factory=lambda: [
        "recall_at_k",
        "precision_at_k", 
        "mrr",
    ])
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class Config:
    """Main configuration container."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # API keys from environment
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            chunking=ChunkingConfig(**data.get("chunking", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            vector_store=VectorStoreConfig(**data.get("vector_store", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict
        
        data = {
            "chunking": asdict(self.chunking),
            "embedding": asdict(self.embedding),
            "vector_store": asdict(self.vector_store),
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
            "evaluation": asdict(self.evaluation),
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from file or return defaults."""
    if path and Path(path).exists():
        return Config.from_yaml(path)
    return Config()
