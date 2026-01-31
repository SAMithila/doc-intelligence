"""
Document loaders for various file formats.

Baseline: Simple text file loader.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import hashlib


@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def doc_id(self) -> str:
        """Generate unique ID from content hash."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def __len__(self) -> int:
        return len(self.content)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """Load a single document from path."""
        pass
    
    @abstractmethod
    def load_directory(self, path: str | Path, pattern: str = "*") -> Iterator[Document]:
        """Load all matching documents from directory."""
        pass


class TextLoader(BaseLoader):
    """
    Simple text file loader.
    
    Handles .txt and .md files with UTF-8 encoding.
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md"}
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, path: str | Path) -> Document:
        """Load a text document."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        content = path.read_text(encoding=self.encoding)
        
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
            }
        )
    
    def load_directory(
        self, 
        path: str | Path, 
        pattern: str = "*"
    ) -> Iterator[Document]:
        """Load all text documents from directory."""
        path = Path(path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    yield self.load(file_path)
                except Exception as e:
                    # Log and skip problematic files
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue
