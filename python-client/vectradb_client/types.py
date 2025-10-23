"""Type definitions for VectraDB client."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


class VectraDBError(Exception):
    """Base exception for VectraDB client errors."""
    pass


@dataclass
class Vector:
    """Represents a vector with metadata."""
    
    id: str
    values: List[float]
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        values_preview = self.values[:3] if len(self.values) > 3 else self.values
        preview = f"{values_preview}..." if len(self.values) > 3 else values_preview
        return f"Vector(id='{self.id}', values={preview}, metadata={self.metadata})"


@dataclass
class SearchResult:
    """Represents a search result with distance."""
    
    id: str
    distance: float
    vector: Optional[Vector] = None
    
    def __repr__(self) -> str:
        return f"SearchResult(id='{self.id}', distance={self.distance:.4f})"


@dataclass
class DatabaseStats:
    """Database statistics."""
    
    total_vectors: int
    dimension: int
    index_type: str
    memory_usage_bytes: int
    
    @property
    def memory_usage_mb(self) -> float:
        """Return memory usage in megabytes."""
        return self.memory_usage_bytes / (1024 * 1024)
    
    def __repr__(self) -> str:
        return (
            f"DatabaseStats(total_vectors={self.total_vectors}, "
            f"dimension={self.dimension}, index_type='{self.index_type}', "
            f"memory_usage={self.memory_usage_mb:.2f}MB)"
        )
