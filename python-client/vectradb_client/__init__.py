"""VectraDB Python Client Library

A Python client for VectraDB vector database using gRPC.
"""

from .client import VectraDBClient
from .async_client import AsyncVectraDBClient
from .types import Vector, SearchResult, DatabaseStats, VectraDBError

__version__ = "0.1.0"
__all__ = [
    "VectraDBClient",
    "AsyncVectraDBClient",
    "Vector",
    "SearchResult",
    "DatabaseStats",
    "VectraDBError",
]
