# VectraDB Python Client

Python client library for VectraDB vector database. This client uses gRPC to communicate with the Rust backend server.

## Installation

```bash
pip install vectradb-client
```

Or install from source:

```bash
cd python-client
pip install -e .
```

## Quick Start

```python
from vectradb_client import VectraDBClient

# Connect to the VectraDB server
client = VectraDBClient(host="localhost", port=50051)

# Create a vector
vector_id = client.create_vector(
    id="vec1",
    values=[0.1, 0.2, 0.3],
    metadata={"type": "example", "count": 42}
)

# Search for similar vectors
results = client.search(
    query=[0.1, 0.2, 0.3],
    k=10,
    ef_search=100
)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance}")

# Get database statistics
stats = client.get_stats()
print(f"Total vectors: {stats.total_vectors}")

# Clean up
client.close()
```

## Async Support

```python
from vectradb_client import AsyncVectraDBClient

async def main():
    async with AsyncVectraDBClient(host="localhost", port=50051) as client:
        # Create a vector
        await client.create_vector(
            id="vec1",
            values=[0.1, 0.2, 0.3],
            metadata={"type": "example"}
        )
        
        # Search
        results = await client.search(query=[0.1, 0.2, 0.3], k=10)
        for result in results:
            print(f"ID: {result.id}, Distance: {result.distance}")
```

## API Reference

### VectraDBClient

#### Methods

- `create_vector(id: str, values: List[float], metadata: Optional[Dict[str, Any]] = None) -> str`
  - Create a new vector with the given ID and values
  
- `get_vector(id: str) -> Vector`
  - Retrieve a vector by ID
  
- `update_vector(id: str, values: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None) -> bool`
  - Update an existing vector's values and/or metadata
  
- `delete_vector(id: str) -> bool`
  - Delete a vector by ID
  
- `upsert_vector(id: str, values: List[float], metadata: Optional[Dict[str, Any]] = None) -> str`
  - Create or update a vector (upsert operation)
  
- `search(query: List[float], k: int = 10, ef_search: Optional[int] = None) -> List[SearchResult]`
  - Search for k nearest neighbors of the query vector
  
- `list_vectors(limit: Optional[int] = None, offset: Optional[int] = None) -> List[Vector]`
  - List all vectors in the database with pagination
  
- `get_stats() -> DatabaseStats`
  - Get database statistics (total vectors, memory usage, etc.)
  
- `health_check() -> bool`
  - Check if the server is healthy and responding

## Requirements

- Python 3.8+
- VectraDB server running with gRPC enabled (`--enable-grpc` flag)

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

Format code:

```bash
black vectradb_client/
```

Type checking:

```bash
mypy vectradb_client/
```

## License

MIT License
