# VectraDB - High-Performance Vector Database

A modern, high-performance vector database built in Rust with multiple indexing strategies and language bindings.

## Features

### Core Components
- **Vector Operations**: Complete CRUD operations (Create, Read, Update, Delete, Upsert)
- **Similarity Functions**: Cosine similarity, Euclidean distance, Manhattan distance, Dot product
- **Advanced Indexing**: HNSW, LSH, Product Quantization
- **Persistent Storage**: Sled-based persistent storage with automatic recovery
- **REST API**: Full-featured HTTP API with JSON endpoints
- **Python Bindings**: Native Python integration with PyO3

### Search Algorithms
- **HNSW (Hierarchical Navigable Small World)**: Fast approximate nearest neighbor search
- **LSH (Locality Sensitive Hashing)**: Efficient approximate search with hash-based indexing
- **Product Quantization (PQ)**: Memory-efficient compression with fast similarity search

## Architecture

```
VectraDB/
├── src/
│   ├── components/          # Core vector operations and data structures
│   ├── search/             # Search algorithms (HNSW, LSH, PQ)
│   ├── storage/            # Persistent storage layer
│   └── api/                # REST API server
├── src_py/                 # Python bindings
└── Cargo.toml             # Workspace configuration
```

## Quick Start

### Using the Rust API

```rust
use vectradb_components::{VectorDatabase, InMemoryVectorDB};
use ndarray::Array1;

// Create database
let mut db = InMemoryVectorDB::new();

// Insert vectors
let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
db.create_vector("doc1".to_string(), vector, None)?;

// Search for similar vectors
let query = Array1::from_vec(vec![1.1, 2.1, 3.1]);
let results = db.search_similar(query, 5)?;
```

### Using the REST API

```bash
# Start the server
cargo run --bin vectradb-server

# Create a vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "vector": [1.0, 2.0, 3.0], "tags": {"category": "example"}}'

# Search for similar vectors
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.1, 2.1, 3.1], "top_k": 5}'
```

### Using Python

```python
import vectradb_py

# Create database
db = vectradb_py.VectraDB(data_dir="./my_vectors", search_algorithm="hnsw")

# Insert vectors
db.create_vector("doc1", [1.0, 2.0, 3.0], {"category": "example"})

# Search for similar vectors
results = db.search_similar([1.1, 2.1, 3.1], top_k=5)
for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
```

## API Endpoints

### Vector Operations
- `POST /vectors` - Create a new vector
- `GET /vectors/{id}` - Get vector by ID
- `PUT /vectors/{id}` - Update existing vector
- `DELETE /vectors/{id}` - Delete vector
- `PUT /vectors/{id}/upsert` - Insert or update vector
- `GET /vectors` - List all vector IDs

### Search
- `POST /search` - Search for similar vectors

### System
- `GET /health` - Health check
- `GET /stats` - Database statistics

## Configuration

### Database Configuration

```rust
use vectradb_storage::{DatabaseConfig, SearchAlgorithm};

let config = DatabaseConfig {
    data_dir: "./vectradb_data".to_string(),
    search_algorithm: SearchAlgorithm::HNSW,
    index_config: SearchConfig {
        algorithm: SearchAlgorithm::HNSW,
        max_connections: 16,
        search_ef: 50,
        construction_ef: 200,
        m: 16,
        ef_construction: 200,
        num_hashes: 10,
        num_buckets: 1000,
    },
    auto_flush: true,
    cache_size: 1000,
};
```

## Performance

VectraDB is designed for high performance with:

- **Fast Search**: Sub-millisecond search times for HNSW with proper configuration
- **Memory Efficient**: PQ compression reduces memory usage by up to 90%
- **Concurrent Access**: Thread-safe operations with async support
- **Persistent Storage**: Crash-resistant with automatic recovery

## Building

### Prerequisites
- Rust 1.70+
- Python 3.8+ (for Python bindings)

#### Windows Setup
1. Install Microsoft Visual C++ Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Or install Visual Studio Community (free)
   - Make sure to include "C++ build tools" workload

#### Linux/macOS Setup
- Standard build tools are usually pre-installed
- Ubuntu/Debian: `sudo apt install build-essential`
- macOS: Install Xcode Command Line Tools: `xcode-select --install`

### Build Commands

```bash
# Build all components
cargo build --release

# Build Python bindings
cd src_py
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] GraphQL API support
- [ ] Distributed clustering
- [ ] GPU acceleration
- [ ] More similarity metrics
- [ ] Streaming ingestion
- [ ] Advanced filtering
- [ ] Monitoring and metrics
- [ ] Docker support
