<p align="center">
  <img src="docs/assets/banner.png" alt="VectraDB Banner" />
</p>

<h1 align="center">VectraDB</h1>

<p align="center">
  <strong>High-performance vector database built in Rust</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust" />
  <a href="https://github.com/Amrithesh-Kakkoth/VectraDB"><img src="https://img.shields.io/github/stars/Amrithesh-Kakkoth/VectraDB" alt="Stars" /></a>
  <a href="https://deepwiki.com/Amrithesh-Kakkoth/VectraDB"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" /></a>
</p>

---

A modern vector database with multiple indexing strategies (HNSW, LSH, PQ), dual API interfaces (REST + gRPC), and a Python client library.

## Features

### Core Components
- **Vector Operations**: Complete CRUD operations (Create, Read, Update, Delete, Upsert)
- **Similarity Functions**: Cosine similarity, Euclidean distance, Manhattan distance, Dot product
- **Advanced Indexing**: HNSW, LSH, Product Quantization
- **Persistent Storage**: Sled-based persistent storage with automatic recovery
- **Dual API**: REST API (HTTP) and gRPC for flexible integration
- **Python Client**: Full-featured Python gRPC client with sync/async support
- **Text Processing**: Document, Markdown, and code chunking utilities

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
│   ├── api/                # REST API handlers
│   ├── server/             # Server with HTTP + gRPC
│   └── chunkers/           # Text chunking utilities
├── proto/                  # Protocol buffer definitions
├── python-client/          # Python gRPC client library
├── src_py/                 # PyO3 Python bindings (alternative)
└── Cargo.toml             # Workspace configuration
```

## Quick Start

### Start the Server

The VectraDB server runs both HTTP (REST) and gRPC APIs concurrently:

```bash
# Start server with both HTTP and gRPC enabled
cargo run --bin vectradb-server -- --enable-grpc -d 384 -D ./vectradb_data

# Server will start on:
# - HTTP REST API: http://0.0.0.0:8080
# - gRPC API: 0.0.0.0:50051
```

**Common Options:**
- `--enable-grpc` - Enable gRPC server (disabled by default)
- `-d, --dimension <DIM>` - Vector dimension (default: 384)
- `-D, --data-dir <DIR>` - Data directory (default: ./vectradb_data)
- `-p, --port <PORT>` - HTTP port (default: 8080)
- `--grpc-port <PORT>` - gRPC port (default: 50051)
- `-a, --algorithm <ALGO>` - Search algorithm: hnsw, lsh, or pq (default: hnsw)

### Using the REST API

```bash
# Health check
curl http://localhost:8080/health

# Create a vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3],
    "tags": {"category": "example", "source": "demo"}
  }'

# Get a vector
curl http://localhost:8080/vectors/doc1

# Search for similar vectors
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10
  }'

# Get statistics
curl http://localhost:8080/stats

# Delete a vector
curl -X DELETE http://localhost:8080/vectors/doc1
```

### Using the Python Client (gRPC)

Install the Python client:

```bash
cd python-client
pip install grpcio grpcio-tools protobuf
python generate_proto.py  # Generate gRPC stubs
pip install -e .
```

Use the simple client:

```python
from vectradb_simple import VectraDB

# Connect to VectraDB
with VectraDB(host="localhost", port=50051) as client:
    # Create vectors
    client.create("doc1", [0.1, 0.2, 0.3], {"type": "example"})
    client.create("doc2", [0.2, 0.3, 0.4], {"type": "example"})
    
    # Search for similar vectors
    results = client.search([0.15, 0.25, 0.35], k=10)
    for result in results.results:
        print(f"ID: {result.id}, Score: {result.score:.4f}")
    
    # Get database statistics
    stats = client.stats()
    print(f"Total vectors: {stats.total_vectors}")
    print(f"Dimension: {stats.dimension}")
    
    # Get a specific vector
    vec = client.get("doc1")
    print(f"Vector: {vec.vector}")
    print(f"Tags: {vec.tags}")
    
    # Update a vector
    client.update("doc1", [0.1, 0.2, 0.3], {"type": "updated"})
    
    # Delete a vector
    client.delete("doc1")
```

Run the complete demo:

```bash
cd python-client
python complete_demo.py
```

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

## API Reference

### REST API Endpoints

#### Vector Operations
- `POST /vectors` - Create a new vector
  ```json
  {
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3],
    "tags": {"key": "value"}
  }
  ```

- `GET /vectors/{id}` - Get vector by ID
- `PUT /vectors/{id}` - Update existing vector
- `DELETE /vectors/{id}` - Delete vector
- `PUT /vectors/{id}/upsert` - Insert or update vector
- `GET /vectors` - List all vector IDs

#### Search
- `POST /search` - Search for similar vectors
  ```json
  {
    "vector": [0.1, 0.2, 0.3],
    "top_k": 10,
    "ef_search": 50
  }
  ```

#### System
- `GET /health` - Health check
- `GET /stats` - Database statistics

### gRPC API

The gRPC API provides the same functionality with better performance and type safety. See `proto/vectradb.proto` for the full schema.

**Available RPCs:**
- `CreateVector` - Create a new vector
- `GetVector` - Get vector by ID
- `UpdateVector` - Update existing vector
- `DeleteVector` - Delete vector
- `UpsertVector` - Insert or update vector
- `SearchSimilar` - Search for similar vectors
- `ListVectors` - List all vector IDs
- `GetStats` - Get database statistics
- `HealthCheck` - Health check

**Testing gRPC:**

```bash
# Using grpcurl (install from https://github.com/fullstorydev/grpcurl)
grpcurl -plaintext localhost:50051 list

# Health check
grpcurl -plaintext localhost:50051 vectradb.VectraDb/HealthCheck

# Create vector
grpcurl -plaintext -d '{
  "id": "test1",
  "vector": [0.1, 0.2, 0.3],
  "tags": {"type": "test"}
}' localhost:50051 vectradb.VectraDb/CreateVector
```

See `GRPC_TESTING.md` for more examples.

### Python Client API

```python
from vectradb_simple import VectraDB

client = VectraDB(host="localhost", port=50051)

# Methods available:
client.health_check()                    # Check server health
client.create(id, vector, tags)          # Create vector
client.get(id)                           # Get vector
client.update(id, vector, tags)          # Update vector
client.delete(id)                        # Delete vector
client.upsert(id, vector, tags)          # Upsert vector
client.search(vector, k)                 # Search similar vectors
client.list()                            # List all vector IDs
client.stats()                           # Get database stats
client.close()                           # Close connection
```

See `python-client/README.md` for detailed Python client documentation.

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
- **Concurrent Access**: Thread-safe operations with async/await support
- **Persistent Storage**: Crash-resistant with automatic recovery
- **Dual API**: Choose REST for simplicity or gRPC for performance
- **Efficient Protocol**: Protocol Buffers for compact binary serialization

### Benchmarks

Performance varies by configuration, but typical results on modern hardware:

- **HNSW Search**: ~0.5ms per query (10k vectors, dim=384)
- **Vector Insert**: ~1ms per vector (with persistence)
- **Concurrent Requests**: 1000+ req/s (HTTP), 5000+ req/s (gRPC)

## Building

### Prerequisites
- **Rust**: 1.70 or later
- **Python**: 3.8+ (for Python client)
- **Protocol Buffers**: `protoc` compiler (for gRPC)

#### Install protoc

**macOS:**
```bash
brew install protobuf
```

**Ubuntu/Debian:**
```bash
sudo apt install protobuf-compiler
```

**Windows:**
Download from https://github.com/protocolbuffers/protobuf/releases

#### Windows Setup (Rust)
1. Install Microsoft Visual C++ Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Or install Visual Studio Community (free)
   - Make sure to include "C++ build tools" workload

#### Linux/macOS Setup (Rust)
- Standard build tools are usually pre-installed
- Ubuntu/Debian: `sudo apt install build-essential`
- macOS: Install Xcode Command Line Tools: `xcode-select --install`

### Build Commands

```bash
# Build all Rust components
cargo build --release

# Build server only
cargo build --release --bin vectradb-server

# Run server
cargo run --bin vectradb-server -- --enable-grpc -d 384

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Python Client Setup

```bash
cd python-client

# Install dependencies
pip install grpcio grpcio-tools protobuf

# Generate gRPC stubs
python generate_proto.py

# Install in development mode
pip install -e .

# Run tests (requires server running)
pytest tests/ -v

# Run examples
python examples/basic_sync.py
python complete_demo.py
```

## Docker Support

Build and run with Docker:

```bash
# Build image
docker build -t vectradb:latest .

# Run container
docker run -p 8080:8080 -p 50051:50051 \
  -v ./data:/data \
  vectradb:latest --enable-grpc -d 384 -D /data
```

## Text Processing & Chunking

VectraDB includes utilities for chunking documents:

```rust
use vectradb_chunkers::{DocumentChunker, ChunkingStrategy};

let chunker = DocumentChunker::new(
    ChunkingStrategy::Semantic,
    500,  // max chunk size
    50    // overlap
);

let chunks = chunker.chunk_text("Your long document text...");
for chunk in chunks {
    // Process and vectorize each chunk
}
```

Supported chunking strategies:
- **Fixed**: Fixed-size chunks with overlap
- **Semantic**: Sentence-aware chunking
- **Markdown**: Markdown-aware (preserves headers)
- **Code**: Programming language-aware

## Use Cases

- **Semantic Search**: Find similar documents, questions, or content
- **RAG Systems**: Retrieval-Augmented Generation for LLMs
- **Recommendation Engines**: Content and product recommendations
- **Deduplication**: Find and remove duplicate content
- **Anomaly Detection**: Identify outliers in vector space
- **Clustering**: Group similar items together

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`cargo test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/VectraDB.git
cd VectraDB

# Build and test
cargo build
cargo test

# Format code
cargo fmt

# Run linter
cargo clippy
```

## Documentation

- **API Documentation**: See `GRPC_TESTING.md` for gRPC examples
- **Python Client**: See `python-client/README.md` and `python-client/QUICKSTART.md`
- **Examples**: Check the `examples/` directory for Rust examples
- **Python Examples**: Check `python-client/examples/` for Python usage

## Project Structure

```
VectraDB/
├── src/
│   ├── api/              # REST API handlers
│   ├── chunkers/         # Text chunking utilities
│   ├── components/       # Core vector operations
│   ├── search/           # HNSW, LSH, PQ algorithms
│   ├── server/           # HTTP + gRPC server
│   └── storage/          # Persistent storage
├── proto/                # Protocol buffer definitions
├── python-client/        # Python gRPC client
│   ├── examples/         # Python usage examples
│   ├── tests/            # Python test suite
│   └── vectradb_client/  # Client package
├── examples/             # Rust examples
├── .github/workflows/    # CI/CD pipelines
└── Cargo.toml           # Workspace configuration
```

## License

MIT License - see LICENSE file for details.

## Roadmap

### In Progress
- ✅ HTTP REST API
- ✅ gRPC API with Protocol Buffers
- ✅ Python client library (gRPC)
- ✅ HNSW, LSH, PQ indexing
- ✅ Persistent storage with Sled
- ✅ Text chunking utilities
- ✅ Docker support

### Planned Features
- [ ] Async Python client
- [ ] JavaScript/TypeScript client
- [ ] Streaming APIs for batch operations
- [ ] Advanced filtering and metadata queries
- [ ] Distributed clustering and sharding
- [ ] GPU acceleration for similarity search
- [ ] More distance metrics (Hamming, Jaccard, etc.)
- [ ] Monitoring dashboard and metrics
- [ ] GraphQL API
- [ ] Vector compression techniques
- [ ] Incremental index updates
- [ ] Backup and restore utilities

## Community & Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/yourusername/VectraDB/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/yourusername/VectraDB/discussions)
- **Documentation**: Check the `docs/` folder and inline code comments

## Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [Tokio](https://tokio.rs/) - Async runtime
- [Axum](https://github.com/tokio-rs/axum) - Web framework
- [Tonic](https://github.com/hyperium/tonic) - gRPC framework
- [Sled](https://github.com/spacejam/sled) - Embedded database
- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays

## Citation

If you use VectraDB in your research, please cite:

```bibtex
@software{vectradb2025,
  title = {VectraDB: High-Performance Vector Database},
  author = {Amrithesh Kakkoth},
  year = {2025},
  url = {https://github.com/Amrithesh-Kakkoth/VectraDB}
}
```

---

**Made with ❤️ by the VectraDB team**
