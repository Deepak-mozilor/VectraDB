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
  <a href="#search-algorithms">Algorithms</a> •
  <a href="#rest-api-reference">API Reference</a> •
  <a href="#python-client">Python Client</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust" />
  <a href="https://github.com/Amrithesh-Kakkoth/VectraDB"><img src="https://img.shields.io/github/stars/Amrithesh-Kakkoth/VectraDB" alt="Stars" /></a>
  <a href="https://deepwiki.com/Amrithesh-Kakkoth/VectraDB"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" /></a>
</p>

---

A high-performance vector database built in Rust. Store, search, and manage high-dimensional vectors with sub-millisecond query times.

VectraDB is designed for AI/ML workloads like semantic search, recommendation engines, and RAG (Retrieval-Augmented Generation) pipelines. It provides a REST API, gRPC API, and Python client out of the box.

## What is a Vector Database?

Modern AI models (like OpenAI, Cohere, or Hugging Face transformers) convert text, images, and other data into **vectors** -- arrays of numbers that capture semantic meaning. Similar items have similar vectors. A vector database lets you:

1. **Store** millions of these vectors with metadata
2. **Search** for the most similar vectors to a query (nearest neighbor search)
3. **Retrieve** the original data associated with each vector

For example, you could store document embeddings and then search for documents similar to a user's question -- this is how RAG systems work.

## Features

- **4 Search Algorithms**: HNSW, LSH, Product Quantization, and ES4D (our custom algorithm)
- **Dual API**: REST (HTTP) and gRPC running concurrently
- **Persistent Storage**: Data survives restarts via Sled embedded database
- **Python Client**: Sync and async gRPC client library
- **Text Chunking**: Built-in utilities for splitting documents, code, and markdown
- **Docker Ready**: Multi-stage Dockerfile included

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) 1.70 or later
- [Protocol Buffers compiler](https://grpc.io/docs/protoc-installation/) (`protoc`)

**Install protoc:**

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt install protobuf-compiler

# Windows — download from https://github.com/protocolbuffers/protobuf/releases
```

### Build and Run

```bash
# Clone the repository
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB

# Build in release mode
cargo build --release

# Start the server (both HTTP and gRPC)
./target/release/vectradb-server --enable-grpc
```

The server starts with these defaults:
- HTTP REST API: `http://localhost:8080`
- gRPC API: `localhost:50051`
- Vector dimension: 384
- Search algorithm: HNSW
- Data directory: `./vectradb_data`

### Your First Vectors

Once the server is running, try these commands in a new terminal:

```bash
# 1. Check the server is running
curl http://localhost:8080/health

# 2. Store a vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "tags": {"title": "Hello World", "source": "demo"}
  }'

# 3. Store another vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc2",
    "vector": [0.15, 0.22, 0.28, 0.41, 0.52],
    "tags": {"title": "Similar Document", "source": "demo"}
  }'

# 4. Search for similar vectors
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.12, 0.21, 0.29, 0.42, 0.51], "top_k": 5}'

# 5. Get a specific vector
curl http://localhost:8080/vectors/doc1

# 6. View database stats
curl http://localhost:8080/stats

# 7. Delete a vector
curl -X DELETE http://localhost:8080/vectors/doc1
```

> **Note:** The vector dimension in your requests must match the server's configured dimension (default: 384). The examples above use 5-dimensional vectors for brevity -- start the server with `-d 5` to use these examples, or use 384-dimensional vectors.

### Server Configuration

```bash
./target/release/vectradb-server [OPTIONS]

Options:
  -d, --dimension <DIM>          Vector dimension [default: 384]
  -D, --data-dir <DIR>           Data directory [default: ./vectradb_data]
  -p, --port <PORT>              HTTP port [default: 8080]
      --grpc-port <PORT>         gRPC port [default: 50051]
      --enable-grpc              Enable gRPC server [default: true]
  -a, --algorithm <ALGO>         Search algorithm: hnsw, lsh, pq, es4d [default: hnsw]
      --max-connections <N>      HNSW max connections [default: 16]
      --search-ef <N>            HNSW search ef [default: 50]
      --construction-ef <N>      HNSW construction ef [default: 200]
      --shard-length <N>         ES4D shard length for DET [default: 64]
      --auto-flush               Flush to disk after each write [default: true]
```

## Search Algorithms

VectraDB supports four search algorithms. Choose based on your needs:

| Algorithm | Speed | Memory | Accuracy | Best For |
|-----------|-------|--------|----------|----------|
| **HNSW** | Fast | High | High | General purpose (default) |
| **ES4D** | Fast | High | Exact | High-dimensional vectors, when you need exact results |
| **LSH** | Medium | Low | Approximate | Large datasets with memory constraints |
| **PQ** | Fast | Very Low | Approximate | Huge datasets, when memory is critical |

### HNSW (default)

Hierarchical Navigable Small World graph. Best balance of speed and accuracy for most workloads.

```bash
./target/release/vectradb-server -a hnsw --max-connections 16 --construction-ef 200
```

### ES4D

Our implementation of the [ES4D paper](https://doi.org/10.1109/ICCD56317.2022.00051), adapted to use HNSW graph navigation. Adds three optimizations on top of HNSW:

- **DET (Dimension-Level Early Termination)**: Computes distance in chunks. If partial distance already exceeds the cutoff, skips the rest -- saving CPU on high-dimensional vectors.
- **Dimension Reordering**: Puts high-variance dimensions first so DET triggers earlier.
- **CET (Cluster-Level Early Termination)**: Pre-clusters vectors and skips entire clusters that can't contain results.

```bash
./target/release/vectradb-server -a es4d --shard-length 64
```

### LSH (Locality Sensitive Hashing)

Hash-based approximate search. Uses random hyperplane hashing to group similar vectors.

```bash
./target/release/vectradb-server -a lsh --num-hashes 10 --num-buckets 1000
```

### PQ (Product Quantization)

Compresses vectors into compact codes for memory-efficient search. Good when your dataset is too large for RAM.

```bash
./target/release/vectradb-server -a pq
```

## REST API Reference

All endpoints return JSON. Error responses include an `error` and `message` field.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Database statistics |
| `POST` | `/vectors` | Create a vector |
| `GET` | `/vectors` | List all vector IDs |
| `GET` | `/vectors/:id` | Get a vector by ID |
| `PUT` | `/vectors/:id` | Update a vector |
| `DELETE` | `/vectors/:id` | Delete a vector |
| `PUT` | `/vectors/:id/upsert` | Create or update a vector |
| `POST` | `/search` | Search for similar vectors |

### Create a Vector

```
POST /vectors
Content-Type: application/json

{
  "id": "my-vector-1",
  "vector": [0.1, 0.2, 0.3, ...],
  "tags": {
    "category": "article",
    "author": "jane"
  }
}
```

- `id` (string, required): Unique identifier
- `vector` (array of floats, required): Must match the configured dimension
- `tags` (object, optional): Key-value metadata

### Search for Similar Vectors

```
POST /search
Content-Type: application/json

{
  "vector": [0.1, 0.2, 0.3, ...],
  "top_k": 10
}
```

- `vector` (array of floats, required): Query vector
- `top_k` (integer, optional): Number of results to return (default: 10, max: 10000)

**Response:**

```json
{
  "results": [
    {
      "id": "doc1",
      "score": 0.95,
      "metadata": {
        "id": "doc1",
        "dimension": 384,
        "created_at": 1700000000,
        "updated_at": 1700000000,
        "tags": {"category": "article"}
      }
    }
  ],
  "total_time_ms": 0.42
}
```

## gRPC API

The gRPC API provides the same functionality with better performance and type safety. See [`proto/vectradb.proto`](proto/vectradb.proto) for the full schema.

### Testing with grpcurl

```bash
# Install grpcurl: https://github.com/fullstorydev/grpcurl

# List available services
grpcurl -plaintext localhost:50051 list

# Health check
grpcurl -plaintext localhost:50051 vectradb.VectraDb/HealthCheck

# Create a vector
grpcurl -plaintext -d '{
  "id": "test1",
  "vector": [0.1, 0.2, 0.3],
  "tags": {"type": "test"}
}' localhost:50051 vectradb.VectraDb/CreateVector

# Search
grpcurl -plaintext -d '{
  "vector": [0.1, 0.2, 0.3],
  "top_k": 5
}' localhost:50051 vectradb.VectraDb/SearchSimilar
```

## Python Client

VectraDB includes a Python gRPC client for easy integration with Python applications.

### Setup

```bash
cd python-client
pip install grpcio grpcio-tools protobuf
python generate_proto.py   # Generate gRPC stubs
pip install -e .
```

### Usage

```python
from vectradb_simple import VectraDB

# Connect (server must be running with --enable-grpc)
with VectraDB(host="localhost", port=50051) as client:
    # Store vectors
    client.create("doc1", [0.1, 0.2, 0.3], {"type": "article"})
    client.create("doc2", [0.2, 0.3, 0.4], {"type": "article"})

    # Search
    results = client.search([0.15, 0.25, 0.35], k=10)
    for r in results.results:
        print(f"  {r.id}: score={r.score:.4f}")

    # Get stats
    stats = client.stats()
    print(f"Total vectors: {stats.total_vectors}")

    # CRUD operations
    vec = client.get("doc1")
    client.update("doc1", [0.11, 0.21, 0.31], {"type": "updated"})
    client.delete("doc2")
```

See [`python-client/README.md`](python-client/README.md) for the full Python client documentation.

## Using the Rust Library

You can also use VectraDB as a Rust library in your own projects:

```rust
use vectradb_components::{VectorDatabase, InMemoryVectorDB};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an in-memory database
    let mut db = InMemoryVectorDB::new();

    // Insert a vector
    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    db.create_vector("doc1".to_string(), vector, None)?;

    // Search for similar vectors
    let query = Array1::from_vec(vec![1.1, 2.1, 3.1]);
    let results = db.search_similar(query, 5)?;

    for result in &results {
        println!("{}: score={:.4}", result.id, result.score);
    }

    Ok(())
}
```

## Docker

```bash
# Build the image
docker build -t vectradb .

# Run with default settings
docker run -p 8080:8080 -p 50051:50051 vectradb

# Run with custom settings and persistent data
docker run -p 8080:8080 -p 50051:50051 \
  -v ./data:/data \
  vectradb --enable-grpc -d 384 -D /data -a hnsw
```

## Project Structure

```
VectraDB/
├── src/
│   ├── components/       Core types, similarity math, vector operations
│   ├── search/           Search algorithms (HNSW, LSH, PQ, ES4D)
│   ├── storage/          Sled-based persistent storage
│   ├── api/              Axum REST API handlers
│   ├── server/           Server binary (HTTP + gRPC)
│   └── chunkers/         Text chunking utilities
├── proto/                Protocol Buffer definitions
├── python-client/        Python gRPC client library
├── src_py/               PyO3 native Python bindings
├── bench/                Benchmarking scripts
├── .github/workflows/    CI/CD (build, test, release, Docker)
└── Cargo.toml            Workspace configuration
```

For a detailed architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Benchmarks

Typical results on an 8-core CPU with 32 GB RAM:

| Metric | gRPC | REST |
|--------|------|------|
| Search throughput (dim=64, k=10, N=50k) | 5,000-8,000 req/s | 1,000-2,000 req/s |
| p95 latency (concurrency=200) | < 20 ms | < 50 ms |

See [BENCHMARKS.md](BENCHMARKS.md) for how to reproduce these numbers.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB
cargo build
cargo test
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/Amrithesh-Kakkoth/VectraDB)
- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](ARCHITECTURE.md#api-layer)
- [Benchmarks](BENCHMARKS.md)
- [Contributing](CONTRIBUTING.md)
- [Python Client](python-client/README.md)
