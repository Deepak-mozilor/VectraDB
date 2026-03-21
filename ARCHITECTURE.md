# Architecture

This document explains how VectraDB is structured, how data flows through the system, and how the major components interact. It is intended for contributors and anyone who wants to understand the internals.

## Overview

VectraDB is a Rust workspace with 7 crates. Each crate has a single responsibility:

```
┌─────────────────────────────────────────────────────────┐
│                      vectradb-server                     │
│               (binary: HTTP + gRPC server)               │
│                                                          │
│  ┌──────────────┐         ┌────────────────────────┐    │
│  │  vectradb-api │         │  gRPC service (tonic)   │    │
│  │  (Axum REST)  │         │                        │    │
│  └──────┬───────┘         └───────────┬────────────┘    │
│         │                             │                  │
│         └──────────┬──────────────────┘                  │
│                    ▼                                     │
│           ┌────────────────┐                             │
│           │ vectradb-storage│                             │
│           │  (Sled + Index) │                             │
│           └───────┬────────┘                             │
│                   │                                      │
│         ┌─────────┴──────────┐                           │
│         ▼                    ▼                            │
│  ┌──────────────┐   ┌──────────────┐                     │
│  │vectradb-search│   │  vectradb-   │                     │
│  │(HNSW/LSH/PQ/ │   │  components  │                     │
│  │    ES4D)      │   │ (types/math) │                     │
│  └──────────────┘   └──────────────┘                     │
└─────────────────────────────────────────────────────────┘

Standalone crates (not in the request path):
  vectradb-chunkers   — text splitting utilities
  vectradb-py         — PyO3 Python bindings
```

## Crate Responsibilities

### vectradb-components

The foundation crate. Defines all shared types and traits.

**Key types:**
- `VectorDocument` — a vector with its metadata (ID, dimension, timestamps, tags)
- `VectorMetadata` — ID, dimension, created_at, updated_at, tags
- `SimilarityResult` — search result with ID, score, and metadata
- `DatabaseStats` — total vectors, dimension, memory usage
- `VectraDBError` — error enum (DimensionMismatch, VectorNotFound, DuplicateVector, InvalidVector, DatabaseError)

**Key traits:**
- `VectorDatabase` — the main trait that storage backends implement (create, get, update, delete, upsert, search, list, stats)

**Modules:**
- `similarity` — cosine, Euclidean, Manhattan, dot product distance functions
- `vector_operations` — create/update/validate/normalize vectors
- `indexing` — LinearIndex and HashIndex (simple in-memory indexes)
- `storage` — InMemoryVectorDB (HashMap-based, used in tests)

### vectradb-search

Search algorithm implementations. Each algorithm implements the `AdvancedSearch` trait:

```rust
pub trait AdvancedSearch {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>>;
    fn insert(&mut self, document: VectorDocument) -> Result<()>;
    fn remove(&mut self, id: &str) -> Result<()>;
    fn update(&mut self, id: &str, document: VectorDocument) -> Result<()>;
    fn build_index(&mut self, documents: Vec<VectorDocument>) -> Result<()>;
    fn get_stats(&self) -> SearchStats;
}
```

**Algorithms:**

| Module | Struct | How it works |
|--------|--------|-------------|
| `hnsw.rs` | `HNSWIndex` | Navigable small-world graph. O(1) node lookup via HashMap. Greedy beam search with configurable ef parameter. |
| `es4d.rs` | `ES4DIndex` | HNSW + dimension-level early termination + k-means clustering for cluster-level pruning + dimension reordering by variance. |
| `lsh.rs` | `LSHIndex` | Random hyperplane hashing. Groups similar vectors into buckets. Falls back to linear scan when needed. |
| `pq.rs` | `PQIndex` | Splits vectors into subspaces, quantizes each with k-means codebooks. Stores compact byte codes. |

### vectradb-storage

The persistence layer. Wraps Sled (an embedded B-tree database) with a search index.

**`PersistentVectorDB`:**
- Stores vectors in a `vectors` Sled tree (serialized with bincode)
- Stores metadata in a `metadata` Sled tree
- Maintains an in-memory search index (`Box<dyn AdvancedSearch>`)
- On startup, rebuilds the search index from persisted data
- Implements the `VectorDatabase` trait

**Data flow for writes:**
1. Create the `VectorDocument`
2. Insert into the search index (in-memory)
3. Serialize and write to Sled (on-disk)
4. Optionally flush to disk

**Data flow for reads:**
- `get_vector` reads directly from Sled
- `search_similar` queries the in-memory search index, then fetches metadata from Sled

### vectradb-api

The REST API layer, built with Axum.

**Routes:**
- `GET /health` — returns `{"status": "healthy"}`
- `GET /stats` — returns database statistics
- `POST /vectors` — create a vector (validates input: non-empty, finite values)
- `GET /vectors/:id` — get vector by ID
- `PUT /vectors/:id` — update vector
- `DELETE /vectors/:id` — delete vector
- `PUT /vectors/:id/upsert` — insert or update
- `POST /search` — similarity search (top_k clamped to 1-10000)
- `GET /vectors` — list all vector IDs

**Shared state:** `Arc<RwLock<PersistentVectorDB>>` — allows concurrent reads, exclusive writes.

### vectradb-server

The binary crate that ties everything together.

- Parses CLI arguments with `clap`
- Creates the database configuration
- Starts the HTTP server (Axum) and gRPC server (Tonic) concurrently using `tokio::spawn` + `tokio::select!`
- Both servers share the same `Arc<RwLock<PersistentVectorDB>>`

### vectradb-chunkers

Standalone text splitting utilities for preparing documents for vectorization. Not used in the server request path.

**Chunkers:**
- `DocumentChunker` — paragraph/sentence-aware splitting
- `MarkdownChunker` — respects heading hierarchy, code blocks, lists
- `CodeChunker` — function/class-aware splitting with language detection
- `ProductionChunker` — adaptive strategy with quality scoring

All implement the `Chunker` trait. Use the `create_chunker("type")` factory function.

### vectradb-py

PyO3-based native Python bindings. Compiles to a `.so`/`.pyd` that can be imported directly in Python without a running server.

## Request Flow

Here's what happens when a client sends a search request:

```
Client                    Server                     Storage              Search Index
  │                         │                          │                      │
  │  POST /search           │                          │                      │
  │  {"vector":[...],       │                          │                      │
  │   "top_k": 10}          │                          │                      │
  │ ───────────────────────►│                          │                      │
  │                         │                          │                      │
  │                         │  validate vector         │                      │
  │                         │  (non-empty, finite)     │                      │
  │                         │                          │                      │
  │                         │  db.read().await          │                      │
  │                         │ ────────────────────────►│                      │
  │                         │                          │                      │
  │                         │                          │  index.search(q, k)  │
  │                         │                          │ ────────────────────►│
  │                         │                          │                      │
  │                         │                          │  HNSW graph traversal│
  │                         │                          │  with DET/CET        │
  │                         │                          │  (if ES4D)           │
  │                         │                          │                      │
  │                         │                          │  Vec<SearchResult>   │
  │                         │                          │ ◄────────────────────│
  │                         │                          │                      │
  │                         │                          │  fetch metadata      │
  │                         │                          │  from Sled           │
  │                         │                          │                      │
  │                         │  SearchResponse          │                      │
  │ ◄───────────────────────│                          │                      │
  │  {"results":[...],      │                          │                      │
  │   "total_time_ms":0.42} │                          │                      │
```

## Concurrency Model

- The database is wrapped in `Arc<RwLock<PersistentVectorDB>>`
- Reads (search, get, list, stats) acquire a read lock — multiple readers can proceed concurrently
- Writes (create, update, delete, upsert) acquire a write lock — exclusive access
- Both HTTP and gRPC handlers share the same lock, so writes from either API block reads on both

## Persistence Model

- **Sled** is an embedded, crash-safe B-tree database
- Data is written to Sled on every write operation
- With `auto_flush: true` (default), data is fsynced to disk after each write
- On server restart, all data is loaded from Sled and the search index is rebuilt in memory
- The search index itself is not persisted — it's reconstructed from the stored vectors

## ES4D Algorithm Details

ES4D adapts the [ES4D paper](https://doi.org/10.1109/ICCD56317.2022.00051) for in-memory HNSW search:

1. **Index construction:**
   - Compute global dimension variance and create a reordering (high-variance dimensions first)
   - K-means clustering with sqrt(n) clusters
   - Build HNSW graph with reordered vectors

2. **Search:**
   - Reorder query dimensions to match stored vectors
   - Seed HNSW search from the cluster closest to the query
   - During graph traversal:
     - **CET check**: before computing distance, check if the candidate's cluster boundary is farther than the current cutoff. If so, skip.
     - **DET check**: compute L2 distance in shards of `shard_length` dimensions. After each shard, if partial distance exceeds cutoff, terminate early.
   - Both checks only activate once we have k results (to avoid premature pruning)

This is most effective on high-dimensional vectors (384+) where full distance computation is expensive.
