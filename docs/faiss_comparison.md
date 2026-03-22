# VectraDB vs FAISS: Comprehensive Comparison

## Feature Comparison

| Feature | VectraDB | FAISS |
|---------|----------|-------|
| **Language** | Rust (with Python bindings) | C++ (with Python bindings) |
| **Index Types** | HNSW, LSH, PQ, ES4D | Flat, IVF, HNSW, PQ, IVF-PQ, ScaNN-like |
| **Mutable Index** | Yes — insert, update, delete on all algorithms | Limited — IVF supports add; HNSW has no delete |
| **Metadata Filtering** | Built-in boolean logic (AND/OR/NOT) during search | No native support; post-filter only |
| **Persistence** | Built-in via Sled KV store | Manual `write_index()` / `read_index()` |
| **API** | REST + gRPC server | Library (import and call) |
| **GPU Support** | Yes — wgpu (Metal/Vulkan/DX12) | Yes — CUDA only |
| **Hybrid Search** | Dense + TF-IDF with RRF/weighted fusion | Dense only |
| **RAG Built-in** | Yes — LLM integration, graph agent | No |
| **Authentication** | API key (admin/read-only roles) | N/A (library) |
| **Rate Limiting** | Per-IP token bucket | N/A (library) |
| **Metrics** | Prometheus endpoint | N/A |
| **TLS/HTTPS** | Yes | N/A |
| **Text Embeddings** | Built-in providers (OpenAI, Ollama, HF, Cohere) | N/A |
| **Evaluation Metrics** | Built-in (recall, precision, MRR, NDCG, MAP) | Limited (`evaluation.py` for recall) |
| **Deployment** | Single binary, Docker | Python package |
| **Distributed** | Single-node | Supports sharding via `IndexShards` |
| **Max Vectors (practical)** | ~1M (single node, memory-bound) | ~1B (with IVF+PQ compression) |

## Performance Comparison

Config: 10,000 vectors | 1536 dimensions | 100 queries | top-10 | L2 (Euclidean)

| Metric | VectraDB (HNSW) | FAISS Flat L2 | FAISS HNSW |
|--------|-----------------|---------------|------------|
| **Insert (vec/s)** | 512.6 | 2,102,280 | 2,199.6 |
| **Insert Total (s)** | 19.509 | 0.005 | 4.546 |
| **p50 Latency (ms)** | 8.62 | 0.953 | 1.078 |
| **p95 Latency (ms)** | 9.035 | 1.189 | 1.176 |
| **p99 Latency (ms)** | 9.398 | 1.305 | 1.275 |
| **Recall@10** | 83.60% | 100.00% | 64.80% |

> **Note:** VectraDB is measured via HTTP REST API (network overhead). FAISS runs in-process (no serialization or network). VectraDB's HNSW recall exceeds FAISS's HNSW for the same configuration because of search_ef tuning differences.

## When to Use Which

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Production microservice** | VectraDB | Built-in REST/gRPC, auth, rate limiting, persistence, monitoring |
| **Research / batch analytics** | FAISS | Fastest in-process search, supports billion-scale with IVF+PQ |
| **RAG applications** | VectraDB | Integrated embedding providers, TF-IDF hybrid search, LLM pipeline |
| **Real-time updates** | VectraDB | Full CRUD on all index types; FAISS HNSW has no delete |
| **GPU-heavy workloads (NVIDIA)** | FAISS | Mature CUDA support, GPU index types (GpuIndexFlat, GpuIndexIVF) |
| **Cross-platform GPU** | VectraDB | wgpu supports Metal (macOS), Vulkan (Linux/Win), DX12 (Win) |
| **Filtered search** | VectraDB | Native metadata filtering during search; FAISS requires post-filter |
| **Billion-scale** | FAISS | IVF+PQ compression stores vectors in bytes; sharded index support |
| **Edge/embedded** | VectraDB | Single Rust binary, Sled embedded DB, minimal dependencies |
| **Python ML pipeline** | FAISS | Native Python, integrates with NumPy/PyTorch directly |

## Similarity Score Reference

| Metric | Range | Perfect Match | Orthogonal | Interpretation |
|--------|-------|---------------|------------|----------------|
| **Cosine Similarity** | [-1, 1] | 1.0 | 0.0 | Higher = more similar |
| **Euclidean (transformed)** | (0, 1] | 1.0 | → 0 | `1/(1+dist)`, higher = closer |
| **Dot Product** | (-∞, ∞) | maximum | 0 | Higher = more similar |
| **Manhattan (transformed)** | (0, 1] | 1.0 | → 0 | `1/(1+dist)`, higher = closer |

## Architectural Differences

### VectraDB
```
Client → REST/gRPC API → Auth → Rate Limit → Search Index (HNSW/ES4D/LSH/PQ)
                                                    ↓
                                              Sled KV Store (persistence)
                                                    ↓
                                              GPU Reranking (optional)
```

### FAISS
```
Python Code → faiss.IndexHNSW / IndexIVFPQ → C++ Engine → (optional CUDA GPU)
                                                  ↓
                                            faiss.write_index() (manual persistence)
```

### Key Architectural Insight

VectraDB is a **database** — it manages state, handles concurrency (`Arc<RwLock>`), provides an API layer, and persists data automatically. FAISS is a **library** — it provides algorithms that you embed into your application and manage yourself.

This means VectraDB has higher per-query latency (network + serialization overhead) but provides operational features (monitoring, auth, persistence) that FAISS doesn't. For the highest raw throughput, FAISS wins. For production deployments with operational requirements, VectraDB provides a complete solution.
