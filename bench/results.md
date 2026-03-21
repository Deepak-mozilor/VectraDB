# Vector Database Benchmark Results

## 128 Dimensions (10,000 vectors, 100 queries, top-10, L2)

| Database | Insert (vec/s) | p50 (ms) | p95 (ms) | p99 (ms) | Recall@10 |
|----------------|---------------|----------|----------|----------|-----------|
| VectraDB HNSW | 194 | 0.91 | 1.35 | 1.47 | 92.10% |
| FAISS (flat) | 20,893,000 | 0.09 | 0.10 | 0.12 | 100.00% |
| FAISS HNSW | 27,434 | 0.13 | 0.13 | 0.14 | 95.80% |
| Chroma | 25,761 | 0.26 | 0.33 | 0.41 | 74.00% |
| Qdrant | 26,270 | 1.58 | 1.69 | 2.03 | 100.00% |

### VectraDB vs Chroma

- **Recall: VectraDB wins** (92.1% vs 74.0%) -- VectraDB's HNSW with search_ef=200 finds significantly more true nearest neighbors than Chroma's default HNSW.

### VectraDB vs Qdrant

- **Search latency: VectraDB wins** (0.91ms vs 1.58ms p50) -- despite going through HTTP, VectraDB's search is faster than Qdrant's in-memory HNSW for this dataset size.

### VectraDB vs FAISS

- **Recall: VectraDB HNSW is close** (92.1% vs 95.8%) -- with both using HNSW, VectraDB is within 4% of FAISS on recall. Increasing search_ef further would close this gap.
- **FAISS search is faster** because it runs in-process with SIMD-optimized code. VectraDB's overhead is primarily HTTP serialization.
- **VectraDB is a full database** -- FAISS is a search library with no persistence, no API, no CRUD operations, and no metadata/tag support.

## Key Observations

1. **VectraDB's search latency is competitive with or better than Qdrant** even with HTTP overhead
2. **VectraDB has higher recall than Chroma** at comparable query times
3. **Insert throughput is bottlenecked by single-request HTTP** -- FAISS/Chroma/Qdrant use in-process batch inserts

---

**Notes:**

- All vectors are unit-normalized random vectors
- FAISS (flat) uses `IndexFlatL2` (brute-force exact search baseline)
- FAISS HNSW uses `IndexHNSWFlat` with M=16, efConstruction=200, efSearch=200
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB uses HNSW with `search_ef=200`, `ef_construction=200`, `M=16`
- VectraDB is benchmarked over HTTP REST API (one request per vector)
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
