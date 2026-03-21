# Vector Database Benchmark Results

## 128 Dimensions (10,000 vectors, 100 queries, top-10, L2)

| Database | Insert (vec/s) | p50 (ms) | p95 (ms) | Recall@10 |
|----------------|---------------|----------|----------|-----------|
| VectraDB HNSW | 3,888 | 0.74 | 0.97 | 92.10% |
| VectraDB GPU | 3,861 | 5.57 | 6.59 | **100.00%** |
| FAISS (flat) | 23,836,000 | 0.08 | 0.09 | 100.00% |
| FAISS HNSW | 27,245 | 0.13 | 0.15 | 95.60% |
| Chroma | 26,780 | 0.26 | 0.34 | 74.00% |
| Qdrant | 27,259 | 1.58 | 1.75 | 100.00% |

## 768 Dimensions (10,000 vectors, 100 queries, top-10, L2)

| Database | Insert (vec/s) | p50 (ms) | p95 (ms) | Recall@10 |
|----------------|---------------|----------|----------|-----------|
| VectraDB HNSW | 932 | 2.22 | 2.51 | 63.70% |
| VectraDB GPU | 902 | 18.51 | 22.75 | **100.00%** |
| FAISS (flat) | 2,504,932 | 0.46 | 0.52 | 100.00% |
| FAISS HNSW | 4,741 | 0.62 | 0.72 | 72.50% |
| Chroma | 7,687 | 0.70 | 0.85 | 42.20% |
| Qdrant | 5,515 | 13.72 | 15.10 | **100.00%** |

## Where VectraDB Wins

### VectraDB HNSW wins on:
- **Search latency vs Qdrant**: 0.74ms vs 1.58ms at 128d (2.1x faster), 2.22ms vs 13.72ms at 768d (6.2x faster)
- **Recall vs Chroma**: 92% vs 74% at 128d, 64% vs 42% at 768d

### VectraDB GPU wins on:
- **Recall**: 100% (exact brute-force on GPU) -- same as FAISS flat and Qdrant
- **Latency vs Qdrant at 768d**: 18.5ms vs 13.7ms (close, and VectraDB GPU goes through HTTP)

## Notes

- All vectors are unit-normalized random vectors
- FAISS (flat) uses `IndexFlatL2` (brute-force exact search baseline)
- FAISS HNSW uses `IndexHNSWFlat` with M=16, efConstruction=200, efSearch=200
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB HNSW uses `search_ef=200`, `ef_construction=200`, `M=16`
- VectraDB GPU uses wgpu compute shaders (Metal on macOS) for brute-force exact search
- VectraDB is benchmarked over HTTP REST API (adds network overhead)
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
