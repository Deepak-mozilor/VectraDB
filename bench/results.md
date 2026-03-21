# Vector Database Benchmark Results

**Config:** 10,000 vectors | 128 dimensions | 100 queries | top-10 | L2 (Euclidean) distance

| Database   |   Insert (vec/s) |   Insert Total (s) |   p50 (ms) |   p95 (ms) |   p99 (ms) | Recall@10   |
|------------|------------------|--------------------|------------|------------|------------|-------------|
| VectraDB   |   3858.6         |              2.592 |      0.7   |      0.855 |      1.037 | 91.80%      |
| FAISS      |      1.54063e+07 |              0.001 |      0.081 |      0.099 |      0.181 | 100.00%     |
| FAISS HNSW |  26111.9         |              0.383 |      0.132 |      0.146 |      0.161 | 95.50%      |
| Chroma     |  26566.9         |              0.376 |      0.246 |      0.32  |      0.39  | 76.50%      |
| Qdrant     |  27209.9         |              0.368 |      1.606 |      1.722 |      1.924 | 100.00%     |

---

**Notes:**

- All vectors are unit-normalized random vectors
- FAISS uses `IndexFlatL2` (brute-force) -- the exact-search baseline
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB uses HNSW with `ef = max(k*2, 10)` (hardcoded); search_ef CLI flag is not yet wired into queries
- VectraDB is benchmarked over HTTP REST API (one request per vector), which adds significant network + serialization overhead to both insert and search
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
