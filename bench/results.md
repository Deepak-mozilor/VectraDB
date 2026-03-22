# Vector Database Benchmark Results

**Config:** 10,000 vectors | 1536 dimensions | 100 queries | top-10 | L2 (Euclidean) distance

| Database   |   Insert (vec/s) |   Insert Total (s) |   p50 (ms) |   p95 (ms) |   p99 (ms) | Recall@10   |
|------------|------------------|--------------------|------------|------------|------------|-------------|
| VectraDB   |    512.6         |             19.509 |      8.62  |      9.035 |      9.398 | 83.60%      |
| FAISS      |      2.10228e+06 |              0.005 |      0.953 |      1.189 |      1.305 | 100.00%     |
| FAISS HNSW |   2199.6         |              4.546 |      1.078 |      1.176 |      1.275 | 64.80%      |

---

**Notes:**

- All vectors are unit-normalized random vectors
- FAISS uses `IndexFlatL2` (brute-force) -- the exact-search baseline
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB uses HNSW with `ef = max(k*2, 10)` (hardcoded); search_ef CLI flag is not yet wired into queries
- VectraDB is benchmarked over HTTP REST API (one request per vector), which adds significant network + serialization overhead to both insert and search
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
