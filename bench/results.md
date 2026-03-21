# Vector Database Benchmark Results

**Config:** 10,000 vectors | 768 dimensions | 100 queries | top-10 | L2 (Euclidean) distance

| Database   |   Insert (vec/s) |   Insert Total (s) |   p50 (ms) |   p95 (ms) |   p99 (ms) | Recall@10   |
|------------|------------------|--------------------|------------|------------|------------|-------------|
| VectraDB   |     934.7        |             10.699 |      5.56  |      6.079 |      6.555 | 88.30%      |
| FAISS      |       4.1544e+06 |              0.002 |      0.475 |      0.582 |      0.607 | 100.00%     |
| FAISS HNSW |    4420.6        |              2.262 |      0.729 |      1.135 |      1.239 | 71.70%      |
| Chroma     |    7839.7        |              1.276 |      0.724 |      0.855 |      0.918 | 43.20%      |
| Qdrant     |    5540.5        |              1.805 |     13.935 |     14.934 |     15.244 | 100.00%     |

---

**Notes:**

- All vectors are unit-normalized random vectors
- FAISS uses `IndexFlatL2` (brute-force) -- the exact-search baseline
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB uses HNSW with `ef = max(k*2, 10)` (hardcoded); search_ef CLI flag is not yet wired into queries
- VectraDB is benchmarked over HTTP REST API (one request per vector), which adds significant network + serialization overhead to both insert and search
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
