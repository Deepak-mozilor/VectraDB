# Vector Database Benchmark Results

**Config:** 10,000 vectors | 128 dimensions | 100 queries | top-10 | L2 (Euclidean) distance

| Database   |   Insert (vec/s) |   Insert Total (s) |   p50 (ms) |   p95 (ms) |   p99 (ms) | Recall@10   |
|------------|------------------|--------------------|------------|------------|------------|-------------|
| VectraDB   |     196.6        |             50.873 |      0.825 |      1.178 |      1.405 | 92.30%      |
| FAISS      |       1.3462e+07 |              0.001 |      0.083 |      0.096 |      0.127 | 100.00%     |
| Chroma     |   26414.2        |              0.379 |      0.242 |      0.292 |      0.435 | 75.60%      |
| Qdrant     |   25814.6        |              0.387 |      1.56  |      1.683 |      2.014 | 100.00%     |

---

**Notes:**

- All vectors are unit-normalized random vectors
- FAISS uses `IndexFlatL2` (brute-force) -- the exact-search baseline
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB uses HNSW with `ef = max(k*2, 10)` (hardcoded); search_ef CLI flag is not yet wired into queries
- VectraDB is benchmarked over HTTP REST API (one request per vector), which adds significant network + serialization overhead to both insert and search
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
