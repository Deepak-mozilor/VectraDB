# Vector Database Benchmark Results

**Config:** 10,000 vectors | 768 dimensions | 100 queries | top-10 | L2 (Euclidean) distance

| Database   |   Insert (vec/s) |   Insert Total (s) |   p50 (ms) |   p95 (ms) |   p99 (ms) | Recall@10   |
|------------|------------------|--------------------|------------|------------|------------|-------------|
| VectraDB   |    883.1         |             11.324 |      5.739 |     11.898 |     12.967 | 89.10%      |
| FAISS      |      2.63109e+06 |              0.004 |      0.665 |      0.943 |      1.34  | 100.00%     |
| Chroma     |   7034.6         |              1.422 |      0.839 |      1.202 |      1.469 | 42.90%      |
| Qdrant     |   5377.2         |              1.86  |     15.769 |     17.328 |     18.288 | 100.00%     |

---

**Notes:**

- All vectors are unit-normalized random vectors
- FAISS uses `IndexFlatL2` (brute-force) -- the exact-search baseline
- Chroma and Qdrant use HNSW internally with default parameters
- VectraDB uses HNSW with `ef = max(k*2, 10)` (hardcoded); search_ef CLI flag is not yet wired into queries
- VectraDB is benchmarked over HTTP REST API (one request per vector), which adds significant network + serialization overhead to both insert and search
- FAISS, Chroma, and Qdrant run in-process (no network overhead)
- Recall is computed against brute-force L2 ground truth
