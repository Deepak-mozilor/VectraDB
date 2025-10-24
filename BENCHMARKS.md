# VectraDB Benchmarks (Throughput, Latency, Recall)

This guide shows how to reproduce end-to-end throughput numbers on a single machine and back up claims like:
- gRPC SearchSimilar: 5,000+ requests/sec
- REST /search: 1,000+ requests/sec

Numbers depend on hardware, OS, dataset size, and tuning. This doc provides a fair, repeatable method.

## Test profile
- Search-only (index preloaded and warmed)
- Dimension = 64, top_k = 10
- HNSW (default)
- Release build, minimal logging

## Prerequisites
- Rust stable toolchain
- Python 3.9+
- Load tools:
  - ghz (gRPC): https://github.com/bojand/ghz
  - hey (HTTP): https://github.com/rakyll/hey
- Optional: WSL2 on Windows for Linux tooling parity

## 1) Build and start server (release)
```bash
# Linux/WSL2 example; adjust paths for Windows
cd /mnt/d/proj/VectraDB
RUST_LOG=warn cargo build -p vectradb-server --release
rm -rf ./vectradb_data_bench64
./target/release/vectradb-server -p 8080 --grpc-port 50051 -d 64 -D ./vectradb_data_bench64
```
Notes:
- Use a release build and low log level to minimize overhead.
- Keep dimension modest (e.g., 64) and top_k small (e.g., 10) for RPS tests.

## 2) Preload dataset (N normalized vectors)
Use the provided Python script to insert N vectors via gRPC.
```bash
# New terminal
cd /mnt/d/proj/VectraDB
python bench/preload.py --host 127.0.0.1 --port 50051 --n 50000 --dim 64
```

## 3) Warm up (optional)
```bash
ghz --insecure \
  --proto ./proto/vectradb.proto \
  --call vectradb.VectraDb/SearchSimilar \
  -d '{"vector": [0.015,0.002, ... 64 dims ...], "top_k": 10}' \
  -c 50 -n 1000 127.0.0.1:50051
```

## 4) gRPC throughput (SearchSimilar)
```bash
# Example: 100k requests, concurrency 200
ghz --insecure \
  --proto ./proto/vectradb.proto \
  --call vectradb.VectraDb/SearchSimilar \
  -d '{"vector": [0.01,0.02, ... 64 dims ...], "top_k": 10}' \
  -c 200 -n 100000 127.0.0.1:50051
# Record: Requests/sec, mean/p95/p99 latency, errors (should be 0)
```

## 5) REST throughput (/search)
```bash
# Prepare payload.json (dim=64, top_k=10)
python - <<'PY'
import json, numpy as np
v=np.random.default_rng(2).standard_normal(64).astype('float32'); v/=np.linalg.norm(v)+1e-9
with open('payload.json','w') as f: json.dump({'vector': v.tolist(), 'top_k': 10}, f)
print('Wrote payload.json')
PY

# 10k requests, concurrency 100
hey -n 10000 -c 100 -m POST -H "Content-Type: application/json" \
  -D payload.json http://127.0.0.1:8080/search
# Record: Requests/sec, latency distribution, non-2xx count (should be 0)
```

## 6) Optional: recall@k vs brute-force
Use the evaluator script (or extend preload.py) to compute recall@k vs exact cosine kNN on a subset (e.g., nq=200).

## 7) Tips for stable results
- Use release build and RUST_LOG=warn (or error)
- Keep top_k small and dimension modest (64)
- Ensure index is warm (perform a warm-up run)
- Close background apps; set high-performance power plan
- Prefer Linux/WSL2 for load tools

## 8) Example guidepost (hardware dependent)
- 8C/16T CPU, 32GB RAM, Linux/WSL2
  - gRPC (k=10, dim=64, N=50k): 5,200–8,000 req/s
  - REST (k=10, dim=64, N=50k): 1,100–2,000 req/s
  - p95 typically sub-20ms at concurrency 200

## PowerShell automation (Windows)
A convenience script is provided at `bench/run-bench.ps1` to orchestrate build → start → preload → ghz/hey.

Run:
```powershell
# From repo root
powershell -ExecutionPolicy Bypass -File .\bench\run-bench.ps1 -Dim 64 -N 50000 -TopK 10 -Concurrency 200 -Requests 100000
```
