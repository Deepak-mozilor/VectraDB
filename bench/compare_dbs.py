#!/usr/bin/env python3
"""
Benchmark: VectraDB vs FAISS vs Chroma vs Qdrant

Compares insert throughput, search latency, and recall@k
using synthetic normalized vectors with L2 (Euclidean) distance.

VectraDB uses Euclidean distance internally, so all DBs are configured
for L2 to keep the comparison fair. Ground truth is also computed with L2.
"""

import argparse
import json
import os
import shutil
import sys
import time
import tempfile

import numpy as np
import requests
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_dataset(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate n normalized float32 vectors of given dimension."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def brute_force_topk_l2(data: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Ground-truth top-k by L2 distance (smallest distance = best match)."""
    diffs = data - query[np.newaxis, :]
    dists = np.sum(diffs ** 2, axis=1)
    return np.argsort(dists)[:k].tolist()


def compute_recall(predicted: list[int], ground_truth: list[int]) -> float:
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)


def percentile_ms(latencies: list[float], p: int) -> float:
    return np.percentile(latencies, p) * 1000


# ---------------------------------------------------------------------------
# VectraDB (REST API)
# ---------------------------------------------------------------------------

class VectraDBBench:
    name = "VectraDB"

    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/")
        self.session = requests.Session()

    def setup(self, dim: int):
        pass  # server already running

    def insert(self, vectors: np.ndarray) -> float:
        url = f"{self.url}/vectors"
        t0 = time.perf_counter()
        for i, vec in enumerate(vectors):
            payload = {
                "id": f"v-{i}",
                "vector": vec.tolist(),
                "tags": {"bench": "true"},
            }
            r = self.session.post(url, json=payload)
            r.raise_for_status()
        return time.perf_counter() - t0

    def search(self, query: np.ndarray, k: int) -> list[int]:
        r = self.session.post(
            f"{self.url}/search",
            json={"vector": query.tolist(), "top_k": k},
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [int(res["id"].split("-")[1]) for res in results]

    def cleanup(self):
        try:
            ids_resp = self.session.get(f"{self.url}/vectors")
            if ids_resp.ok:
                ids = ids_resp.json()
                if isinstance(ids, list):
                    for vid in ids:
                        self.session.delete(f"{self.url}/vectors/{vid}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FAISS (flat L2 — exact baseline)
# ---------------------------------------------------------------------------

class FAISSBench:
    name = "FAISS"

    def __init__(self):
        self.index = None

    def setup(self, dim: int):
        import faiss
        self.index = faiss.IndexFlatL2(dim)

    def insert(self, vectors: np.ndarray) -> float:
        t0 = time.perf_counter()
        self.index.add(vectors)
        return time.perf_counter() - t0

    def search(self, query: np.ndarray, k: int) -> list[int]:
        D, I = self.index.search(query.reshape(1, -1), k)
        return I[0].tolist()

    def cleanup(self):
        self.index = None


# ---------------------------------------------------------------------------
# Chroma (HNSW with L2)
# ---------------------------------------------------------------------------

class ChromaBench:
    name = "Chroma"

    def __init__(self):
        self.client = None
        self.collection = None

    def setup(self, dim: int):
        import chromadb
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("bench")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="bench",
            metadata={"hnsw:space": "l2"},
        )

    def insert(self, vectors: np.ndarray) -> float:
        batch_size = 5000
        t0 = time.perf_counter()
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            ids = [f"v-{i}" for i in range(start, end)]
            embeds = vectors[start:end].tolist()
            self.collection.add(ids=ids, embeddings=embeds)
        return time.perf_counter() - t0

    def search(self, query: np.ndarray, k: int) -> list[int]:
        results = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
        )
        return [int(rid.split("-")[1]) for rid in results["ids"][0]]

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection("bench")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Qdrant (in-memory, HNSW with Euclid)
# ---------------------------------------------------------------------------

class QdrantBench:
    name = "Qdrant"

    def __init__(self):
        self.client = None

    def setup(self, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(":memory:")
        self.client.recreate_collection(
            collection_name="bench",
            vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
        )
        self.dim = dim

    def insert(self, vectors: np.ndarray) -> float:
        from qdrant_client.models import PointStruct

        batch_size = 1000
        t0 = time.perf_counter()
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            points = [
                PointStruct(id=i, vector=vectors[i].tolist())
                for i in range(start, end)
            ]
            self.client.upsert(collection_name="bench", points=points)
        return time.perf_counter() - t0

    def search(self, query: np.ndarray, k: int) -> list[int]:
        results = self.client.query_points(
            collection_name="bench",
            query=query.tolist(),
            limit=k,
        ).points
        return [p.id for p in results]

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection("bench")
                self.client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(bench, data: np.ndarray, queries: np.ndarray,
                  ground_truth: list[list[int]], k: int, dim: int) -> dict:
    print(f"\n{'='*50}")
    print(f"  Benchmarking: {bench.name}")
    print(f"{'='*50}")

    # Setup
    bench.setup(dim)

    # Insert
    print(f"  Inserting {len(data)} vectors...", end=" ", flush=True)
    insert_time = bench.insert(data)
    insert_rate = len(data) / insert_time
    print(f"{insert_time:.2f}s ({insert_rate:,.0f} vec/s)")

    # Search
    print(f"  Running {len(queries)} searches (top-{k})...", end=" ", flush=True)
    latencies = []
    all_results = []
    for q in queries:
        t0 = time.perf_counter()
        results = bench.search(q, k)
        latencies.append(time.perf_counter() - t0)
        all_results.append(results)
    print("done")

    # Recall
    recalls = [compute_recall(pred, gt) for pred, gt in zip(all_results, ground_truth)]
    avg_recall = np.mean(recalls)

    result = {
        "name": bench.name,
        "insert_time_s": round(insert_time, 3),
        "insert_rate": round(insert_rate, 1),
        "p50_ms": round(percentile_ms(latencies, 50), 3),
        "p95_ms": round(percentile_ms(latencies, 95), 3),
        "p99_ms": round(percentile_ms(latencies, 99), 3),
        "avg_recall": round(avg_recall, 4),
    }

    print(f"  Results: p50={result['p50_ms']}ms  p95={result['p95_ms']}ms  "
          f"recall@{k}={result['avg_recall']:.2%}")

    # Cleanup
    bench.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(description="Vector DB Benchmark")
    parser.add_argument("--n", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--queries", type=int, default=100, help="Number of search queries")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    parser.add_argument("--vectradb-url", default="http://127.0.0.1:8080",
                        help="VectraDB REST API URL")
    parser.add_argument("--skip", nargs="*", default=[], help="DBs to skip")
    args = parser.parse_args()

    print(f"Benchmark Configuration:")
    print(f"  Vectors:   {args.n:,}")
    print(f"  Dimension: {args.dim}")
    print(f"  Queries:   {args.queries}")
    print(f"  Top-k:     {args.k}")
    print(f"  Metric:    L2 (Euclidean)")
    print()

    # Generate dataset
    print("Generating dataset...", end=" ", flush=True)
    data = gen_dataset(args.n, args.dim, seed=42)
    queries = gen_dataset(args.queries, args.dim, seed=99)
    print("done")

    # Ground truth via brute-force L2
    print("Computing ground truth (brute-force L2)...", end=" ", flush=True)
    ground_truth = [brute_force_topk_l2(data, q, args.k) for q in queries]
    print("done")

    # Benchmarks to run
    benchmarks = []
    skip = [s.lower() for s in args.skip]

    if "vectradb" not in skip:
        try:
            r = requests.get(f"{args.vectradb_url}/health", timeout=3)
            if r.ok:
                benchmarks.append(VectraDBBench(args.vectradb_url))
            else:
                print("WARNING: VectraDB not reachable, skipping.")
        except Exception:
            print("WARNING: VectraDB not reachable, skipping.")

    if "faiss" not in skip:
        benchmarks.append(FAISSBench())

    if "chroma" not in skip:
        benchmarks.append(ChromaBench())

    if "qdrant" not in skip:
        benchmarks.append(QdrantBench())

    # Run
    results = []
    for bench in benchmarks:
        try:
            result = run_benchmark(bench, data, queries, ground_truth, args.k, args.dim)
            results.append(result)
        except Exception as e:
            print(f"  ERROR benchmarking {bench.name}: {e}")
            import traceback
            traceback.print_exc()

    # Output
    if not results:
        print("\nNo benchmarks completed.")
        return

    headers = ["Database", "Insert (vec/s)", "Insert Total (s)",
               "p50 (ms)", "p95 (ms)", "p99 (ms)", f"Recall@{args.k}"]
    rows = [
        [r["name"], f'{r["insert_rate"]:,.1f}', r["insert_time_s"],
         r["p50_ms"], r["p95_ms"], r["p99_ms"], f'{r["avg_recall"]:.2%}']
        for r in results
    ]

    print(f"\n{'='*70}")
    print(f"  RESULTS  ({args.n:,} vectors, dim={args.dim}, {args.queries} queries, top-{args.k})")
    print(f"{'='*70}\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Save markdown report
    report_path = os.path.join(os.path.dirname(__file__), "results.md")
    with open(report_path, "w") as f:
        f.write("# Vector Database Benchmark Results\n\n")
        f.write(f"**Config:** {args.n:,} vectors | {args.dim} dimensions | "
                f"{args.queries} queries | top-{args.k} | L2 (Euclidean) distance\n\n")
        f.write(tabulate(rows, headers=headers, tablefmt="github"))
        f.write("\n\n---\n\n")
        f.write("**Notes:**\n\n")
        f.write("- All vectors are unit-normalized random vectors\n")
        f.write("- FAISS uses `IndexFlatL2` (brute-force) -- the exact-search baseline\n")
        f.write("- Chroma and Qdrant use HNSW internally with default parameters\n")
        f.write("- VectraDB uses HNSW with `ef = max(k*2, 10)` (hardcoded); "
                "search_ef CLI flag is not yet wired into queries\n")
        f.write("- VectraDB is benchmarked over HTTP REST API (one request per vector), "
                "which adds significant network + serialization overhead to both insert and search\n")
        f.write("- FAISS, Chroma, and Qdrant run in-process (no network overhead)\n")
        f.write("- Recall is computed against brute-force L2 ground truth\n")

    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
