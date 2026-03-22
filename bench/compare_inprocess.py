#!/usr/bin/env python3
"""
In-Process Benchmark: VectraDB (PyO3) vs FAISS vs Chroma vs Qdrant

All databases run in-process — no HTTP overhead.
This is the fair apples-to-apples comparison.
"""

import os
import shutil
import sys
import time
import tempfile

import numpy as np
from tabulate import tabulate


def gen_dataset(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def brute_force_topk_l2(data, query, k):
    diffs = data - query[np.newaxis, :]
    dists = np.sum(diffs ** 2, axis=1)
    return np.argsort(dists)[:k].tolist()


def compute_recall(predicted, ground_truth):
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)


def percentile_ms(latencies, p):
    return np.percentile(latencies, p) * 1000


# ---------------------------------------------------------------------------
# VectraDB In-Process (PyO3)
# ---------------------------------------------------------------------------

class VectraDBInProcess:
    name = "VectraDB (in-process)"

    def __init__(self):
        self.db = None
        self.tmpdir = None

    def setup(self, dim):
        from vectradb_py import VectraDB
        self.tmpdir = tempfile.mkdtemp(prefix="vectradb_bench_")
        # Use high ef for near-perfect recall on CPU (faster than GPU for <50k vectors)
        self.db = VectraDB(
            data_dir=self.tmpdir,
            search_algorithm="hnsw",
            dimension=dim,
            search_ef=500,
        )

    def insert(self, vectors):
        ids = [f"v-{i}" for i in range(len(vectors))]
        vecs = [v.tolist() for v in vectors]
        t0 = time.perf_counter()
        self.db.batch_create(ids, vecs, None)
        return time.perf_counter() - t0

    def search(self, query, k):
        results = self.db.search_similar(query.tolist(), k)
        return [int(r.id.split("-")[1]) for r in results]

    def cleanup(self):
        self.db = None
        if self.tmpdir and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# FAISS Flat (brute-force exact baseline)
# ---------------------------------------------------------------------------

class FAISSFlat:
    name = "FAISS (flat)"

    def __init__(self):
        self.index = None

    def setup(self, dim):
        import faiss
        self.index = faiss.IndexFlatL2(dim)

    def insert(self, vectors):
        t0 = time.perf_counter()
        self.index.add(vectors)
        return time.perf_counter() - t0

    def search(self, query, k):
        D, I = self.index.search(query.reshape(1, -1), k)
        return I[0].tolist()

    def cleanup(self):
        self.index = None


# ---------------------------------------------------------------------------
# FAISS HNSW
# ---------------------------------------------------------------------------

class FAISSHNSW:
    name = "FAISS HNSW"

    def __init__(self):
        self.index = None

    def setup(self, dim):
        import faiss
        self.index = faiss.IndexHNSWFlat(dim, 16)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 200

    def insert(self, vectors):
        t0 = time.perf_counter()
        self.index.add(vectors)
        return time.perf_counter() - t0

    def search(self, query, k):
        D, I = self.index.search(query.reshape(1, -1), k)
        return I[0].tolist()

    def cleanup(self):
        self.index = None


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

class ChromaBench:
    name = "Chroma"

    def __init__(self):
        self.client = None
        self.collection = None

    def setup(self, dim):
        import chromadb
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("bench")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="bench", metadata={"hnsw:space": "l2"},
        )

    def insert(self, vectors):
        batch_size = 5000
        t0 = time.perf_counter()
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            ids = [f"v-{i}" for i in range(start, end)]
            embeds = vectors[start:end].tolist()
            self.collection.add(ids=ids, embeddings=embeds)
        return time.perf_counter() - t0

    def search(self, query, k):
        results = self.collection.query(
            query_embeddings=[query.tolist()], n_results=k,
        )
        return [int(rid.split("-")[1]) for rid in results["ids"][0]]

    def cleanup(self):
        if self.client:
            try:
                self.client.delete_collection("bench")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class QdrantBench:
    name = "Qdrant"

    def __init__(self):
        self.client = None

    def setup(self, dim):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.client = QdrantClient(":memory:")
        self.client.recreate_collection(
            collection_name="bench",
            vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
        )

    def insert(self, vectors):
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

    def search(self, query, k):
        results = self.client.query_points(
            collection_name="bench", query=query.tolist(), limit=k,
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
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(bench, data, queries, ground_truth, k, dim):
    print(f"\n{'='*50}")
    print(f"  {bench.name}")
    print(f"{'='*50}")

    bench.setup(dim)

    # Insert
    print(f"  Inserting {len(data)} vectors...", end=" ", flush=True)
    insert_time = bench.insert(data)
    insert_rate = len(data) / insert_time
    print(f"{insert_time:.2f}s ({insert_rate:,.0f} vec/s)")

    # Search
    print(f"  Searching {len(queries)} queries (top-{k})...", end=" ", flush=True)
    latencies = []
    all_results = []
    for q in queries:
        t0 = time.perf_counter()
        results = bench.search(q, k)
        latencies.append(time.perf_counter() - t0)
        all_results.append(results)
    print("done")

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

    print(f"  -> p50={result['p50_ms']}ms  p95={result['p95_ms']}ms  "
          f"recall@{k}={result['avg_recall']:.2%}")

    bench.cleanup()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="In-Process Vector DB Benchmark")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    print(f"In-Process Benchmark (NO HTTP overhead)")
    print(f"  Vectors:   {args.n:,}")
    print(f"  Dimension: {args.dim}")
    print(f"  Queries:   {args.queries}")
    print(f"  Top-k:     {args.k}")
    print(f"  Metric:    L2 (Euclidean)")
    print()

    data = gen_dataset(args.n, args.dim, seed=42)
    queries = gen_dataset(args.queries, args.dim, seed=99)

    print("Computing ground truth...", end=" ", flush=True)
    ground_truth = [brute_force_topk_l2(data, q, args.k) for q in queries]
    print("done")

    skip = [s.lower() for s in args.skip]
    benchmarks = []
    if "vectradb" not in skip:
        benchmarks.append(VectraDBInProcess())
    if "faiss" not in skip:
        benchmarks.append(FAISSFlat())
    if "faiss_hnsw" not in skip:
        benchmarks.append(FAISSHNSW())
    if "chroma" not in skip:
        benchmarks.append(ChromaBench())
    if "qdrant" not in skip:
        benchmarks.append(QdrantBench())

    results = []
    for bench in benchmarks:
        try:
            result = run_benchmark(bench, data, queries, ground_truth, args.k, args.dim)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("\nNo results.")
        return

    headers = ["Database", "Insert (vec/s)", "p50 (ms)", "p95 (ms)", "p99 (ms)", f"Recall@{args.k}"]
    rows = [
        [r["name"], f'{r["insert_rate"]:,.1f}', r["p50_ms"], r["p95_ms"], r["p99_ms"],
         f'{r["avg_recall"]:.2%}']
        for r in results
    ]

    print(f"\n{'='*70}")
    print(f"  IN-PROCESS RESULTS  ({args.n:,} vectors, dim={args.dim}, top-{args.k})")
    print(f"{'='*70}\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()


if __name__ == "__main__":
    main()
