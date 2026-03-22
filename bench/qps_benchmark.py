#!/usr/bin/env python3
"""
QPS Benchmark: VectraDB vs FAISS vs Qdrant

Measures:
- QPS (queries per second) under concurrent load
- Latency percentiles: p50, p95, p99
- Throughput scaling with concurrency (1, 10, 50, 100, 200 clients)
"""

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
from tabulate import tabulate


def gen_dataset(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def percentiles(latencies):
    a = np.array(latencies) * 1000  # to ms
    return {
        "p50": round(np.percentile(a, 50), 3),
        "p95": round(np.percentile(a, 95), 3),
        "p99": round(np.percentile(a, 99), 3),
        "mean": round(np.mean(a), 3),
    }


# ---------------------------------------------------------------------------
# VectraDB QPS (HTTP async)
# ---------------------------------------------------------------------------

async def vectradb_qps(url, queries, k, concurrency, duration_s=10):
    """Fire queries at VectraDB for `duration_s` seconds with `concurrency` workers."""
    results = []
    stop = asyncio.Event()
    n_queries = len(queries)
    counter = {"i": 0}

    async def worker(session):
        while not stop.is_set():
            idx = counter["i"] % n_queries
            counter["i"] += 1
            payload = {"vector": queries[idx].tolist(), "top_k": k}
            t0 = time.perf_counter()
            try:
                async with session.post(f"{url}/search", json=payload) as resp:
                    await resp.read()
                    results.append(time.perf_counter() - t0)
            except Exception:
                pass

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(concurrency)]
        await asyncio.sleep(duration_s)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    return results


# ---------------------------------------------------------------------------
# FAISS QPS (in-process, threaded)
# ---------------------------------------------------------------------------

def faiss_qps_bench(data, queries, k, concurrency, duration_s=10, use_hnsw=False):
    import faiss

    dim = data.shape[1]
    if use_hnsw:
        index = faiss.IndexHNSWFlat(dim, 16)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 200
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(data)

    results = []
    stop = threading.Event()
    n_queries = len(queries)
    lock = threading.Lock()
    counter = {"i": 0}

    def worker():
        while not stop.is_set():
            with lock:
                idx = counter["i"] % n_queries
                counter["i"] += 1
            q = queries[idx].reshape(1, -1)
            t0 = time.perf_counter()
            index.search(q, k)
            results.append(time.perf_counter() - t0)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join()

    return results


# ---------------------------------------------------------------------------
# Qdrant QPS (in-process, threaded)
# ---------------------------------------------------------------------------

def qdrant_qps_bench(data, queries, k, concurrency, duration_s=10):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(":memory:")
    dim = data.shape[1]
    client.recreate_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
    )
    # Insert in batches
    batch_size = 1000
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        points = [PointStruct(id=i, vector=data[i].tolist()) for i in range(start, end)]
        client.upsert(collection_name="bench", points=points)

    results = []
    stop = threading.Event()
    n_queries = len(queries)
    lock = threading.Lock()
    counter = {"i": 0}

    def worker():
        while not stop.is_set():
            with lock:
                idx = counter["i"] % n_queries
                counter["i"] += 1
            t0 = time.perf_counter()
            client.query_points(
                collection_name="bench",
                query=queries[idx].tolist(),
                limit=k,
            )
            results.append(time.perf_counter() - t0)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join()

    client.delete_collection("bench")
    client.close()
    return results


# ---------------------------------------------------------------------------
# VectraDB in-process QPS (threaded via PyO3)
# ---------------------------------------------------------------------------

def vectradb_inprocess_qps(data, queries, k, concurrency, duration_s=10):
    from vectradb_py import VectraDB

    dim = data.shape[1]
    tmpdir = tempfile.mkdtemp(prefix="vectradb_qps_")
    db = VectraDB(data_dir=tmpdir, search_algorithm="hnsw", dimension=dim, search_ef=500)
    ids = [f"v-{i}" for i in range(len(data))]
    vecs = [data[i].tolist() for i in range(len(data))]
    db.batch_create(ids, vecs, None)

    results = []
    stop = threading.Event()
    n_queries = len(queries)
    lock = threading.Lock()
    counter = {"i": 0}

    def worker():
        while not stop.is_set():
            with lock:
                idx = counter["i"] % n_queries
                counter["i"] += 1
            q = queries[idx].tolist()
            t0 = time.perf_counter()
            db.search_similar(q, k)
            results.append(time.perf_counter() - t0)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join()

    db = None
    shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def run_qps(name, bench_fn, concurrency_levels, duration_s):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    all_results = []
    for c in concurrency_levels:
        latencies = bench_fn(c, duration_s)
        qps = len(latencies) / duration_s
        p = percentiles(latencies)
        print(f"  concurrency={c:4d}  QPS={qps:8,.0f}  "
              f"p50={p['p50']:.2f}ms  p95={p['p95']:.2f}ms  p99={p['p99']:.2f}ms")
        all_results.append({
            "name": name, "concurrency": c, "qps": round(qps, 1),
            "p50": p["p50"], "p95": p["p95"], "p99": p["p99"],
            "total_queries": len(latencies),
        })
    return all_results


def main():
    parser = argparse.ArgumentParser(description="QPS Benchmark")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--duration", type=int, default=10, help="Seconds per concurrency level")
    parser.add_argument("--vectradb-url", default="http://127.0.0.1:8085")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    concurrency_levels = [1, 10, 50, 100]

    print(f"QPS Benchmark")
    print(f"  Vectors:      {args.n:,}")
    print(f"  Dimension:    {args.dim}")
    print(f"  Queries pool: {args.queries}")
    print(f"  Top-k:        {args.k}")
    print(f"  Duration:     {args.duration}s per concurrency level")
    print(f"  Concurrency:  {concurrency_levels}")
    print()

    data = gen_dataset(args.n, args.dim, seed=42)
    queries = gen_dataset(args.queries, args.dim, seed=99)

    all_results = []
    skip = [s.lower() for s in args.skip]

    # VectraDB in-process
    if "vectradb_inprocess" not in skip:
        try:
            r = run_qps(
                "VectraDB (in-process)",
                lambda c, d: vectradb_inprocess_qps(data, queries, args.k, c, d),
                concurrency_levels, args.duration,
            )
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # FAISS HNSW
    if "faiss" not in skip:
        r = run_qps(
            "FAISS HNSW",
            lambda c, d: faiss_qps_bench(data, queries, args.k, c, d, use_hnsw=True),
            concurrency_levels, args.duration,
        )
        all_results.extend(r)

    # Qdrant
    if "qdrant" not in skip:
        try:
            r = run_qps(
                "Qdrant",
                lambda c, d: qdrant_qps_bench(data, queries, args.k, c, d),
                concurrency_levels, args.duration,
            )
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # VectraDB HTTP server
    if "vectradb_http" not in skip:
        import requests
        try:
            requests.get(f"{args.vectradb_url}/health", timeout=2).raise_for_status()

            # Pre-load data via batch insert
            print(f"\n  Pre-loading {args.n} vectors into VectraDB server...")
            batch_size = 1000
            for start in range(0, args.n, batch_size):
                end = min(start + batch_size, args.n)
                batch = {
                    "vectors": [
                        {"id": f"v-{i}", "vector": data[i].tolist()}
                        for i in range(start, end)
                    ]
                }
                requests.post(f"{args.vectradb_url}/vectors/batch", json=batch).raise_for_status()
            print("  Done.")

            url = args.vectradb_url

            def http_bench(c, d):
                return asyncio.run(vectradb_qps(url, queries, args.k, c, d))

            r = run_qps("VectraDB (HTTP)", http_bench, concurrency_levels, args.duration)
            all_results.extend(r)
        except Exception as e:
            print(f"  VectraDB HTTP not reachable: {e}")

    # Summary table per concurrency level
    if all_results:
        for c in concurrency_levels:
            rows = [r for r in all_results if r["concurrency"] == c]
            if rows:
                print(f"\n--- Concurrency = {c} ---")
                headers = ["Database", "QPS", "p50 (ms)", "p95 (ms)", "p99 (ms)"]
                table = [
                    [r["name"], f'{r["qps"]:,.0f}', r["p50"], r["p95"], r["p99"]]
                    for r in rows
                ]
                print(tabulate(table, headers=headers, tablefmt="github"))

    # Save results
    report_path = os.path.join(os.path.dirname(__file__), "qps_results.md")
    with open(report_path, "w") as f:
        f.write(f"# QPS Benchmark Results\n\n")
        f.write(f"**Config:** {args.n:,} vectors | dim={args.dim} | top-{args.k} | "
                f"{args.duration}s per level\n\n")
        for c in concurrency_levels:
            rows = [r for r in all_results if r["concurrency"] == c]
            if rows:
                f.write(f"## Concurrency = {c}\n\n")
                headers = ["Database", "QPS", "p50 (ms)", "p95 (ms)", "p99 (ms)"]
                table = [
                    [r["name"], f'{r["qps"]:,.0f}', r["p50"], r["p95"], r["p99"]]
                    for r in rows
                ]
                f.write(tabulate(table, headers=headers, tablefmt="github"))
                f.write("\n\n")
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
