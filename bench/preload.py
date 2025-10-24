import argparse
import os
import sys
import time
import numpy as np

# Import Python client
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python-client"))
try:
    from vectradb_simple import VectraDB  # type: ignore
except Exception as e:
    print("Error: could not import python-client/vectradb_simple.")
    print("Make sure the python-client is present and importable. Error:", e)
    sys.exit(1)


def gen_dataset(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--prefix", default="bench")
    args = p.parse_args()

    X = gen_dataset(args.n, args.dim)
    client = VectraDB(host=args.host, port=args.port)

    t0 = time.time()
    for i in range(args.n):
        vid = f"{args.prefix}-{i}"
        client.create(vid, X[i].tolist(), {"bench": "true"})
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"Inserted {i+1}/{args.n} in {elapsed:.1f}s ({rate:.0f} vec/s)")
    total = time.time() - t0
    print(f"Done. Inserted {args.n} vectors in {total:.1f}s ({args.n/max(total,1e-6):.0f} vec/s)")


if __name__ == "__main__":
    main()
