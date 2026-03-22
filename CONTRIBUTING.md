# Contributing to VectraDB

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- **Rust** 1.70+ ([install](https://rustup.rs/))
- **protoc** (Protocol Buffers compiler) — needed for gRPC
- **Git**

```bash
# Clone the repository
git clone https://github.com/Amrithesh-Kakkoth/VectraDB.git
cd VectraDB

# Build everything
cargo build

# Run all tests
cargo test -p vectradb-components -p vectradb-search -p vectradb-storage -p vectradb-chunkers -p vectradb-api

# Check formatting
cargo fmt --all -- --check

# Run linter (must pass with zero warnings)
cargo clippy --workspace -- -D warnings
```

## Making Changes

### 1. Create a branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make your changes

- Write code
- Add tests for new functionality
- Run `cargo fmt --all` before committing
- Run `cargo clippy --workspace -- -D warnings` and fix any warnings

### 3. Test your changes

```bash
# Run all tests
cargo test -p vectradb-components -p vectradb-search -p vectradb-storage -p vectradb-chunkers -p vectradb-api

# Run a specific crate's tests
cargo test -p vectradb-search

# Run a specific test
cargo test -p vectradb-search -- es4d::tests::test_es4d_insert_and_search
```

### 4. Submit a Pull Request

- Push your branch: `git push -u origin feature/your-feature-name`
- Open a Pull Request on GitHub
- Describe what you changed and why
- Link any related issues

## Code Style

- **Formatting**: Run `cargo fmt --all` before every commit. CI enforces this.
- **Linting**: `cargo clippy --workspace -- -D warnings` must pass with zero warnings.
- **Naming**: Use `snake_case` for functions/variables, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- **Error handling**: Return `Result` instead of panicking. Use `thiserror` for error types, `anyhow` for ad-hoc errors.
- **Unsafe code**: Avoid it. If absolutely necessary, document why it's safe.

## Project Structure

```
src/vectradb/     — In-process library API (start here for library usage)
src/components/   — Core types and traits (start here to understand internals)
src/search/       — Search algorithms (HNSW, ES4D, IVF, SQ, LSH, PQ, SIMD, Tensor)
src/storage/      — Persistent storage (Sled + search index)
src/api/          — REST API (Axum) + auth + rate limiting + metrics
src/server/       — Server binary (HTTP + gRPC + TLS)
src/embeddings/   — Embedding model providers (Ollama, OpenAI, HF, Cohere)
src/chunkers/     — Text chunking utilities
src/tfidf/        — TF-IDF sparse text retrieval
src/rag/          — RAG pipeline
src/eval/         — Evaluation framework
proto/            — Protocol Buffer definitions
tests/            — Integration & stress tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of how components interact.

## What to Work On

- Check [GitHub Issues](https://github.com/Amrithesh-Kakkoth/VectraDB/issues) for open tasks
- Issues labeled `good first issue` are great starting points
- The [Roadmap](#roadmap) section in the README lists planned features

### Ideas for Contributions

- **New distance metrics**: Hamming, Jaccard, etc.
- **Filtering**: Metadata-based filtering during search
- **Batch operations**: Bulk insert/delete endpoints
- **Streaming API**: gRPC streaming for large batch operations
- **JavaScript/TypeScript client**: Similar to the Python client
- **Monitoring**: Prometheus metrics endpoint
- **Benchmarks**: More comprehensive benchmarks with standard datasets (SIFT, GloVe, etc.)

## Adding a New Search Algorithm

1. Create a new file in `src/search/src/` (e.g., `my_algo.rs`)
2. Implement the `AdvancedSearch` trait
3. Add `pub mod my_algo;` and `pub use my_algo::MyAlgoIndex;` in `src/search/src/lib.rs`
4. Add a variant to the `SearchAlgorithm` enum in `lib.rs`
5. Add a match arm in `src/storage/src/lib.rs` → `PersistentVectorDB::new()`
6. Add CLI parsing in `src/server/src/main.rs`
7. Add tests
8. Update documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
