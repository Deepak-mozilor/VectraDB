# Build
FROM rust:1-bookworm AS build
WORKDIR /app
COPY . .
RUN cargo build -p vectradb-server --release

# Runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/target/release/vectradb-server /usr/local/bin/vectradb-server
EXPOSE 8080 50051

# Default: 384-dim, HNSW, default data dir
ENTRYPOINT ["vectradb-server"]
