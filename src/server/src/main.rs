use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;
use vectradb_search::{DistanceMetric, SearchAlgorithm};
use vectradb_storage::{DatabaseConfig, PersistentVectorDB};

mod grpc;
use grpc::VectraDbService;

/// VectraDB Server - High-performance vector database
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Data directory for persistent storage
    #[arg(short = 'D', long, default_value = "./vectradb_data")]
    data_dir: PathBuf,

    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// gRPC server port
    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    /// Enable gRPC server
    #[arg(long, default_value = "true")]
    enable_grpc: bool,

    /// Search algorithm (hnsw, lsh, pq)
    #[arg(short, long, default_value = "hnsw")]
    algorithm: String,

    /// Vector dimension
    #[arg(short = 'd', long, default_value = "384")]
    dimension: usize,

    /// Maximum connections for HNSW
    #[arg(long, default_value = "16")]
    max_connections: usize,

    /// Search ef parameter
    #[arg(long, default_value = "50")]
    search_ef: usize,

    /// Construction ef parameter
    #[arg(long, default_value = "200")]
    construction_ef: usize,

    /// Number of hash functions for LSH
    #[arg(long, default_value = "10")]
    num_hashes: usize,

    /// Number of buckets for LSH
    #[arg(long, default_value = "1000")]
    num_buckets: usize,

    /// Shard length for ES4D DET (dimension-level early termination)
    #[arg(long, default_value = "64")]
    shard_length: usize,

    /// Distance metric: euclidean, cosine, dot
    #[arg(long, default_value = "euclidean")]
    metric: String,

    /// Enable auto-flush
    #[arg(long, default_value = "true")]
    auto_flush: bool,

    /// Cache size
    #[arg(long, default_value = "1000")]
    cache_size: usize,

    /// Embedding provider: ollama, openai, huggingface, cohere (disabled if not set)
    #[arg(long)]
    embedding_provider: Option<String>,

    /// Embedding model name (e.g., nomic-embed-text, text-embedding-3-small)
    #[arg(long, default_value = "nomic-embed-text")]
    embedding_model: String,

    /// Embedding API URL (e.g., http://localhost:11434 for Ollama)
    #[arg(long)]
    embedding_url: Option<String>,

    /// Embedding API key (or use OPENAI_API_KEY / HF_API_KEY / COHERE_API_KEY env vars)
    #[arg(long)]
    embedding_api_key: Option<String>,

    /// Admin API key (full read+write access). Can be specified multiple times.
    /// Also reads from VECTRADB_API_KEY env var.
    #[arg(long)]
    api_key: Vec<String>,

    /// Read-only API key (search, get, list only). Can be specified multiple times.
    /// Also reads from VECTRADB_API_KEY_READONLY env var.
    #[arg(long)]
    api_key_readonly: Vec<String>,

    /// TLS certificate file (PEM format). Enables HTTPS when set with --tls-key.
    #[arg(long)]
    tls_cert: Option<PathBuf>,

    /// TLS private key file (PEM format). Enables HTTPS when set with --tls-cert.
    #[arg(long)]
    tls_key: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    env_logger::init();

    // Parse search algorithm
    let search_algorithm = match args.algorithm.to_lowercase().as_str() {
        "hnsw" => SearchAlgorithm::HNSW,
        "lsh" => SearchAlgorithm::LSH,
        "pq" => SearchAlgorithm::PQ,
        "es4d" => SearchAlgorithm::ES4D,
        _ => {
            eprintln!(
                "Invalid algorithm: {}. Supported algorithms: hnsw, lsh, pq, es4d",
                args.algorithm
            );
            std::process::exit(1);
        }
    };

    // Parse distance metric
    let metric = match args.metric.to_lowercase().as_str() {
        "euclidean" | "l2" => DistanceMetric::Euclidean,
        "cosine" => DistanceMetric::Cosine,
        "dot" | "dot_product" | "ip" => DistanceMetric::DotProduct,
        _ => {
            eprintln!(
                "Invalid metric: {}. Supported: euclidean, cosine, dot",
                args.metric
            );
            std::process::exit(1);
        }
    };

    // Create database configuration
    let config = DatabaseConfig {
        data_dir: args.data_dir.to_string_lossy().to_string(),
        search_algorithm,
        index_config: vectradb_search::SearchConfig {
            algorithm: search_algorithm,
            max_connections: args.max_connections,
            search_ef: args.search_ef,
            construction_ef: args.construction_ef,
            m: args.max_connections,
            ef_construction: args.construction_ef,
            num_hashes: args.num_hashes,
            num_buckets: args.num_buckets,
            dimension: Some(args.dimension),
            num_subspaces: Some(8),
            codes_per_subspace: Some(256),
            shard_length: Some(args.shard_length),
            metric,
        },
        auto_flush: args.auto_flush,
        cache_size: args.cache_size,
    };

    println!("Starting VectraDB server...");
    println!("Data directory: {}", config.data_dir);
    println!("Search algorithm: {:?}", config.search_algorithm);
    println!("Vector dimension: {}", args.dimension);
    println!("HTTP port: {}", args.port);
    if args.enable_grpc {
        println!("gRPC port: {}", args.grpc_port);
    }

    // Initialize database
    let db = PersistentVectorDB::new(config.clone()).await?;
    let db_arc = Arc::new(RwLock::new(db));

    // Initialize embedding provider (optional)
    let embedder: Option<std::sync::Arc<dyn vectradb_embeddings::EmbeddingProvider>> =
        if let Some(provider_name) = &args.embedding_provider {
            let emb_config = vectradb_embeddings::EmbeddingConfig {
                provider: provider_name.clone(),
                model: args.embedding_model.clone(),
                api_url: args.embedding_url.clone(),
                api_key: args.embedding_api_key.clone(),
                dimension: Some(args.dimension),
            };
            match vectradb_embeddings::create_provider(&emb_config) {
                Ok(provider) => {
                    println!(
                        "Embedding provider: {} (model: {})",
                        provider.provider_name(),
                        provider.model_name()
                    );
                    Some(std::sync::Arc::from(provider))
                }
                Err(e) => {
                    eprintln!("Failed to initialize embedding provider: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        };

    // Build auth config from CLI args + env vars
    let mut admin_keys = args.api_key;
    if let Ok(env_key) = std::env::var("VECTRADB_API_KEY") {
        admin_keys.push(env_key);
    }
    let mut readonly_keys = args.api_key_readonly;
    if let Ok(env_key) = std::env::var("VECTRADB_API_KEY_READONLY") {
        readonly_keys.push(env_key);
    }
    let auth_config = Arc::new(vectradb_api::AuthConfig::new(admin_keys, readonly_keys));
    if auth_config.enabled {
        println!("API key authentication: enabled");
    }

    // Initialize GPU engine (optional)
    #[cfg(feature = "gpu")]
    let gpu_engine: Option<Arc<vectradb_search::gpu::GpuDistanceEngine>> = {
        match vectradb_search::gpu::GpuDistanceEngine::new(100_000) {
            Some(engine) => {
                println!("GPU acceleration: enabled (wgpu)");
                Some(Arc::new(engine))
            }
            None => {
                println!("GPU acceleration: no adapter found, disabled");
                None
            }
        }
    };

    // TLS configuration
    let tls_config = match (&args.tls_cert, &args.tls_key) {
        (Some(cert), Some(key)) => {
            println!(
                "TLS: enabled (cert={}, key={})",
                cert.display(),
                key.display()
            );
            Some((cert.clone(), key.clone()))
        }
        (Some(_), None) | (None, Some(_)) => {
            eprintln!("Error: both --tls-cert and --tls-key must be provided together");
            std::process::exit(1);
        }
        (None, None) => {
            println!("TLS: disabled (use --tls-cert and --tls-key to enable)");
            None
        }
    };

    // Clone shared database for HTTP server
    let http_db = db_arc.clone();
    let http_embedder = embedder.clone();
    let http_auth = auth_config.clone();
    #[cfg(feature = "gpu")]
    let http_gpu = gpu_engine.clone();
    let http_port = args.port;
    let http_tls = tls_config.clone();

    // Start HTTP server task
    let http_handle = tokio::spawn(async move {
        let state = vectradb_api::AppState {
            db: http_db,
            embedder: http_embedder,
            auth: http_auth,
            #[cfg(feature = "gpu")]
            gpu: http_gpu,
        };
        let app = vectradb_api::create_router(state);
        let addr = format!("0.0.0.0:{}", http_port);

        if let Some((cert_path, key_path)) = http_tls {
            // HTTPS with TLS
            let tls =
                match axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                    .await
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Failed to load TLS cert/key: {e}");
                        return;
                    }
                };
            println!("VectraDB HTTPS server running on https://{addr}");
            if let Err(e) = axum_server::bind_rustls(addr.parse().unwrap(), tls)
                .serve(app.into_make_service())
                .await
            {
                eprintln!("HTTPS server error: {e}");
            }
        } else {
            // Plain HTTP
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Failed to bind HTTP port {http_port}: {e}");
                    return;
                }
            };
            println!("VectraDB HTTP server running on http://{addr}");
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("HTTP server error: {e}");
            }
        }
    });

    // Start gRPC server if enabled
    if args.enable_grpc {
        let grpc_addr = format!("0.0.0.0:{}", args.grpc_port).parse()?;
        let grpc_service = VectraDbService::new(db_arc).into_service();

        let grpc_handle = tokio::spawn(async move {
            let mut builder = Server::builder();

            // Enable gRPC-TLS if cert/key provided
            if let Some((cert_path, key_path)) = tls_config {
                let cert_pem = std::fs::read_to_string(&cert_path).unwrap_or_else(|e| {
                    eprintln!("Failed to read TLS cert for gRPC: {e}");
                    std::process::exit(1);
                });
                let key_pem = std::fs::read_to_string(&key_path).unwrap_or_else(|e| {
                    eprintln!("Failed to read TLS key for gRPC: {e}");
                    std::process::exit(1);
                });
                let tls = tonic::transport::ServerTlsConfig::new()
                    .identity(tonic::transport::Identity::from_pem(cert_pem, key_pem));
                builder = builder.tls_config(tls).unwrap_or_else(|e| {
                    eprintln!("Failed to configure gRPC TLS: {e}");
                    std::process::exit(1);
                });
                println!("VectraDB gRPC-TLS server running on {grpc_addr}");
            } else {
                println!("VectraDB gRPC server running on {grpc_addr}");
            }

            if let Err(e) = builder.add_service(grpc_service).serve(grpc_addr).await {
                eprintln!("gRPC server error: {e}");
            }
        });

        // Wait for either server to exit
        tokio::select! {
            _ = http_handle => {
                println!("HTTP server exited");
            }
            _ = grpc_handle => {
                println!("gRPC server exited");
            }
        }
    } else {
        // Only HTTP server
        http_handle.await?;
    }

    Ok(())
}
