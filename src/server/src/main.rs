use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;
use vectradb_search::SearchAlgorithm;
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

    /// Enable auto-flush
    #[arg(long, default_value = "true")]
    auto_flush: bool,

    /// Cache size
    #[arg(long, default_value = "1000")]
    cache_size: usize,
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

    // Clone shared database for HTTP server
    let http_db = db_arc.clone();
    let _http_config = config.clone();
    let http_port = args.port;

    // Start HTTP server task
    let http_handle = tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{}", http_port)).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Failed to bind HTTP port {}: {}", http_port, e);
                return;
            }
        };
        println!(
            "VectraDB HTTP API server running on http://0.0.0.0:{}",
            http_port
        );

        let state = vectradb_api::AppState { db: http_db };
        let app = vectradb_api::create_router(state);

        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("HTTP server error: {}", e);
        }
    });

    // Start gRPC server if enabled
    if args.enable_grpc {
        let grpc_addr = format!("0.0.0.0:{}", args.grpc_port).parse()?;
        let grpc_service = VectraDbService::new(db_arc).into_service();

        println!("VectraDB gRPC server running on {}", grpc_addr);

        let grpc_handle = tokio::spawn(async move {
            if let Err(e) = Server::builder()
                .add_service(grpc_service)
                .serve(grpc_addr)
                .await
            {
                eprintln!("gRPC server error: {}", e);
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
