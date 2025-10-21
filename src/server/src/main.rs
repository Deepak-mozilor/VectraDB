use vectradb_api::start_server;
use vectradb_storage::DatabaseConfig;
use vectradb_search::SearchAlgorithm;
use clap::Parser;
use std::path::PathBuf;

/// VectraDB Server - High-performance vector database
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Data directory for persistent storage
    #[arg(short, long, default_value = "./vectradb_data")]
    data_dir: PathBuf,

    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Search algorithm (hnsw, lsh, pq)
    #[arg(short, long, default_value = "hnsw")]
    algorithm: String,

    /// Vector dimension
    #[arg(short, long, default_value = "384")]
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
        _ => {
            eprintln!("Invalid algorithm: {}. Supported algorithms: hnsw, lsh, pq", args.algorithm);
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
        },
        auto_flush: args.auto_flush,
        cache_size: args.cache_size,
    };

    println!("Starting VectraDB server...");
    println!("Data directory: {}", config.data_dir);
    println!("Search algorithm: {:?}", config.search_algorithm);
    println!("Vector dimension: {}", args.dimension);
    println!("Server port: {}", args.port);

    // Start the server
    start_server(config, args.port).await?;

    Ok(())
}





