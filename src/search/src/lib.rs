use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use vectradb_components::{VectorDocument, VectraDBError};

/// HNSW (Hierarchical Navigable Small World) search implementation
pub mod hnsw;

/// LSH (Locality Sensitive Hashing) search implementation
pub mod lsh;

/// Product Quantization search implementation
pub mod pq;

/// ES4D: Exact Similarity Search via Vector Slicing, adapted for HNSW
pub mod es4d;

/// TensorSearch: Parallel Similarity Search on multi-dimensional tensors
pub mod tensor;

pub use es4d::{ES4DConfig, ES4DIndex};
/// Re-export search algorithms
pub use hnsw::HNSWIndex;
pub use lsh::LSHIndex;
pub use pq::PQIndex;
pub use tensor::TensorSearchEngine;

/// Search configuration for different algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub algorithm: SearchAlgorithm,
    pub max_connections: usize,
    pub search_ef: usize,
    pub construction_ef: usize,
    pub m: usize,                          // For HNSW
    pub ef_construction: usize,            // For HNSW
    pub num_hashes: usize,                 // For LSH
    pub num_buckets: usize,                // For LSH
    pub dimension: Option<usize>,          // Vector dimension
    pub num_subspaces: Option<usize>,      // For PQ
    pub codes_per_subspace: Option<usize>, // For PQ
    pub shard_length: Option<usize>,       // For ES4D (DET shard size, default 64)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SearchAlgorithm {
    HNSW,
    LSH,
    PQ,
    Linear,
    ES4D,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            algorithm: SearchAlgorithm::HNSW,
            max_connections: 16,
            search_ef: 50,
            construction_ef: 200,
            m: 16,
            ef_construction: 200,
            num_hashes: 10,
            num_buckets: 1000,
            dimension: Some(384),
            num_subspaces: Some(8),
            codes_per_subspace: Some(256),
            shard_length: Some(64),
        }
    }
}

/// Search result with distance and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub similarity: f32,
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for max-heap (min distance = max priority)
        other.distance.total_cmp(&self.distance)
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for SearchResult {}

/// Trait for advanced search algorithms
pub trait AdvancedSearch {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError>;
    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError>;
    fn remove(&mut self, id: &str) -> Result<(), VectraDBError>;
    fn update(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError>;
    fn build_index(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError>;
    fn get_stats(&self) -> SearchStats;
}

/// Search algorithm statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub total_vectors: usize,
    pub index_size_bytes: usize,
    pub average_search_time_ms: f64,
    pub construction_time_ms: f64,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            total_vectors: 0,
            index_size_bytes: 0,
            average_search_time_ms: 0.0,
            construction_time_ms: 0.0,
        }
    }
}
