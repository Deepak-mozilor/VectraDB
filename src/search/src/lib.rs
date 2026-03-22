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

/// Scalar Quantization index (4x memory reduction)
pub mod sq;

/// SIMD-accelerated distance functions (AVX2 / SSE / NEON with scalar fallback)
pub mod simd;

/// GPU-accelerated batch distance computation (requires `gpu` feature)
#[cfg(feature = "gpu")]
pub mod gpu;

pub use es4d::{ES4DConfig, ES4DIndex};
/// Re-export search algorithms
pub use hnsw::HNSWIndex;
pub use lsh::LSHIndex;
pub use pq::PQIndex;
pub use sq::SQIndex;
pub use tensor::TensorSearchEngine;

/// Distance metric used by search algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
}

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
    pub metric: DistanceMetric,            // Distance metric
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SearchAlgorithm {
    HNSW,
    LSH,
    PQ,
    Linear,
    ES4D,
    /// Scalar Quantization — 4x memory reduction, brute-force search.
    SQ,
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
            metric: DistanceMetric::Euclidean,
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
    /// Search with a per-query ef override. Implementations that support it
    /// should use `ef` instead of their configured default. Falls back to
    /// `search()` by default.
    fn search_with_ef(
        &self,
        query: &Array1<f32>,
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>, VectraDBError> {
        let _ = ef; // default ignores ef
        self.search(query, k)
    }
    /// Hybrid GPU search: use index to get candidates, then GPU re-ranks exactly.
    ///
    /// `rerank_ef` controls how many candidates the index fetches (higher = better
    /// recall but more GPU work). The GPU then computes exact distances for all
    /// candidates and returns the true top-k.
    #[cfg(feature = "gpu")]
    fn search_gpu_rerank(
        &self,
        query: &Array1<f32>,
        k: usize,
        rerank_ef: usize,
        gpu: &gpu::GpuDistanceEngine,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>, VectraDBError> {
        // Default: fall back to CPU search (overridden for HNSW/ES4D)
        let _ = (gpu, metric, rerank_ef);
        self.search(query, k)
    }

    /// Search by similarity threshold instead of top-k.
    /// Returns all results with similarity >= `min_similarity`, up to `max_results`.
    fn search_by_threshold(
        &self,
        query: &Array1<f32>,
        min_similarity: f32,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, VectraDBError> {
        let results = self.search(query, max_results)?;
        Ok(results
            .into_iter()
            .filter(|r| r.similarity >= min_similarity)
            .collect())
    }

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
