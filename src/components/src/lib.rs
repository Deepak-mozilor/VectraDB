use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// Re-export commonly used types
pub use ndarray::{Array1, ArrayView1};

/// Vector database error types
#[derive(Error, Debug)]
pub enum VectraDBError {
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },
    #[error("Vector already exists: {id}")]
    DuplicateVector { id: String },
    #[error("Invalid vector data")]
    InvalidVector,
    #[error("Database error: {0}")]
    DatabaseError(#[from] anyhow::Error),
}

/// Vector metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub id: String,
    pub dimension: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub tags: HashMap<String, String>,
}

/// Vector document structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub metadata: VectorMetadata,
    pub data: Array1<f32>,
}

/// Vector similarity result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub id: String,
    pub score: f32,
    pub metadata: VectorMetadata,
}

/// Vector database trait for different implementations
pub trait VectorDatabase {
    /// Create a new vector in the database
    fn create_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError>;

    /// Fetch a vector by ID
    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError>;

    /// Update an existing vector
    fn update_vector(
        &mut self,
        id: &str,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError>;

    /// Delete a vector by ID
    fn delete_vector(&mut self, id: &str) -> Result<(), VectraDBError>;

    /// Upsert (insert or update) a vector
    fn upsert_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError>;

    /// Search for similar vectors
    fn search_similar(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError>;

    /// Get all vector IDs
    fn list_vectors(&self) -> Result<Vec<String>, VectraDBError>;

    /// Get database statistics
    fn get_stats(&self) -> Result<DatabaseStats, VectraDBError>;
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatabaseStats {
    pub total_vectors: usize,
    pub dimension: usize,
    pub memory_usage: u64,
}

// Module declarations
pub mod indexing;
pub mod similarity;
pub mod storage;
pub mod tensor;
pub mod vector_operations;

// Re-export main functionality
pub use similarity::*;
pub use vector_operations::*;
