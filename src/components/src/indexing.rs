use crate::{ArrayView1, SimilarityResult, VectorDocument, VectraDBError};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Indexing strategies for efficient vector search.
///
/// Basic linear index - scans all vectors (O(n) search time)
pub struct LinearIndex {
    vectors: Vec<VectorDocument>,
}

impl LinearIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, document: VectorDocument) {
        self.vectors.push(document);
    }

    pub fn remove_vector(&mut self, id: &str) -> Result<(), VectraDBError> {
        let pos = self
            .vectors
            .iter()
            .position(|v| v.metadata.id == id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;
        self.vectors.remove(pos);
        Ok(())
    }

    pub fn search(
        &self,
        query_vector: &Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        crate::similarity::find_similar_vectors_cosine(&query_vector.view(), &self.vectors, top_k)
    }

    pub fn update_vector(
        &mut self,
        id: &str,
        document: VectorDocument,
    ) -> Result<(), VectraDBError> {
        let pos = self
            .vectors
            .iter()
            .position(|v| v.metadata.id == id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;
        self.vectors[pos] = document;
        Ok(())
    }

    pub fn get_vector(&self, id: &str) -> Result<&VectorDocument, VectraDBError> {
        self.vectors
            .iter()
            .find(|v| v.metadata.id == id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

impl Default for LinearIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash-based index for approximate nearest neighbor search
pub struct HashIndex {
    buckets: HashMap<u64, Vec<VectorDocument>>,
    hash_functions: Vec<HashFunction>,
}

/// Simple hash function for vector quantization
#[derive(Debug, Clone)]
pub struct HashFunction {
    pub weights: Vec<f32>,
    pub threshold: f32,
}

impl HashFunction {
    pub fn new(dimension: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let weights: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            weights,
            threshold: 0.0,
        }
    }

    pub fn hash(&self, vector: &ArrayView1<f32>) -> u64 {
        let mut hash_value = 0u64;

        for (i, &weight) in self.weights.iter().enumerate() {
            if vector[i] * weight > self.threshold {
                hash_value |= 1 << (i % 64);
            }
        }

        hash_value
    }
}

impl HashIndex {
    pub fn new(dimension: usize, num_hash_functions: usize) -> Self {
        let hash_functions: Vec<HashFunction> = (0..num_hash_functions)
            .map(|_| HashFunction::new(dimension))
            .collect();

        Self {
            buckets: HashMap::new(),
            hash_functions,
        }
    }

    pub fn add_vector(&mut self, document: VectorDocument) {
        let bucket_keys = self.get_bucket_keys(&document.data);

        for key in bucket_keys {
            self.buckets.entry(key).or_default().push(document.clone());
        }
    }

    fn get_bucket_keys(&self, vector: &Array1<f32>) -> Vec<u64> {
        self.hash_functions
            .iter()
            .map(|hf| hf.hash(&vector.view()))
            .collect()
    }

    pub fn search(
        &self,
        query_vector: &Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        let bucket_keys = self.get_bucket_keys(query_vector);
        let mut candidates = Vec::new();

        // Collect candidates from matching buckets
        for key in bucket_keys {
            if let Some(bucket) = self.buckets.get(&key) {
                candidates.extend(bucket.iter().cloned());
            }
        }

        // Remove duplicates
        candidates.sort_by(|a, b| a.metadata.id.cmp(&b.metadata.id));
        candidates.dedup_by(|a, b| a.metadata.id == b.metadata.id);

        // Calculate similarities and return top-k
        crate::similarity::find_similar_vectors_cosine(&query_vector.view(), &candidates, top_k)
    }

    pub fn remove_vector(&mut self, id: &str) -> Result<(), VectraDBError> {
        let mut found = false;

        for bucket in self.buckets.values_mut() {
            if let Some(pos) = bucket.iter().position(|v| v.metadata.id == id) {
                bucket.remove(pos);
                found = true;
                break;
            }
        }

        if found {
            Ok(())
        } else {
            Err(VectraDBError::VectorNotFound { id: id.to_string() })
        }
    }

    pub fn update_vector(
        &mut self,
        id: &str,
        document: VectorDocument,
    ) -> Result<(), VectraDBError> {
        // Remove old vector
        self.remove_vector(id)?;

        // Add updated vector
        self.add_vector(document);
        Ok(())
    }
}

/// Index configuration for different indexing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: IndexType,
    pub dimension: usize,
    pub hash_functions: Option<usize>,    // For hash index
    pub rebuild_threshold: Option<usize>, // When to rebuild index
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Linear,
    Hash,
    HNSW, // Will be implemented in search module
    LSH,  // Will be implemented in search module
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_type: IndexType::Linear,
            dimension: 384,
            hash_functions: Some(10),
            rebuild_threshold: Some(10000),
        }
    }
}

/// Index trait for different indexing implementations
pub trait VectorIndex {
    fn add_vector(&mut self, document: VectorDocument) -> Result<(), VectraDBError>;
    fn remove_vector(&mut self, id: &str) -> Result<(), VectraDBError>;
    fn update_vector(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError>;
    fn search(
        &self,
        query_vector: &Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError>;
    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn rebuild(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError>;
}

impl VectorIndex for LinearIndex {
    fn add_vector(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        self.add_vector(document);
        Ok(())
    }

    fn remove_vector(&mut self, id: &str) -> Result<(), VectraDBError> {
        self.remove_vector(id)
    }

    fn update_vector(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError> {
        self.update_vector(id, document)
    }

    fn search(
        &self,
        query_vector: &Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        self.search(query_vector, top_k)
    }

    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        Ok(self.get_vector(id)?.clone())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn rebuild(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError> {
        self.vectors = documents;
        Ok(())
    }
}

impl VectorIndex for HashIndex {
    fn add_vector(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        self.add_vector(document);
        Ok(())
    }

    fn remove_vector(&mut self, id: &str) -> Result<(), VectraDBError> {
        self.remove_vector(id)
    }

    fn update_vector(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError> {
        self.update_vector(id, document)
    }

    fn search(
        &self,
        query_vector: &Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<SimilarityResult>, VectraDBError> {
        self.search(query_vector, top_k)
    }

    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        for bucket in self.buckets.values() {
            if let Some(doc) = bucket.iter().find(|v| v.metadata.id == id) {
                return Ok(doc.clone());
            }
        }
        Err(VectraDBError::VectorNotFound { id: id.to_string() })
    }

    fn len(&self) -> usize {
        self.buckets.values().map(|bucket| bucket.len()).sum()
    }

    fn rebuild(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError> {
        self.buckets.clear();
        for doc in documents {
            self.add_vector(doc);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_operations::create_vector_document;

    #[test]
    fn test_linear_index() {
        let mut index = LinearIndex::new();
        let doc =
            create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]), None)
                .unwrap();

        index.add_vector(doc);
        assert_eq!(index.len(), 1);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_hash_index() {
        let mut index = HashIndex::new(3, 5);
        let doc =
            create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]), None)
                .unwrap();

        index.add_vector(doc);
        assert!(index.len() > 0);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_hash_function() {
        let hf = HashFunction::new(3);
        let vector = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let hash = hf.hash(&vector.view());

        // Hash should be consistent for same input
        let hash2 = hf.hash(&vector.view());
        assert_eq!(hash, hash2);
    }
}
