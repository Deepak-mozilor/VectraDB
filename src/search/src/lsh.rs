use super::{AdvancedSearch, SearchResult, SearchStats};
use ndarray::Array1;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

/// LSH (Locality Sensitive Hashing) index implementation
pub struct LSHIndex {
    hash_functions: Vec<LSHFunction>,
    buckets: HashMap<String, Vec<VectorDocument>>,
    dimension: usize,
    num_hashes: usize,
    stats: SearchStats,
}

/// LSH hash function using random hyperplanes
#[derive(Debug, Clone)]
struct LSHFunction {
    hyperplane: Array1<f32>,
    threshold: f32,
}

impl LSHFunction {
    fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let hyperplane: Array1<f32> =
            Array1::from_iter((0..dimension).map(|_| rng.gen_range(-1.0..1.0)));
        let threshold = rng.gen_range(-1.0..1.0);

        Self {
            hyperplane,
            threshold,
        }
    }

    fn hash(&self, vector: &Array1<f32>) -> bool {
        let dot_product = vector.dot(&self.hyperplane);
        dot_product > self.threshold
    }
}

impl LSHIndex {
    /// Create a new LSH index
    pub fn new(dimension: usize, num_hashes: usize) -> Self {
        let hash_functions: Vec<LSHFunction> = (0..num_hashes)
            .map(|_| LSHFunction::new(dimension))
            .collect();

        Self {
            hash_functions,
            buckets: HashMap::new(),
            dimension,
            num_hashes,
            stats: SearchStats::default(),
        }
    }

    /// Generate hash signature for a vector
    fn hash_signature(&self, vector: &Array1<f32>) -> String {
        let mut signature = String::new();

        for hash_fn in &self.hash_functions {
            signature.push(if hash_fn.hash(vector) { '1' } else { '0' });
        }

        signature
    }

    /// Calculate Jaccard similarity between two hash signatures
    fn jaccard_similarity(&self, sig1: &str, sig2: &str) -> f32 {
        if sig1.len() != sig2.len() {
            return 0.0;
        }

        let mut intersection = 0;
        let mut union = 0;

        for (c1, c2) in sig1.chars().zip(sig2.chars()) {
            if c1 == '1' || c2 == '1' {
                union += 1;
                if c1 == '1' && c2 == '1' {
                    intersection += 1;
                }
            }
        }

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

impl AdvancedSearch for LSHIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let start_time = Instant::now();
        let query_signature = self.hash_signature(query);

        // Find candidates using hash signatures
        let mut candidates = Vec::new();
        let mut visited = HashSet::new();

        // Look in the same bucket as query
        if let Some(bucket) = self.buckets.get(&query_signature) {
            for doc in bucket {
                if !visited.contains(&doc.metadata.id) {
                    let similarity = self.cosine_similarity(query, &doc.data);
                    candidates.push(SearchResult {
                        id: doc.metadata.id.clone(),
                        distance: 1.0 - similarity,
                        similarity,
                    });
                    visited.insert(doc.metadata.id.clone());
                }
            }
        }

        // Also look in similar buckets (Hamming distance = 1)
        for (bucket_signature, bucket) in &self.buckets {
            if self.hamming_distance(&query_signature, bucket_signature) <= 1 {
                for doc in bucket {
                    if !visited.contains(&doc.metadata.id) {
                        let similarity = self.cosine_similarity(query, &doc.data);
                        candidates.push(SearchResult {
                            id: doc.metadata.id.clone(),
                            distance: 1.0 - similarity,
                            similarity,
                        });
                        visited.insert(doc.metadata.id.clone());
                    }
                }
            }
        }

        // Sort by similarity and take top-k
        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        candidates.truncate(k);

        // Update stats
        let search_time = start_time.elapsed().as_millis() as f64;
        let mut stats = self.stats.clone();
        stats.average_search_time_ms = (stats.average_search_time_ms + search_time) / 2.0;

        Ok(candidates)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }

        let signature = self.hash_signature(&document.data);
        self.buckets
            .entry(signature)
            .or_insert_with(Vec::new)
            .push(document);

        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        let mut found = false;

        for bucket in self.buckets.values_mut() {
            if let Some(pos) = bucket.iter().position(|doc| doc.metadata.id == id) {
                bucket.remove(pos);
                found = true;
                break;
            }
        }

        if found {
            self.stats.total_vectors -= 1;
            Ok(())
        } else {
            Err(VectraDBError::VectorNotFound { id: id.to_string() })
        }
    }

    fn update(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError> {
        self.remove(id)?;
        self.insert(document)
    }

    fn build_index(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError> {
        let start_time = Instant::now();

        for document in documents {
            self.insert(document)?;
        }

        self.stats.construction_time_ms = start_time.elapsed().as_millis() as f64;
        Ok(())
    }

    fn get_stats(&self) -> SearchStats {
        SearchStats {
            total_vectors: self.stats.total_vectors,
            index_size_bytes: self.buckets.len() * 64
                + self.stats.total_vectors * self.dimension * 4,
            average_search_time_ms: self.stats.average_search_time_ms,
            construction_time_ms: self.stats.construction_time_ms,
        }
    }
}

impl LSHIndex {
    /// Calculate Hamming distance between two binary strings
    fn hamming_distance(&self, s1: &str, s2: &str) -> usize {
        if s1.len() != s2.len() {
            return usize::MAX;
        }

        s1.chars()
            .zip(s2.chars())
            .filter(|(c1, c2)| c1 != c2)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectradb_components::vector_operations::create_vector_document;

    #[test]
    fn test_lsh_creation() {
        let index = LSHIndex::new(3, 10);
        assert_eq!(index.dimension, 3);
        assert_eq!(index.num_hashes, 10);
    }

    #[test]
    fn test_lsh_hash_signature() {
        let index = LSHIndex::new(3, 5);
        let vector = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let signature = index.hash_signature(&vector);
        assert_eq!(signature.len(), 5);
    }

    #[test]
    fn test_lsh_insert_and_search() {
        let mut index = LSHIndex::new(3, 10);

        let doc1 =
            create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]), None)
                .unwrap();
        let doc2 =
            create_vector_document("2".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0]), None)
                .unwrap();

        index.insert(doc1).unwrap();
        index.insert(doc2).unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1"); // Should be most similar
    }

    #[test]
    fn test_hamming_distance() {
        let index = LSHIndex::new(3, 5);
        assert_eq!(index.hamming_distance("10101", "10101"), 0);
        assert_eq!(index.hamming_distance("10101", "10100"), 1);
        assert_eq!(index.hamming_distance("10101", "01010"), 5);
    }
}
