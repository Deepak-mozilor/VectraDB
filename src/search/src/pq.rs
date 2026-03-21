use super::{AdvancedSearch, SearchResult, SearchStats};
use ndarray::{s, Array1, Array2, ArrayView1};
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

/// Product Quantization (PQ) index implementation
pub struct PQIndex {
    codebooks: Vec<Array2<f32>>,     // Codebooks for each subspace
    codes: HashMap<String, Vec<u8>>, // Compressed codes for each vector
    dimension: usize,
    num_subspaces: usize,
    subspace_size: usize,
    codes_per_subspace: usize,
    stats: SearchStats,
}

impl PQIndex {
    /// Create a new PQ index
    pub fn new(dimension: usize, num_subspaces: usize, codes_per_subspace: usize) -> Self {
        let subspace_size = dimension / num_subspaces;

        // Initialize codebooks
        let mut codebooks = Vec::new();
        for _ in 0..num_subspaces {
            let mut rng = rand::thread_rng();
            let mut centroids = Array2::zeros((codes_per_subspace, subspace_size));

            // Initialize centroids with random values
            for i in 0..codes_per_subspace {
                for j in 0..subspace_size {
                    centroids[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }

            codebooks.push(centroids);
        }

        Self {
            codebooks,
            codes: HashMap::new(),
            dimension,
            num_subspaces,
            subspace_size,
            codes_per_subspace,
            stats: SearchStats::default(),
        }
    }

    /// Train codebooks using k-means clustering
    pub fn train_codebooks(
        &mut self,
        training_vectors: &[Array1<f32>],
    ) -> Result<(), VectraDBError> {
        if training_vectors.is_empty() {
            return Ok(());
        }

        let max_iterations = 100;

        for subspace_idx in 0..self.num_subspaces {
            let start_idx = subspace_idx * self.subspace_size;
            let end_idx = start_idx + self.subspace_size;

            // Extract subspace data
            let mut subspace_data = Vec::new();
            for vector in training_vectors {
                let subspace = vector.slice(s![start_idx..end_idx]).to_owned();
                subspace_data.push(subspace);
            }

            // K-means clustering for this subspace
            self.kmeans_subspace(subspace_idx, &subspace_data, max_iterations)?;
        }

        Ok(())
    }

    /// K-means clustering for a single subspace
    fn kmeans_subspace(
        &mut self,
        subspace_idx: usize,
        data: &[Array1<f32>],
        max_iterations: usize,
    ) -> Result<(), VectraDBError> {
        let mut centroids = self.codebooks[subspace_idx].clone();

        for _ in 0..max_iterations {
            let mut assignments = Vec::new();
            let mut new_centroids = Array2::zeros((self.codes_per_subspace, self.subspace_size));
            let mut counts = vec![0; self.codes_per_subspace];

            // Assign each point to nearest centroid
            for point in data.iter() {
                let mut best_centroid = 0;
                let mut best_distance = f32::INFINITY;

                for (centroid_idx, centroid) in centroids.rows().into_iter().enumerate() {
                    let distance = self.euclidean_distance(&point.view(), &centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_centroid = centroid_idx;
                    }
                }

                assignments.push(best_centroid);
                // Accumulate point data for centroid
                for (i, &val) in point.iter().enumerate() {
                    new_centroids[[best_centroid, i]] += val;
                }
                counts[best_centroid] += 1;
            }

            // Update centroids
            for i in 0..self.codes_per_subspace {
                if counts[i] > 0 {
                    let divisor = counts[i] as f32;
                    for j in 0..self.subspace_size {
                        new_centroids[[i, j]] /= divisor;
                    }
                }
            }

            // Check for convergence
            if self.centroids_converged(&centroids, &new_centroids, 1e-6) {
                break;
            }

            centroids = new_centroids;
        }

        self.codebooks[subspace_idx] = centroids;
        Ok(())
    }

    /// Check if centroids have converged
    fn centroids_converged(&self, old: &Array2<f32>, new: &Array2<f32>, threshold: f32) -> bool {
        let diff = old - new;
        diff.mapv(|x| x * x).sum().sqrt() < threshold
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// Encode a vector using PQ
    fn encode_vector(&self, vector: &Array1<f32>) -> Result<Vec<u8>, VectraDBError> {
        if vector.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let mut codes = Vec::new();

        for subspace_idx in 0..self.num_subspaces {
            let start_idx = subspace_idx * self.subspace_size;
            let end_idx = start_idx + self.subspace_size;
            let subspace = vector.slice(s![start_idx..end_idx]);

            // Find closest centroid in this subspace
            let mut best_code = 0;
            let mut best_distance = f32::INFINITY;

            for (code_idx, centroid) in self.codebooks[subspace_idx].rows().into_iter().enumerate()
            {
                let distance = self.euclidean_distance(&subspace, &centroid);
                if distance < best_distance {
                    best_distance = distance;
                    best_code = code_idx;
                }
            }

            codes.push(best_code as u8);
        }

        Ok(codes)
    }

    /// Decode PQ codes back to approximate vector
    #[cfg(test)]
    fn decode_vector(&self, codes: &[u8]) -> Result<Array1<f32>, VectraDBError> {
        if codes.len() != self.num_subspaces {
            return Err(VectraDBError::InvalidVector);
        }

        let mut reconstructed = Array1::zeros(self.dimension);

        for (subspace_idx, &code) in codes.iter().enumerate() {
            if code as usize >= self.codes_per_subspace {
                return Err(VectraDBError::InvalidVector);
            }

            let start_idx = subspace_idx * self.subspace_size;
            let end_idx = start_idx + self.subspace_size;

            let centroid = self.codebooks[subspace_idx].row(code as usize);
            reconstructed
                .slice_mut(s![start_idx..end_idx])
                .assign(&centroid);
        }

        Ok(reconstructed)
    }

    /// Calculate approximate distance using PQ codes
    fn pq_distance(&self, codes1: &[u8], codes2: &[u8]) -> f32 {
        let mut distance = 0.0;

        for (subspace_idx, (&code1, &code2)) in codes1.iter().zip(codes2.iter()).enumerate() {
            let centroid1 = self.codebooks[subspace_idx].row(code1 as usize);
            let centroid2 = self.codebooks[subspace_idx].row(code2 as usize);

            let diff = &centroid1 - &centroid2;
            distance += diff.dot(&diff);
        }

        distance.sqrt()
    }
}

impl AdvancedSearch for PQIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let start_time = Instant::now();
        let query_codes = self.encode_vector(query)?;

        let mut candidates = Vec::new();

        // Calculate approximate distances using PQ codes
        for (id, stored_codes) in &self.codes {
            let distance = self.pq_distance(&query_codes, stored_codes);
            let similarity = 1.0 / (1.0 + distance);

            candidates.push(SearchResult {
                id: id.clone(),
                distance,
                similarity,
            });
        }

        // Sort by similarity and take top-k
        candidates.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        candidates.truncate(k);

        let _search_time = start_time.elapsed().as_millis() as f64;
        // Note: stats update requires &mut self but search takes &self.

        Ok(candidates)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }

        let codes = self.encode_vector(&document.data)?;
        self.codes.insert(document.metadata.id, codes);

        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        if self.codes.remove(id).is_some() {
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

        // Extract training vectors
        let training_vectors: Vec<Array1<f32>> =
            documents.iter().map(|doc| doc.data.clone()).collect();

        // Train codebooks
        self.train_codebooks(&training_vectors)?;

        // Encode all documents
        for document in documents {
            self.insert(document)?;
        }

        self.stats.construction_time_ms = start_time.elapsed().as_millis() as f64;
        Ok(())
    }

    fn get_stats(&self) -> SearchStats {
        SearchStats {
            total_vectors: self.stats.total_vectors,
            index_size_bytes: self.codes.len() * self.num_subspaces
                + self.codebooks.len() * self.codes_per_subspace * self.subspace_size * 4,
            average_search_time_ms: self.stats.average_search_time_ms,
            construction_time_ms: self.stats.construction_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectradb_components::vector_operations::create_vector_document;

    #[test]
    fn test_pq_creation() {
        let index = PQIndex::new(6, 2, 4);
        assert_eq!(index.dimension, 6);
        assert_eq!(index.num_subspaces, 2);
        assert_eq!(index.subspace_size, 3);
    }

    #[test]
    fn test_pq_encode_decode() {
        let index = PQIndex::new(6, 2, 4);
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let codes = index.encode_vector(&vector).unwrap();
        assert_eq!(codes.len(), 2);

        let reconstructed = index.decode_vector(&codes).unwrap();
        assert_eq!(reconstructed.len(), 6);
    }

    #[test]
    fn test_pq_insert_and_search() {
        let mut index = PQIndex::new(6, 2, 4);

        let doc1 = create_vector_document(
            "1".to_string(),
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            None,
        )
        .unwrap();

        index.insert(doc1).unwrap();

        let query = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }
}
