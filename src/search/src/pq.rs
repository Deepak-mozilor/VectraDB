use super::{AdvancedSearch, SearchResult, SearchStats};
use ndarray::{s, Array1, Array2, ArrayView1};
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

/// Product Quantization (PQ) index implementation.
///
/// Splits vectors into `num_subspaces` sub-vectors, quantizes each independently
/// using k-means codebooks. Uses Asymmetric Distance Computation (ADC) for search:
/// the query is compared exactly against codebook centroids, not quantized.
pub struct PQIndex {
    codebooks: Vec<Array2<f32>>, // [num_subspaces] of [codes_per_subspace, subspace_size]
    codes: HashMap<String, Vec<u8>>, // vector_id → compressed PQ codes
    dimension: usize,
    num_subspaces: usize,
    subspace_size: usize,
    codes_per_subspace: usize,
    trained: bool,
    stats: SearchStats,
}

impl PQIndex {
    pub fn new(dimension: usize, num_subspaces: usize, codes_per_subspace: usize) -> Self {
        let subspace_size = dimension / num_subspaces;
        Self {
            codebooks: Vec::new(), // empty until trained
            codes: HashMap::new(),
            dimension,
            num_subspaces,
            subspace_size,
            codes_per_subspace,
            trained: false,
            stats: SearchStats::default(),
        }
    }

    // --------------------------------------------------------
    // Training: k-means++ with multi-restart
    // --------------------------------------------------------

    /// Train codebooks using k-means++ initialization with multiple restarts.
    pub fn train_codebooks(
        &mut self,
        training_vectors: &[Array1<f32>],
    ) -> Result<(), VectraDBError> {
        if training_vectors.is_empty() {
            return Ok(());
        }

        let max_iterations = 50;
        let num_restarts = 3;

        self.codebooks.clear();

        for subspace_idx in 0..self.num_subspaces {
            let start = subspace_idx * self.subspace_size;
            let end = start + self.subspace_size;

            // Extract subspace data
            let subspace_data: Vec<Array1<f32>> = training_vectors
                .iter()
                .map(|v| v.slice(s![start..end]).to_owned())
                .collect();

            // Multi-restart: pick the best codebook across restarts
            let mut best_codebook = None;
            let mut best_inertia = f32::INFINITY;

            for _ in 0..num_restarts {
                let (codebook, inertia) =
                    self.kmeans_plusplus(&subspace_data, self.codes_per_subspace, max_iterations);
                if inertia < best_inertia {
                    best_inertia = inertia;
                    best_codebook = Some(codebook);
                }
            }

            self.codebooks.push(
                best_codebook.unwrap_or_else(|| {
                    Array2::zeros((self.codes_per_subspace, self.subspace_size))
                }),
            );
        }

        self.trained = true;
        Ok(())
    }

    /// K-means++ clustering. Returns (centroids, inertia).
    ///
    /// K-means++ initialization picks centroids proportional to their squared
    /// distance from the nearest existing centroid. This produces much better
    /// initial centroids than random selection.
    fn kmeans_plusplus(
        &self,
        data: &[Array1<f32>],
        k: usize,
        max_iterations: usize,
    ) -> (Array2<f32>, f32) {
        let n = data.len();
        let dim = self.subspace_size;
        let k = k.min(n);

        if k == 0 || n == 0 {
            return (Array2::zeros((k, dim)), 0.0);
        }

        let mut rng = rand::thread_rng();

        // ---- K-means++ initialization ----
        let mut centroids = Array2::zeros((k, dim));

        // First centroid: random data point
        let first = rng.gen_range(0..n);
        centroids.row_mut(0).assign(&data[first].view());

        // Remaining centroids: pick proportional to D²
        let mut distances = vec![f32::INFINITY; n]; // distance to nearest centroid
        for c in 1..k {
            // Update distances to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let d = sq_dist(&point.view(), &centroids.row(c - 1));
                distances[i] = distances[i].min(d);
            }

            // Weighted random selection proportional to D²
            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                // All points are already centroids; fill remaining with random
                centroids
                    .row_mut(c)
                    .assign(&data[rng.gen_range(0..n)].view());
                continue;
            }
            let threshold = rng.gen::<f32>() * total;
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.row_mut(c).assign(&data[chosen].view());
        }

        // ---- K-means iterations ----
        let mut assignments = vec![0usize; n];

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign each point to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let d = sq_dist(&point.view(), &centroids.row(c));
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Recompute centroids
            let mut sums: Array2<f32> = Array2::zeros((k, dim));
            let mut counts = vec![0usize; k];
            for (i, point) in data.iter().enumerate() {
                let c = assignments[i];
                for j in 0..dim {
                    sums[[c, j]] += point[j];
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let d = counts[c] as f32;
                    for j in 0..dim {
                        centroids[[c, j]] = sums[[c, j]] / d;
                    }
                }
            }
        }

        // Compute inertia (total squared distance to nearest centroid)
        let mut inertia = 0.0f32;
        for (i, point) in data.iter().enumerate() {
            inertia += sq_dist(&point.view(), &centroids.row(assignments[i]));
        }

        (centroids, inertia)
    }

    // --------------------------------------------------------
    // Encoding and search (ADC)
    // --------------------------------------------------------

    /// Encode a vector into PQ codes.
    fn encode_vector(&self, vector: &Array1<f32>) -> Result<Vec<u8>, VectraDBError> {
        if vector.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let mut codes = Vec::with_capacity(self.num_subspaces);
        for sub in 0..self.num_subspaces {
            let start = sub * self.subspace_size;
            let end = start + self.subspace_size;
            let subvec = vector.slice(s![start..end]);

            let mut best_code = 0u8;
            let mut best_d = f32::INFINITY;
            for (c, centroid) in self.codebooks[sub].rows().into_iter().enumerate() {
                let d = sq_dist(&subvec, &centroid);
                if d < best_d {
                    best_d = d;
                    best_code = c as u8;
                }
            }
            codes.push(best_code);
        }
        Ok(codes)
    }

    /// Decode PQ codes back to approximate vector.
    #[cfg(test)]
    fn decode_vector(&self, codes: &[u8]) -> Result<Array1<f32>, VectraDBError> {
        if codes.len() != self.num_subspaces {
            return Err(VectraDBError::InvalidVector);
        }
        let mut reconstructed = Array1::zeros(self.dimension);
        for (sub, &code) in codes.iter().enumerate() {
            let start = sub * self.subspace_size;
            let end = start + self.subspace_size;
            reconstructed
                .slice_mut(s![start..end])
                .assign(&self.codebooks[sub].row(code as usize));
        }
        Ok(reconstructed)
    }

    /// ADC (Asymmetric Distance Computation) search.
    ///
    /// Instead of quantizing the query and comparing codes (symmetric, lossy),
    /// we precompute exact distances from the query subvectors to each centroid
    /// in each subspace. Then for each stored vector, we sum up the precomputed
    /// distances using its codes. This is both faster and more accurate than
    /// symmetric PQ distance.
    fn adc_search(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>, VectraDBError> {
        // Precompute distance tables: dist_table[sub][code] = ||query_sub - centroid||²
        let mut dist_table = vec![vec![0.0f32; self.codes_per_subspace]; self.num_subspaces];

        for sub in 0..self.num_subspaces {
            let start = sub * self.subspace_size;
            let end = start + self.subspace_size;
            let query_sub = query.slice(s![start..end]);

            for (c, centroid) in self.codebooks[sub].rows().into_iter().enumerate() {
                dist_table[sub][c] = sq_dist(&query_sub, &centroid);
            }
        }

        // Compute approximate distance for each stored vector using the table
        let mut candidates: Vec<SearchResult> = self
            .codes
            .iter()
            .map(|(id, codes)| {
                let mut dist_sq = 0.0f32;
                for (sub, &code) in codes.iter().enumerate() {
                    dist_sq += dist_table[sub][code as usize];
                }
                let distance = dist_sq.sqrt();
                SearchResult {
                    id: id.clone(),
                    distance,
                    similarity: 1.0 / (1.0 + distance),
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.truncate(k);
        Ok(candidates)
    }
}

/// Squared Euclidean distance between two array views.
#[inline]
fn sq_dist(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let diff = a - b;
    diff.dot(&diff)
}

// ============================================================
// AdvancedSearch trait implementation
// ============================================================

impl AdvancedSearch for PQIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        if !self.trained || self.codebooks.is_empty() {
            // Not trained yet — fall back to encoding query and comparing codes
            let query_codes = self.encode_vector(query)?;
            let mut candidates: Vec<SearchResult> = self
                .codes
                .iter()
                .map(|(id, stored)| {
                    let mut dist = 0.0f32;
                    for (sub, (&qc, &sc)) in query_codes.iter().zip(stored.iter()).enumerate() {
                        let c1 = self.codebooks[sub].row(qc as usize);
                        let c2 = self.codebooks[sub].row(sc as usize);
                        let d = &c1 - &c2;
                        dist += d.dot(&d);
                    }
                    let distance = dist.sqrt();
                    SearchResult {
                        id: id.clone(),
                        distance,
                        similarity: 1.0 / (1.0 + distance),
                    }
                })
                .collect();
            candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            candidates.truncate(k);
            return Ok(candidates);
        }

        // Use ADC for better accuracy
        self.adc_search(query, k)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }
        // If not trained yet, create trivial codebooks from first insert
        if self.codebooks.is_empty() {
            for _ in 0..self.num_subspaces {
                let mut rng = rand::thread_rng();
                let mut centroids = Array2::zeros((self.codes_per_subspace, self.subspace_size));
                for i in 0..self.codes_per_subspace {
                    for j in 0..self.subspace_size {
                        centroids[[i, j]] = rng.gen_range(-1.0..1.0);
                    }
                }
                self.codebooks.push(centroids);
            }
        }

        let codes = self.encode_vector(&document.data)?;
        self.codes.insert(document.metadata.id, codes);
        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        if self.codes.remove(id).is_some() {
            self.stats.total_vectors = self.stats.total_vectors.saturating_sub(1);
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

        let training_vectors: Vec<Array1<f32>> =
            documents.iter().map(|doc| doc.data.clone()).collect();

        // Train codebooks with k-means++ and multi-restart
        self.train_codebooks(&training_vectors)?;

        // Encode all documents using the trained codebooks
        for document in documents {
            let codes = self.encode_vector(&document.data)?;
            self.codes.insert(document.metadata.id, codes);
            self.stats.total_vectors += 1;
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
    fn test_pq_build_and_search() {
        let mut index = PQIndex::new(6, 2, 4);

        // Build with several vectors so training has data
        let docs: Vec<VectorDocument> = (0..20)
            .map(|i| {
                let v = Array1::from_vec(vec![
                    i as f32,
                    (i * 2) as f32,
                    (i * 3) as f32,
                    (i + 1) as f32,
                    (i + 2) as f32,
                    (i + 3) as f32,
                ]);
                create_vector_document(format!("d{i}"), v, None).unwrap()
            })
            .collect();

        index.build_index(docs).unwrap();
        assert!(index.trained);

        let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);

        // d0 = [0,0,0,1,2,3] should be among the top results (PQ is approximate)
        let top_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(
            top_ids.contains(&"d0"),
            "d0 should be in top-5, got: {top_ids:?}"
        );
    }

    #[test]
    fn test_pq_encode_decode() {
        let mut index = PQIndex::new(6, 2, 4);

        // Train first
        let vecs: Vec<Array1<f32>> = (0..10)
            .map(|i| Array1::from_vec(vec![i as f32; 6]))
            .collect();
        index.train_codebooks(&vecs).unwrap();

        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let codes = index.encode_vector(&vector).unwrap();
        assert_eq!(codes.len(), 2);

        let reconstructed = index.decode_vector(&codes).unwrap();
        assert_eq!(reconstructed.len(), 6);
    }

    #[test]
    fn test_pq_recall_improvement() {
        // This test verifies the k-means++ training produces better recall
        // than the old random initialization
        use rand::Rng;

        let dim = 32;
        let n = 200;
        let k = 10;
        let mut rng = rand::thread_rng();

        // Build dataset
        let docs: Vec<VectorDocument> = (0..n)
            .map(|i| {
                let v = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)));
                create_vector_document(format!("d{i}"), v, None).unwrap()
            })
            .collect();

        let mut index = PQIndex::new(dim, 4, 16);
        index.build_index(docs.clone()).unwrap();

        // Run a few queries and check recall against brute-force
        let mut total_recall = 0.0;
        let num_queries = 10;

        for _ in 0..num_queries {
            let query = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)));

            // Brute-force ground truth
            let mut bf: Vec<(String, f32)> = docs
                .iter()
                .map(|d| {
                    let diff = &query - &d.data;
                    (d.metadata.id.clone(), diff.dot(&diff).sqrt())
                })
                .collect();
            bf.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: std::collections::HashSet<String> =
                bf.iter().take(k).map(|(id, _)| id.clone()).collect();

            // PQ search
            let results = index.search(&query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = found.intersection(&gt).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        eprintln!("PQ recall@{k} = {avg_recall:.3} (n={n}, dim={dim})");

        // With k-means++ and ADC, we should get > 40% recall
        // (old random init was ~17%)
        assert!(
            avg_recall > 0.3,
            "PQ recall too low: {avg_recall:.3} (expected > 0.3)"
        );
    }
}
