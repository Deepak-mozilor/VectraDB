//! Inverted File Index (IVF) — the workhorse of large-scale vector search.
//!
//! Partitions vectors into `nlist` clusters via k-means, then only searches
//! the `nprobe` closest clusters per query.
//!
//! At 1M vectors with nlist=1000 and nprobe=10, only ~1% of vectors are
//! compared per query — giving 100x speedup over brute force.
//!
//! # Usage
//! ```bash
//! ./vectradb-server -a ivf --ivf-nlist 256 --ivf-nprobe 16 -d 384
//! ```

use super::simd;
use super::{AdvancedSearch, DistanceMetric, SearchResult, SearchStats};
use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

/// IVF index configuration.
#[derive(Debug, Clone)]
pub struct IVFConfig {
    pub dimension: usize,
    /// Number of clusters (partitions). Rule of thumb: sqrt(n) to 4*sqrt(n).
    pub nlist: usize,
    /// Number of clusters to search per query. Higher = better recall, slower.
    pub nprobe: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl Default for IVFConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            nlist: 256,
            nprobe: 16,
            metric: DistanceMetric::Euclidean,
        }
    }
}

/// A single cluster (inverted list) in the IVF index.
struct IVFList {
    /// IDs of vectors in this cluster.
    ids: Vec<String>,
    /// Flat storage of vector data: ids.len() * dimension floats.
    data: Vec<f32>,
}

/// Inverted File Index.
pub struct IVFIndex {
    /// Cluster centroids: [nlist, dimension].
    centroids: Vec<Vec<f32>>,
    /// Inverted lists: one per cluster.
    lists: Vec<IVFList>,
    /// Full documents for metadata retrieval.
    documents: HashMap<String, VectorDocument>,
    config: IVFConfig,
    trained: bool,
    stats: SearchStats,
}

impl IVFIndex {
    pub fn new(config: IVFConfig) -> Self {
        Self {
            centroids: Vec::new(),
            lists: Vec::new(),
            documents: HashMap::new(),
            config,
            trained: false,
            stats: SearchStats::default(),
        }
    }

    // --------------------------------------------------------
    // Training: k-means++ on full vectors
    // --------------------------------------------------------

    fn train(&mut self, vectors: &[&[f32]]) {
        if vectors.is_empty() {
            return;
        }

        let k = self.config.nlist.min(vectors.len());
        let dim = self.config.dimension;
        let max_iters = 25;
        let mut rng = rand::thread_rng();

        // K-means++ initialization
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

        // First centroid: random point
        centroids.push(vectors[rng.gen_range(0..vectors.len())].to_vec());

        // Remaining: D² weighted
        let mut distances = vec![f32::INFINITY; vectors.len()];
        for c in 1..k {
            for (i, vec) in vectors.iter().enumerate() {
                let d = sq_dist_slice(vec, &centroids[c - 1]);
                distances[i] = distances[i].min(d);
            }
            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                centroids.push(vectors[rng.gen_range(0..vectors.len())].to_vec());
                continue;
            }
            let thresh = rng.gen::<f32>() * total;
            let mut cum = 0.0;
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                cum += d;
                if cum >= thresh {
                    chosen = i;
                    break;
                }
            }
            centroids.push(vectors[chosen].to_vec());
        }

        // K-means iterations
        let mut assignments = vec![0usize; vectors.len()];
        for _ in 0..max_iters {
            let mut changed = false;
            for (i, vec) in vectors.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f32::INFINITY;
                for (c, cent) in centroids.iter().enumerate() {
                    let d = sq_dist_slice(vec, cent);
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
            let mut sums = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, vec) in vectors.iter().enumerate() {
                let c = assignments[i];
                for j in 0..dim {
                    sums[c][j] += vec[j];
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let d = counts[c] as f32;
                    for j in 0..dim {
                        centroids[c][j] = sums[c][j] / d;
                    }
                }
            }
        }

        self.centroids = centroids;
        self.lists = (0..k)
            .map(|_| IVFList {
                ids: Vec::new(),
                data: Vec::new(),
            })
            .collect();
        self.trained = true;
    }

    /// Find the cluster index for a vector.
    fn assign(&self, vector: &[f32]) -> usize {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for (c, cent) in self.centroids.iter().enumerate() {
            let d = self.distance_slice(vector, cent);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }

    /// Find the `nprobe` closest clusters to a query.
    fn find_nearest_clusters(&self, query: &[f32]) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.distance_slice(query, c)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored
            .iter()
            .take(self.config.nprobe)
            .map(|&(i, _)| i)
            .collect()
    }

    /// Distance between two slices using the configured metric.
    #[inline]
    fn distance_slice(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Euclidean => simd::simd_l2_distance(a, b),
            DistanceMetric::Cosine => simd::simd_cosine_distance(a, b),
            DistanceMetric::DotProduct => -simd::simd_dot(a, b),
        }
    }

    /// Add a vector to the appropriate inverted list.
    fn add_to_list(&mut self, id: &str, vector: &[f32]) {
        let cluster = self.assign(vector);
        self.lists[cluster].ids.push(id.to_string());
        self.lists[cluster].data.extend_from_slice(vector);
    }
}

/// Squared distance between two slices (for training, metric-agnostic).
#[inline]
fn sq_dist_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

impl AdvancedSearch for IVFIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.config.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let query_slice = query.as_slice().unwrap();
        let dim = self.config.dimension;

        // Find nprobe closest clusters
        let probe_clusters = self.find_nearest_clusters(query_slice);

        // Search only within those clusters
        let mut candidates: Vec<SearchResult> = Vec::new();

        for &cluster_id in &probe_clusters {
            let list = &self.lists[cluster_id];
            let num_vectors = list.ids.len();

            for i in 0..num_vectors {
                let vec_start = i * dim;
                let vec_data = &list.data[vec_start..vec_start + dim];
                let distance = self.distance_slice(query_slice, vec_data);
                let similarity = match self.config.metric {
                    DistanceMetric::DotProduct => -distance,
                    _ => 1.0 / (1.0 + distance),
                };

                candidates.push(SearchResult {
                    id: list.ids[i].clone(),
                    distance,
                    similarity,
                });
            }
        }

        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.truncate(k);
        Ok(candidates)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.config.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.config.dimension,
                actual: document.data.len(),
            });
        }

        let id = document.metadata.id.clone();
        let vec_slice = document.data.as_slice().unwrap().to_vec();

        if !self.trained {
            // Bootstrap: train on first insert with a single cluster
            self.centroids = vec![vec_slice.clone()];
            self.lists = vec![IVFList {
                ids: Vec::new(),
                data: Vec::new(),
            }];
            self.trained = true;
        }

        self.add_to_list(&id, &vec_slice);
        self.documents.insert(id, document);
        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        if self.documents.remove(id).is_none() {
            return Err(VectraDBError::VectorNotFound { id: id.to_string() });
        }

        let dim = self.config.dimension;
        for list in &mut self.lists {
            if let Some(pos) = list.ids.iter().position(|i| i == id) {
                list.ids.remove(pos);
                list.data.drain(pos * dim..(pos + 1) * dim);
                break;
            }
        }

        self.stats.total_vectors = self.stats.total_vectors.saturating_sub(1);
        Ok(())
    }

    fn update(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError> {
        self.remove(id)?;
        self.insert(document)
    }

    fn build_index(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError> {
        let start = Instant::now();
        let dim = self.config.dimension;

        // Collect flat vector data for training
        let flat_vecs: Vec<Vec<f32>> = documents
            .iter()
            .map(|d| d.data.as_slice().unwrap().to_vec())
            .collect();
        let vec_refs: Vec<&[f32]> = flat_vecs.iter().map(|v| v.as_slice()).collect();

        // Train k-means clusters
        self.train(&vec_refs);

        // Assign all vectors to clusters
        for (doc, vec_data) in documents.iter().zip(flat_vecs.iter()) {
            self.add_to_list(&doc.metadata.id, vec_data);
            self.documents.insert(doc.metadata.id.clone(), doc.clone());
        }

        self.stats.total_vectors = documents.len();
        self.stats.construction_time_ms = start.elapsed().as_millis() as f64;
        self.stats.index_size_bytes = documents.len() * dim * 4;
        Ok(())
    }

    fn get_stats(&self) -> SearchStats {
        SearchStats {
            total_vectors: self.stats.total_vectors,
            index_size_bytes: self.stats.total_vectors * self.config.dimension * 4
                + self.centroids.len() * self.config.dimension * 4,
            average_search_time_ms: self.stats.average_search_time_ms,
            construction_time_ms: self.stats.construction_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectradb_components::vector_operations::create_vector_document;

    fn random_docs(n: usize, dim: usize) -> Vec<VectorDocument> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|i| {
                let v = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)));
                create_vector_document(format!("d{i}"), v, None).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_ivf_build_and_search() {
        let config = IVFConfig {
            dimension: 16,
            nlist: 8,
            nprobe: 4,
            metric: DistanceMetric::Euclidean,
        };
        let mut index = IVFIndex::new(config);
        let docs = random_docs(200, 16);
        index.build_index(docs).unwrap();

        assert_eq!(index.stats.total_vectors, 200);

        let query = Array1::from_iter((0..16).map(|_| rand::thread_rng().gen_range(-1.0..1.0)));
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }

    #[test]
    fn test_ivf_recall() {
        let dim = 32;
        let n = 500;
        let k = 10;
        let mut rng = rand::thread_rng();

        let docs = random_docs(n, dim);

        let config = IVFConfig {
            dimension: dim,
            nlist: 32,
            nprobe: 8, // searching 25% of clusters
            metric: DistanceMetric::Euclidean,
        };
        let mut index = IVFIndex::new(config);
        index.build_index(docs.clone()).unwrap();

        let mut total_recall = 0.0;
        let queries = 20;

        for _ in 0..queries {
            let query = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)));

            // Brute-force ground truth
            let mut bf: Vec<(String, f32)> = docs
                .iter()
                .map(|d| {
                    let a = query.as_slice().unwrap();
                    let b = d.data.as_slice().unwrap();
                    (d.metadata.id.clone(), simd::simd_l2_distance(a, b))
                })
                .collect();
            bf.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: std::collections::HashSet<String> =
                bf.iter().take(k).map(|(id, _)| id.clone()).collect();

            let results = index.search(&query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = found.intersection(&gt).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg = total_recall / queries as f64;
        eprintln!("IVF recall@{k} = {avg:.3} (n={n}, nlist=32, nprobe=8)");

        // With nprobe=8 out of 32 clusters (25%), recall should be decent
        assert!(avg > 0.5, "IVF recall too low: {avg:.3}");
    }

    #[test]
    fn test_ivf_insert_remove() {
        let config = IVFConfig {
            dimension: 4,
            nlist: 2,
            nprobe: 2,
            metric: DistanceMetric::Euclidean,
        };
        let mut index = IVFIndex::new(config);

        let docs = vec![
            create_vector_document("a".into(), Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), None)
                .unwrap(),
            create_vector_document("b".into(), Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]), None)
                .unwrap(),
        ];
        index.build_index(docs).unwrap();
        assert_eq!(index.stats.total_vectors, 2);

        index.remove("a").unwrap();
        assert_eq!(index.stats.total_vectors, 1);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_ivf_nprobe_affects_recall() {
        let dim = 16;
        let n = 200;
        let docs = random_docs(n, dim);
        let query = Array1::from_iter((0..dim).map(|_| rand::thread_rng().gen_range(-1.0..1.0)));

        // Low nprobe
        let mut low = IVFIndex::new(IVFConfig {
            dimension: dim,
            nlist: 16,
            nprobe: 1,
            metric: DistanceMetric::Euclidean,
        });
        low.build_index(docs.clone()).unwrap();
        let low_results = low.search(&query, 10).unwrap();

        // High nprobe (search all clusters = brute force)
        let mut high = IVFIndex::new(IVFConfig {
            dimension: dim,
            nlist: 16,
            nprobe: 16,
            metric: DistanceMetric::Euclidean,
        });
        high.build_index(docs).unwrap();
        let high_results = high.search(&query, 10).unwrap();

        // High nprobe should find better (closer) vectors
        assert!(
            high_results[0].distance <= low_results[0].distance + 1e-6,
            "Full probe should be at least as good as single probe"
        );
    }
}
