//! ES4D: Exact Similarity Search via Vector Slicing — adapted for HNSW
//!
//! Implements the ES4D paper's key optimizations on top of HNSW graph navigation:
//!
//! 1. **DET (Dimension-Level Early Termination)**: Computes L2 distance in shards.
//!    If the partial distance exceeds the current cutoff (distance to kth-best result),
//!    computation terminates early — saving CPU on high-dimensional vectors.
//!
//! 2. **Dimension Reordering**: Reorders dimensions by variance (descending) so
//!    high-discriminating dimensions are computed first, causing DET to trigger earlier.
//!
//! 3. **CET (Cluster-Level Early Termination)**: Pre-clusters vectors via k-means.
//!    During HNSW search, candidates whose cluster proximity exceeds the cutoff
//!    are skipped without computing distance.
//!
//! 4. **Cluster Proximity Ordering**: Seeds the HNSW search from a node in the
//!    cluster closest to the query, tightening the cutoff faster.

use super::{AdvancedSearch, SearchResult, SearchStats};
use ndarray::Array1;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

// ============================================================
// Configuration
// ============================================================

/// ES4D configuration parameters.
#[derive(Debug, Clone)]
pub struct ES4DConfig {
    /// Vector dimensionality.
    pub dimension: usize,
    /// Shard length for DET (number of dimensions per shard). Paper recommends ~64.
    pub shard_length: usize,
    /// HNSW max bidirectional connections per node.
    pub m: usize,
    /// HNSW ef parameter used during index construction.
    pub ef_construction: usize,
    /// Enable cluster-level early termination.
    pub enable_cet: bool,
    /// Enable dimension-level early termination.
    pub enable_det: bool,
    /// Enable dimension reordering by variance.
    pub enable_dimension_reorder: bool,
    /// HNSW ef parameter used during search (higher = better recall, slower).
    pub search_ef: usize,
}

impl Default for ES4DConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            shard_length: 64,
            m: 16,
            ef_construction: 200,
            enable_cet: true,
            enable_det: true,
            enable_dimension_reorder: true,
            search_ef: 50,
        }
    }
}

// ============================================================
// Internal helper types
// ============================================================

/// Heap entry for HNSW search. `Ord` compares by distance ascending
/// so that `BinaryHeap` (max-heap) with `Reverse<HeapEntry>` acts as a min-heap
/// and a bare `BinaryHeap<HeapEntry>` acts as a max-heap.
#[derive(Clone)]
struct HeapEntry {
    distance: f32,
    node: NodeIndex,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

/// A cluster of vectors for CET.
#[derive(Debug, Clone)]
struct ES4DCluster {
    centroid: Array1<f32>,
    radius: f32,
    member_ids: Vec<String>,
}

// ============================================================
// ES4D Index
// ============================================================

/// ES4D search index: HNSW graph with dimension-level and cluster-level
/// early termination for accelerated exact similarity search.
pub struct ES4DIndex {
    // -- HNSW graph --
    graph: DiGraph<String, f32>,
    entry_point: Option<NodeIndex>,
    id_to_node: HashMap<String, NodeIndex>,

    // -- Vector storage --
    /// Vectors stored in reordered dimension layout (for DET).
    vectors: HashMap<String, Array1<f32>>,
    /// Original documents (for metadata retrieval).
    documents: HashMap<String, VectorDocument>,

    // -- CET --
    clusters: Vec<ES4DCluster>,
    vector_cluster: HashMap<String, usize>,

    // -- Dimension reordering --
    /// `dimension_order[i]` = the original dimension index placed at position `i`.
    dimension_order: Vec<usize>,

    // -- Config / stats --
    config: ES4DConfig,
    dimension: usize,
    stats: SearchStats,
}

impl ES4DIndex {
    /// Create a new, empty ES4D index.
    pub fn new(config: ES4DConfig) -> Self {
        let dimension = config.dimension;
        let dimension_order: Vec<usize> = (0..dimension).collect(); // identity
        Self {
            graph: DiGraph::new(),
            entry_point: None,
            id_to_node: HashMap::new(),
            vectors: HashMap::new(),
            documents: HashMap::new(),
            clusters: Vec::new(),
            vector_cluster: HashMap::new(),
            dimension_order,
            config,
            dimension,
            stats: SearchStats::default(),
        }
    }

    // --------------------------------------------------------
    // Distance functions
    // --------------------------------------------------------

    /// Full L2 distance (no early termination).
    fn full_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// L2 distance with dimension-level early termination (DET).
    ///
    /// Computes distance in shards of `shard_length` dimensions.
    /// After each shard, checks if the accumulated squared distance already
    /// exceeds `cutoff²`. If so, returns `None` — the candidate cannot be
    /// in the top-k result.
    fn distance_det(
        a: &Array1<f32>,
        b: &Array1<f32>,
        cutoff: f32,
        shard_length: usize,
    ) -> Option<f32> {
        let cutoff_sq = cutoff * cutoff;
        let mut partial_sq: f32 = 0.0;
        let dim = a.len();

        let mut i = 0;
        while i < dim {
            let end = (i + shard_length).min(dim);
            for j in i..end {
                let diff = a[j] - b[j];
                partial_sq += diff * diff;
            }
            // DET check after each shard
            if partial_sq > cutoff_sq {
                return None;
            }
            i = end;
        }
        Some(partial_sq.sqrt())
    }

    // --------------------------------------------------------
    // Dimension reordering
    // --------------------------------------------------------

    /// Compute a global dimension reordering by variance (descending).
    /// Dimensions with higher variance are placed first so that partial
    /// distances grow faster, making DET more effective.
    fn compute_dimension_order(vectors: &[&Array1<f32>], dim: usize) -> Vec<usize> {
        if vectors.is_empty() || dim == 0 {
            return (0..dim).collect();
        }
        let n = vectors.len() as f32;

        // Mean per dimension
        let mut means = vec![0.0f32; dim];
        for vec in vectors {
            for j in 0..dim {
                means[j] += vec[j];
            }
        }
        for m in means.iter_mut() {
            *m /= n;
        }

        // Variance per dimension
        let mut variances = vec![0.0f32; dim];
        for vec in vectors {
            for j in 0..dim {
                let diff = vec[j] - means[j];
                variances[j] += diff * diff;
            }
        }

        // Sort dimensions by variance descending
        let mut indices: Vec<usize> = (0..dim).collect();
        indices.sort_by(|&a, &b| variances[b].total_cmp(&variances[a]));
        indices
    }

    /// Reorder a vector according to `dimension_order`.
    fn reorder_vector(vec: &Array1<f32>, order: &[usize]) -> Array1<f32> {
        Array1::from_iter(order.iter().map(|&i| vec[i]))
    }

    // --------------------------------------------------------
    // K-means clustering (for CET)
    // --------------------------------------------------------

    /// Simple k-means clustering. Returns groups of vector IDs.
    fn kmeans(vecs: &[(&str, &Array1<f32>)], k: usize, max_iters: usize) -> Vec<Vec<String>> {
        if vecs.is_empty() || k == 0 {
            return vec![];
        }
        let k = k.min(vecs.len());
        let dim = vecs[0].1.len();

        // Initialise centroids with k random distinct vectors
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Array1<f32>> = Vec::with_capacity(k);
        let mut picked = HashSet::new();
        while centroids.len() < k {
            let idx = rng.gen_range(0..vecs.len());
            if picked.insert(idx) {
                centroids.push(vecs[idx].1.clone());
            }
        }

        let mut assignments = vec![0usize; vecs.len()];

        for _ in 0..max_iters {
            let mut changed = false;

            // Assign each vector to nearest centroid
            for (vi, (_, vec)) in vecs.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f32::INFINITY;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = Self::l2_sq(vec, c);
                    if d < best_d {
                        best_d = d;
                        best = ci;
                    }
                }
                if assignments[vi] != best {
                    assignments[vi] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // Recompute centroids
            let mut sums = vec![Array1::zeros(dim); k];
            let mut counts = vec![0usize; k];
            for (vi, (_, vec)) in vecs.iter().enumerate() {
                let ci = assignments[vi];
                sums[ci] = &sums[ci] + *vec;
                counts[ci] += 1;
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    centroids[ci] = &sums[ci] / counts[ci] as f32;
                }
            }
        }

        // Group by cluster
        let mut groups: Vec<Vec<String>> = vec![Vec::new(); k];
        for (vi, (id, _)) in vecs.iter().enumerate() {
            groups[assignments[vi]].push(id.to_string());
        }
        groups.retain(|g| !g.is_empty());
        groups
    }

    /// Squared L2 distance (no sqrt — used for comparisons).
    fn l2_sq(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff)
    }

    /// Build clusters from the currently stored (reordered) vectors.
    fn build_clusters(&mut self, k: usize) {
        let vec_refs: Vec<(&str, &Array1<f32>)> = self
            .vectors
            .iter()
            .map(|(id, v)| (id.as_str(), v))
            .collect();

        let groups = Self::kmeans(&vec_refs, k, 20);

        self.clusters.clear();
        self.vector_cluster.clear();

        for (ci, members) in groups.into_iter().enumerate() {
            if members.is_empty() {
                continue;
            }

            // Centroid
            let mut centroid = Array1::zeros(self.dimension);
            for id in &members {
                if let Some(v) = self.vectors.get(id) {
                    centroid += v;
                }
            }
            centroid /= members.len() as f32;

            // Radius = max distance from centroid to any member
            let mut radius: f32 = 0.0;
            for id in &members {
                if let Some(v) = self.vectors.get(id) {
                    let d = Self::full_distance(&centroid, v);
                    if d > radius {
                        radius = d;
                    }
                }
            }

            for id in &members {
                self.vector_cluster.insert(id.clone(), ci);
            }

            self.clusters.push(ES4DCluster {
                centroid,
                radius,
                member_ids: members,
            });
        }
    }

    // --------------------------------------------------------
    // HNSW graph operations
    // --------------------------------------------------------

    /// Insert a single vector into the HNSW graph.
    /// The vector must already be stored in `self.vectors`.
    fn insert_to_graph(&mut self, id: &str) -> Result<(), VectraDBError> {
        let node_idx = self.graph.add_node(id.to_string());
        self.id_to_node.insert(id.to_string(), node_idx);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            return Ok(());
        }

        let vec = self
            .vectors
            .get(id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?
            .clone();

        // Find neighbours via greedy search (no DET during construction
        // — we want accurate connections).
        let neighbours =
            self.search_graph_internal(&vec, self.config.ef_construction, false, self.config.m);

        // Connect to the M nearest neighbours (bidirectional)
        for entry in neighbours.iter().take(self.config.m) {
            if let Some(&nbr_idx) = self.id_to_node.get(&self.graph[entry.node]) {
                self.graph.add_edge(node_idx, nbr_idx, entry.distance);
                self.graph.add_edge(nbr_idx, node_idx, entry.distance);
            }
        }

        // Prune neighbours that exceed 2*M connections
        let neighbours_to_prune: Vec<NodeIndex> = neighbours
            .iter()
            .take(self.config.m)
            .filter_map(|e| self.id_to_node.get(&self.graph[e.node]).copied())
            .collect();
        for nbr_idx in neighbours_to_prune {
            self.prune_connections(nbr_idx);
        }

        Ok(())
    }

    /// Keep at most `2*M` connections for a node, retaining the closest ones.
    fn prune_connections(&mut self, node: NodeIndex) {
        let max_conn = self.config.m * 2;
        let mut edges: Vec<(petgraph::graph::EdgeIndex, f32)> = self
            .graph
            .edges(node)
            .map(|e| (e.id(), *e.weight()))
            .collect();

        if edges.len() <= max_conn {
            return;
        }

        edges.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Remove edges beyond max_conn (farthest first)
        for &(edge_id, _) in edges.iter().skip(max_conn) {
            self.graph.remove_edge(edge_id);
        }
    }

    // --------------------------------------------------------
    // Core search
    // --------------------------------------------------------

    /// HNSW greedy search with optional DET and CET.
    ///
    /// When `use_early_termination` is true, DET is applied during distance
    /// computation and CET skips candidates whose clusters are too far.
    /// `k` is the number of results the caller needs — DET cutoff only
    /// activates once we have at least `k` results so we don't prune too early.
    fn search_graph_internal(
        &self,
        query: &Array1<f32>,
        ef: usize,
        use_early_termination: bool,
        k: usize,
    ) -> Vec<HeapEntry> {
        let entry = match self.entry_point {
            Some(ep) => ep,
            None => return vec![],
        };

        // Pre-compute cluster distances for CET (O(num_clusters * dim), done once)
        let cluster_dists: Vec<f32> = if use_early_termination && self.config.enable_cet {
            self.clusters
                .iter()
                .map(|c| Self::full_distance(query, &c.centroid))
                .collect()
        } else {
            vec![]
        };

        // Candidates: min-heap (process closest first)
        let mut candidates: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        // Results: max-heap (evict farthest, peek = current worst)
        let mut results: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut visited: HashSet<NodeIndex> = HashSet::new();

        // Seed with entry point
        if let Some(entry_vec) = self.vectors.get(&self.graph[entry]) {
            let d = Self::full_distance(query, entry_vec);
            candidates.push(Reverse(HeapEntry {
                distance: d,
                node: entry,
            }));
            results.push(HeapEntry {
                distance: d,
                node: entry,
            });
            visited.insert(entry);
        }

        // Optionally seed with a node from the cluster closest to the query
        // (cluster proximity ordering)
        if use_early_termination && self.config.enable_cet && !cluster_dists.is_empty() {
            if let Some((ci, _)) = cluster_dists
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
            {
                if let Some(first_id) = self.clusters[ci].member_ids.first() {
                    if let Some(&node) = self.id_to_node.get(first_id) {
                        if visited.insert(node) {
                            if let Some(v) = self.vectors.get(first_id) {
                                let d = Self::full_distance(query, v);
                                candidates.push(Reverse(HeapEntry { distance: d, node }));
                                results.push(HeapEntry { distance: d, node });
                            }
                        }
                    }
                }
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            // Stop if current candidate is farther than the ef-th best result
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            // Explore neighbours
            for nbr_idx in self.graph.neighbors(current.node) {
                if !visited.insert(nbr_idx) {
                    continue;
                }

                let nbr_id = &self.graph[nbr_idx];

                // --- CET check ---
                if use_early_termination
                    && self.config.enable_cet
                    && !cluster_dists.is_empty()
                    && results.len() >= k
                {
                    if let Some(&cid) = self.vector_cluster.get(nbr_id) {
                        if cid < cluster_dists.len() {
                            let proximity = cluster_dists[cid] - self.clusters[cid].radius;
                            let cutoff = results.peek().map(|r| r.distance).unwrap_or(f32::MAX);
                            if proximity > cutoff {
                                continue; // cluster too far — skip
                            }
                        }
                    }
                }

                let nbr_vec = match self.vectors.get(nbr_id) {
                    Some(v) => v,
                    None => continue,
                };

                // --- Compute distance (with DET if enabled) ---
                // Only apply DET cutoff once we have at least k results,
                // otherwise we might prune candidates we still need.
                let cutoff = if results.len() >= k {
                    results.peek().map(|r| r.distance).unwrap_or(f32::MAX)
                } else {
                    f32::MAX
                };

                let dist = if use_early_termination && self.config.enable_det {
                    match Self::distance_det(query, nbr_vec, cutoff, self.config.shard_length) {
                        Some(d) => d,
                        None => continue, // DET early termination — skip
                    }
                } else {
                    Self::full_distance(query, nbr_vec)
                };

                if results.len() < ef || dist < results.peek().unwrap().distance {
                    candidates.push(Reverse(HeapEntry {
                        distance: dist,
                        node: nbr_idx,
                    }));
                    results.push(HeapEntry {
                        distance: dist,
                        node: nbr_idx,
                    });
                    if results.len() > ef {
                        results.pop(); // evict farthest
                    }
                }
            }
        }

        // Drain into a sorted vec (ascending distance)
        let mut out: Vec<HeapEntry> = results.into_vec();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out
    }
}

// ============================================================
// AdvancedSearch trait implementation
// ============================================================

impl AdvancedSearch for ES4DIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let _start = Instant::now();

        // Reorder query dimensions to match stored vectors
        let q = Self::reorder_vector(query, &self.dimension_order);

        let ef = self.config.search_ef.max(k);
        let entries = self.search_graph_internal(&q, ef, true, k);

        let results = entries
            .into_iter()
            .take(k)
            .map(|e| {
                let id = self.graph[e.node].clone();
                SearchResult {
                    id,
                    distance: e.distance,
                    similarity: 1.0 / (1.0 + e.distance),
                }
            })
            .collect();

        Ok(results)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }

        let id = document.metadata.id.clone();

        // Reorder and store
        let reordered = Self::reorder_vector(&document.data, &self.dimension_order);
        self.vectors.insert(id.clone(), reordered);
        self.documents.insert(id.clone(), document);

        // Assign to nearest cluster (if clusters exist)
        if !self.clusters.is_empty() {
            let vec = self.vectors.get(&id).unwrap();
            let mut best_ci = 0;
            let mut best_d = f32::INFINITY;
            for (ci, cluster) in self.clusters.iter().enumerate() {
                let d = Self::l2_sq(vec, &cluster.centroid);
                if d < best_d {
                    best_d = d;
                    best_ci = ci;
                }
            }
            self.vector_cluster.insert(id.clone(), best_ci);
            self.clusters[best_ci].member_ids.push(id.clone());
            // Update radius if necessary
            let d = Self::full_distance(
                self.vectors.get(&id).unwrap(),
                &self.clusters[best_ci].centroid,
            );
            if d > self.clusters[best_ci].radius {
                self.clusters[best_ci].radius = d;
            }
        }

        // Insert into HNSW graph
        self.insert_to_graph(&id)?;
        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        let node_idx = self
            .id_to_node
            .remove(id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        self.graph.remove_node(node_idx);
        self.vectors.remove(id);
        self.documents.remove(id);

        // Update entry point if removed
        if self.entry_point == Some(node_idx) {
            self.entry_point = self.graph.node_indices().next();
        }

        // Remove from cluster
        if let Some(cid) = self.vector_cluster.remove(id) {
            if cid < self.clusters.len() {
                self.clusters[cid].member_ids.retain(|v| v != id);
            }
        }

        // id_to_node indices may be stale after remove_node (petgraph
        // swaps the last node into the removed slot). Rebuild the map.
        self.id_to_node.clear();
        for idx in self.graph.node_indices() {
            self.id_to_node.insert(self.graph[idx].clone(), idx);
        }

        self.stats.total_vectors -= 1;
        Ok(())
    }

    fn update(&mut self, id: &str, document: VectorDocument) -> Result<(), VectraDBError> {
        self.remove(id)?;
        self.insert(document)
    }

    fn build_index(&mut self, documents: Vec<VectorDocument>) -> Result<(), VectraDBError> {
        let start = Instant::now();

        // Reset
        self.graph = DiGraph::new();
        self.entry_point = None;
        self.id_to_node.clear();
        self.vectors.clear();
        self.documents.clear();
        self.clusters.clear();
        self.vector_cluster.clear();

        if documents.is_empty() {
            return Ok(());
        }

        // 1) Dimension reordering
        if self.config.enable_dimension_reorder {
            let refs: Vec<&Array1<f32>> = documents.iter().map(|d| &d.data).collect();
            self.dimension_order = Self::compute_dimension_order(&refs, self.dimension);
        }

        // 2) Reorder & store vectors
        for doc in &documents {
            let reordered = Self::reorder_vector(&doc.data, &self.dimension_order);
            self.vectors.insert(doc.metadata.id.clone(), reordered);
            self.documents.insert(doc.metadata.id.clone(), doc.clone());
        }

        // 3) CET clustering (sqrt(n) clusters, as recommended by the paper)
        if self.config.enable_cet && documents.len() > 1 {
            let num_clusters = ((documents.len() as f64).sqrt() as usize).max(1);
            self.build_clusters(num_clusters);
        }

        // 4) Build HNSW graph
        for doc in &documents {
            self.insert_to_graph(&doc.metadata.id)?;
        }

        self.stats.total_vectors = documents.len();
        self.stats.construction_time_ms = start.elapsed().as_millis() as f64;
        self.stats.index_size_bytes = self.vectors.len() * self.dimension * 4;
        Ok(())
    }

    fn get_stats(&self) -> SearchStats {
        SearchStats {
            total_vectors: self.stats.total_vectors,
            index_size_bytes: self.vectors.len() * self.dimension * 4,
            average_search_time_ms: self.stats.average_search_time_ms,
            construction_time_ms: self.stats.construction_time_ms,
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use vectradb_components::vector_operations::create_vector_document;

    #[test]
    fn test_distance_det_no_termination() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        // cutoff is large — no early termination
        let d = ES4DIndex::distance_det(&a, &b, 100.0, 2);
        assert!(d.is_some());
        assert!((d.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_det_early_termination() {
        let a = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0]);
        let b = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        // cutoff is 5.0 — partial distance after first shard (2 dims) = sqrt(200) ≈ 14.1 > 5
        let d = ES4DIndex::distance_det(&a, &b, 5.0, 2);
        assert!(d.is_none());
    }

    #[test]
    fn test_dimension_reordering() {
        // dim 0: all same (variance=0), dim 1: high variance, dim 2: medium
        let v1 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let v2 = Array1::from_vec(vec![1.0, 10.0, 3.0]);
        let v3 = Array1::from_vec(vec![1.0, -10.0, 5.0]);
        let vecs = vec![&v1, &v2, &v3];

        let order = ES4DIndex::compute_dimension_order(&vecs, 3);
        // dim 1 has highest variance, then dim 2, then dim 0
        assert_eq!(order[0], 1);
        assert_eq!(order[1], 2);
        assert_eq!(order[2], 0);
    }

    #[test]
    fn test_reorder_vector() {
        let v = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let order = vec![2, 0, 1];
        let reordered = ES4DIndex::reorder_vector(&v, &order);
        assert_eq!(reordered[0], 30.0);
        assert_eq!(reordered[1], 10.0);
        assert_eq!(reordered[2], 20.0);
    }

    #[test]
    fn test_es4d_insert_and_search() {
        let config = ES4DConfig {
            dimension: 3,
            shard_length: 2,
            m: 4,
            ef_construction: 50,
            enable_cet: false,
            enable_det: true,
            enable_dimension_reorder: false,
            search_ef: 50,
        };
        let mut index = ES4DIndex::new(config);

        let doc1 =
            create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]), None)
                .unwrap();
        let doc2 =
            create_vector_document("2".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0]), None)
                .unwrap();
        let doc3 =
            create_vector_document("3".to_string(), Array1::from_vec(vec![1.0, 1.0, 0.0]), None)
                .unwrap();

        index.insert(doc1).unwrap();
        index.insert(doc2).unwrap();
        index.insert(doc3).unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_es4d_build_index_full() {
        let config = ES4DConfig {
            dimension: 3,
            shard_length: 2,
            m: 4,
            ef_construction: 50,
            enable_cet: true,
            enable_det: true,
            enable_dimension_reorder: true,
            search_ef: 50,
        };
        let mut index = ES4DIndex::new(config);

        let docs: Vec<VectorDocument> = (0..20)
            .map(|i| {
                let mut rng = rand::thread_rng();
                let v = Array1::from_iter((0..3).map(|_| rng.gen::<f32>()));
                create_vector_document(format!("v{}", i), v, None).unwrap()
            })
            .collect();

        index.build_index(docs).unwrap();
        assert_eq!(index.stats.total_vectors, 20);
        assert!(!index.clusters.is_empty());

        let query = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance ascending
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }

    #[test]
    fn test_es4d_dimension_mismatch() {
        let config = ES4DConfig {
            dimension: 3,
            ..Default::default()
        };
        let mut index = ES4DIndex::new(config);
        let doc = create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 2.0]), None)
            .unwrap();
        assert!(index.insert(doc).is_err());
    }

    #[test]
    fn test_es4d_remove() {
        let config = ES4DConfig {
            dimension: 3,
            shard_length: 2,
            m: 4,
            ef_construction: 50,
            enable_cet: false,
            enable_det: true,
            enable_dimension_reorder: false,
            search_ef: 50,
        };
        let mut index = ES4DIndex::new(config);

        let doc =
            create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]), None)
                .unwrap();
        index.insert(doc).unwrap();
        assert_eq!(index.stats.total_vectors, 1);

        index.remove("1").unwrap();
        assert_eq!(index.stats.total_vectors, 0);
    }

    #[test]
    fn test_kmeans_basic() {
        let v1 = Array1::from_vec(vec![0.0, 0.0]);
        let v2 = Array1::from_vec(vec![0.1, 0.1]);
        let v3 = Array1::from_vec(vec![10.0, 10.0]);
        let v4 = Array1::from_vec(vec![10.1, 10.1]);
        let vecs = vec![("a", &v1), ("b", &v2), ("c", &v3), ("d", &v4)];
        let groups = ES4DIndex::kmeans(&vecs, 2, 20);
        assert_eq!(groups.len(), 2);
        // a,b should be in one cluster and c,d in another
        for group in &groups {
            if group.contains(&"a".to_string()) {
                assert!(group.contains(&"b".to_string()));
            }
            if group.contains(&"c".to_string()) {
                assert!(group.contains(&"d".to_string()));
            }
        }
    }
}
