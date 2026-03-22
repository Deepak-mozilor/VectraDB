use super::{AdvancedSearch, DistanceMetric, SearchResult, SearchStats};
use ndarray::Array1;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

// Distance functions use SIMD-accelerated implementations from simd.rs
use super::simd::{simd_cosine_distance, simd_dot, simd_l2_distance};

/// HNSW (Hierarchical Navigable Small World) index implementation
pub struct HNSWIndex {
    graph: DiGraph<VectorDocument, f32>,
    entry_point: Option<NodeIndex>,
    /// O(1) lookup: vector ID → graph node index
    id_to_node: HashMap<String, NodeIndex>,
    max_connections: usize,
    ef_construction: usize,
    search_ef: usize,
    max_level: usize,
    stats: SearchStats,
    dimension: usize,
    metric: DistanceMetric,
    /// Contiguous vector storage: all vectors packed as [v0_d0..v0_dN, v1_d0..v1_dN, ...]
    /// Indexed by node_index * dimension. Cache-friendly for distance computation.
    flat_vectors: Vec<f32>,
}

/// Heap entry ordered by distance for greedy search.
#[derive(Clone)]
struct HNSWEntry {
    distance: f32,
    node: NodeIndex,
}

impl PartialEq for HNSWEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for HNSWEntry {}
impl PartialOrd for HNSWEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HNSWEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

impl HNSWIndex {
    /// Create a new HNSW index
    pub fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        search_ef: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            graph: DiGraph::new(),
            entry_point: None,
            id_to_node: HashMap::new(),
            max_connections: m,
            ef_construction,
            search_ef,
            max_level: 0,
            stats: SearchStats::default(),
            dimension,
            metric,
            flat_vectors: Vec::new(),
        }
    }

    /// Calculate the level for a new node using exponential decay
    fn calculate_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f32>() < 0.5 && level < 16 {
            level += 1;
        }
        level
    }

    /// Greedy beam search on the graph.
    ///
    /// Optimized for cache-friendliness:
    /// - Bit-vector visited set (no hashing overhead)
    /// - Contiguous flat_vectors for distance computation (no pointer chasing)
    /// - Pre-allocated heaps
    fn search_layer(
        &self,
        query: &Array1<f32>,
        entry_points: &[NodeIndex],
        ef: usize,
    ) -> Vec<HNSWEntry> {
        let query_slice = query.as_slice().unwrap();
        let node_count = self.graph.node_count();

        // Bit-vector visited set: O(1) lookup, no hashing, cache-friendly
        let mut visited = vec![false; node_count];

        // Pre-allocated heaps
        let mut candidates: BinaryHeap<Reverse<HNSWEntry>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<HNSWEntry> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            let idx = ep.index();
            if idx < node_count {
                let d = self.distance_to_flat(query_slice, idx);
                candidates.push(Reverse(HNSWEntry { distance: d, node: ep }));
                results.push(HNSWEntry { distance: d, node: ep });
                visited[idx] = true;
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            for nbr in self.graph.neighbors(current.node) {
                let nbr_idx = nbr.index();
                if nbr_idx >= node_count || visited[nbr_idx] {
                    continue;
                }
                visited[nbr_idx] = true;

                let d = self.distance_to_flat(query_slice, nbr_idx);
                if results.len() < ef || d < results.peek().unwrap().distance {
                    candidates.push(Reverse(HNSWEntry { distance: d, node: nbr }));
                    results.push(HNSWEntry { distance: d, node: nbr });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<HNSWEntry> = results.into_vec();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out
    }

    /// Compute distance using the contiguous flat_vectors store (cache-friendly).
    /// Falls back to graph node data if flat_vectors is out of sync.
    #[inline]
    fn distance_to_flat(&self, query: &[f32], node_idx: usize) -> f32 {
        let offset = node_idx * self.dimension;
        let end = offset + self.dimension;

        if end <= self.flat_vectors.len() {
            let b = &self.flat_vectors[offset..end];
            match self.metric {
                DistanceMetric::Euclidean => simd_l2_distance(query, b),
                DistanceMetric::Cosine => simd_cosine_distance(query, b),
                DistanceMetric::DotProduct => -simd_dot(query, b),
            }
        } else {
            // Fallback: read from graph (slower, pointer chasing)
            let node = NodeIndex::new(node_idx);
            if let Some(doc) = self.graph.node_weight(node) {
                let b = doc.data.as_slice().unwrap();
                match self.metric {
                    DistanceMetric::Euclidean => simd_l2_distance(query, b),
                    DistanceMetric::Cosine => simd_cosine_distance(query, b),
                    DistanceMetric::DotProduct => -simd_dot(query, b),
                }
            } else {
                f32::INFINITY
            }
        }
    }

    /// Calculate distance between two Array1 vectors.
    fn calculate_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let a = a.as_slice().unwrap();
        let b = b.as_slice().unwrap();
        match self.metric {
            DistanceMetric::Euclidean => simd_l2_distance(a, b),
            DistanceMetric::Cosine => simd_cosine_distance(a, b),
            DistanceMetric::DotProduct => -simd_dot(a, b),
        }
    }

    /// Keep at most `2*M` edges for a node, retaining closest.
    fn prune_connections(&mut self, node: NodeIndex) {
        let max_conn = self.max_connections * 2;
        let mut edges: Vec<(petgraph::graph::EdgeIndex, f32)> = self
            .graph
            .edges(node)
            .map(|e| (e.id(), *e.weight()))
            .collect();

        if edges.len() <= max_conn {
            return;
        }
        edges.sort_by(|a, b| a.1.total_cmp(&b.1));
        for &(eid, _) in edges.iter().skip(max_conn) {
            self.graph.remove_edge(eid);
        }
    }

    /// Insert a new vector into the HNSW index
    fn insert_vector(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }

        let id = document.metadata.id.clone();
        let vec_data = document.data.as_slice().unwrap().to_vec();
        let node_idx = self.graph.add_node(document.clone());
        self.id_to_node.insert(id.clone(), node_idx);

        // Add to contiguous flat_vectors store
        let required_len = (node_idx.index() + 1) * self.dimension;
        if self.flat_vectors.len() < required_len {
            self.flat_vectors.resize(required_len, 0.0);
        }
        let offset = node_idx.index() * self.dimension;
        self.flat_vectors[offset..offset + self.dimension].copy_from_slice(&vec_data);

        let level = self.calculate_level();

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_level = level;
            return Ok(());
        }

        // Search for neighbours
        let entry = self.entry_point.unwrap();
        let candidates = self.search_layer(&document.data, &[entry], self.ef_construction);

        // Connect to M nearest neighbours (bidirectional)
        for entry in candidates.iter().take(self.max_connections) {
            let nbr_idx = entry.node;
            // Distance between the NEW node and the NEIGHBOUR (fix #24)
            let dist = self.calculate_distance(&document.data, &self.graph[nbr_idx].data);
            self.graph.add_edge(node_idx, nbr_idx, dist);
            self.graph.add_edge(nbr_idx, node_idx, dist);
            self.prune_connections(nbr_idx);
        }

        // Update entry point only if new node has higher level (fix #13)
        if level >= self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_idx);
        }

        Ok(())
    }
}

impl AdvancedSearch for HNSWIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        self.search_with_ef(query, k, self.search_ef)
    }

    fn search_with_ef(
        &self,
        query: &Array1<f32>,
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let results = if let Some(entry) = self.entry_point {
            let ef = ef.max(k);
            let entries = self.search_layer(query, &[entry], ef);
            entries
                .into_iter()
                .take(k)
                .map(|e| {
                    let id = self.graph[e.node].metadata.id.clone();
                    let similarity = match self.metric {
                        DistanceMetric::Cosine => 1.0 - e.distance,
                        DistanceMetric::DotProduct => -e.distance,
                        DistanceMetric::Euclidean => 1.0 / (1.0 + e.distance),
                    };
                    SearchResult {
                        id,
                        distance: e.distance,
                        similarity,
                    }
                })
                .collect()
        } else {
            vec![]
        };

        Ok(results)
    }

    #[cfg(feature = "gpu")]
    fn search_gpu_rerank(
        &self,
        query: &Array1<f32>,
        k: usize,
        rerank_ef: usize,
        gpu: &super::gpu::GpuDistanceEngine,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let entry = match self.entry_point {
            Some(e) => e,
            None => return Ok(vec![]),
        };

        // Step 1: HNSW fetches a large candidate pool on CPU
        let ef = rerank_ef.max(k);
        let candidates = self.search_layer(query, &[entry], ef);

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Collect candidate vectors into a flat buffer for GPU
        let mut ids: Vec<String> = Vec::with_capacity(candidates.len());
        let mut flat_data: Vec<f32> = Vec::with_capacity(candidates.len() * self.dimension);

        for c in &candidates {
            let doc = &self.graph[c.node];
            ids.push(doc.metadata.id.clone());
            if let Some(s) = doc.data.as_slice() {
                flat_data.extend_from_slice(s);
            }
        }

        // Step 3: GPU computes exact distances
        let query_slice = query.as_slice().unwrap();
        let distances = gpu.batch_distances(query_slice, &flat_data, self.dimension, metric);

        // Step 4: Sort by distance and return top-k
        let mut scored: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);

        let results = scored
            .into_iter()
            .map(|(i, dist)| {
                let similarity = match metric {
                    DistanceMetric::Cosine => 1.0 - dist,
                    DistanceMetric::DotProduct => -dist,
                    DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
                };
                SearchResult {
                    id: ids[i].clone(),
                    distance: dist,
                    similarity,
                }
            })
            .collect();

        Ok(results)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        self.insert_vector(document)?;
        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        let node_idx = self
            .id_to_node
            .remove(id)
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        self.graph.remove_node(node_idx);

        if self.entry_point == Some(node_idx) {
            self.entry_point = self.graph.node_indices().next();
        }

        // After remove_node, petgraph may swap indices — rebuild map + flat_vectors
        self.id_to_node.clear();
        self.flat_vectors.clear();
        self.flat_vectors
            .resize(self.graph.node_count() * self.dimension, 0.0);
        for idx in self.graph.node_indices() {
            let doc = &self.graph[idx];
            self.id_to_node
                .insert(doc.metadata.id.clone(), idx);
            if let Some(s) = doc.data.as_slice() {
                let offset = idx.index() * self.dimension;
                self.flat_vectors[offset..offset + self.dimension].copy_from_slice(s);
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
        let start_time = Instant::now();

        for document in documents {
            self.insert_vector(document)?;
        }

        self.stats.construction_time_ms = start_time.elapsed().as_millis() as f64;
        self.stats.total_vectors = self.graph.node_count();

        Ok(())
    }

    fn get_stats(&self) -> SearchStats {
        SearchStats {
            total_vectors: self.stats.total_vectors,
            index_size_bytes: self.graph.node_count() * self.dimension * 4,
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
    fn test_hnsw_creation() {
        let index = HNSWIndex::new(3, 16, 200, 50, DistanceMetric::Euclidean);
        assert_eq!(index.dimension, 3);
        assert_eq!(index.max_connections, 16);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HNSWIndex::new(3, 4, 50, 50, DistanceMetric::Euclidean);

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
        assert_eq!(results[0].id, "1"); // Should be most similar
    }

    #[test]
    fn test_hnsw_dimension_mismatch() {
        let mut index = HNSWIndex::new(3, 16, 200, 50, DistanceMetric::Euclidean);
        let doc = create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 2.0]), None)
            .unwrap();

        assert!(index.insert(doc).is_err());
    }
}
