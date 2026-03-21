use super::{AdvancedSearch, DistanceMetric, SearchResult, SearchStats};
use ndarray::Array1;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

// ---- Fast distance helpers (no allocation, auto-vectorized) ----

/// L2 squared distance computed in 4-wide chunks for auto-vectorization.
#[inline]
fn fast_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        sum0 += d * d;
    }
    (sum0 + sum1 + sum2 + sum3).sqrt()
}

/// Dot product in 4-wide chunks.
#[inline]
fn fast_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        sum0 += a[j] * b[j];
        sum1 += a[j + 1] * b[j + 1];
        sum2 += a[j + 2] * b[j + 2];
        sum3 += a[j + 3] * b[j + 3];
    }
    for i in (chunks * 4)..n {
        sum0 += a[i] * b[i];
    }
    sum0 + sum1 + sum2 + sum3
}

/// Cosine distance: 1 - cosine_similarity.
#[inline]
fn fast_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}

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

    /// Greedy beam search on the graph. Returns up to `ef` closest results.
    fn search_layer(
        &self,
        query: &Array1<f32>,
        entry_points: &[NodeIndex],
        ef: usize,
    ) -> Vec<HNSWEntry> {
        let mut candidates: BinaryHeap<Reverse<HNSWEntry>> = BinaryHeap::new(); // min-heap
        let mut results: BinaryHeap<HNSWEntry> = BinaryHeap::new(); // max-heap
        let mut visited = HashSet::new();

        for &ep in entry_points {
            if let Some(doc) = self.graph.node_weight(ep) {
                let d = self.calculate_distance(query, &doc.data);
                candidates.push(Reverse(HNSWEntry {
                    distance: d,
                    node: ep,
                }));
                results.push(HNSWEntry {
                    distance: d,
                    node: ep,
                });
                visited.insert(ep);
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
                if !visited.insert(nbr) {
                    continue;
                }
                if let Some(doc) = self.graph.node_weight(nbr) {
                    let d = self.calculate_distance(query, &doc.data);
                    if results.len() < ef || d < results.peek().unwrap().distance {
                        candidates.push(Reverse(HNSWEntry {
                            distance: d,
                            node: nbr,
                        }));
                        results.push(HNSWEntry {
                            distance: d,
                            node: nbr,
                        });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<HNSWEntry> = results.into_vec();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out
    }

    /// Calculate distance between vectors using raw slices to avoid allocation.
    fn calculate_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let a = a.as_slice().unwrap();
        let b = b.as_slice().unwrap();
        match self.metric {
            DistanceMetric::Euclidean => fast_l2_distance(a, b),
            DistanceMetric::Cosine => fast_cosine_distance(a, b),
            DistanceMetric::DotProduct => -fast_dot(a, b),
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
        let node_idx = self.graph.add_node(document.clone());
        self.id_to_node.insert(id.clone(), node_idx);
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

        // After remove_node, petgraph may swap indices — rebuild the map
        self.id_to_node.clear();
        for idx in self.graph.node_indices() {
            self.id_to_node
                .insert(self.graph[idx].metadata.id.clone(), idx);
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
