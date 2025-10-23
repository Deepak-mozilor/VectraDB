use super::{AdvancedSearch, SearchResult, SearchStats};
use ndarray::Array1;
use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

/// HNSW (Hierarchical Navigable Small World) index implementation
pub struct HNSWIndex {
    graph: DiGraph<VectorDocument, f32>,
    entry_point: Option<NodeIndex>,
    max_connections: usize,
    ef_construction: usize,
    m: usize,
    stats: SearchStats,
    dimension: usize,
}

/// Node in the HNSW graph
#[derive(Debug, Clone)]
struct HNSWNode {
    document: VectorDocument,
    level: usize,
    connections: Vec<NodeIndex>,
}

impl HNSWIndex {
    /// Create a new HNSW index
    pub fn new(dimension: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            graph: DiGraph::new(),
            entry_point: None,
            max_connections: m,
            ef_construction,
            m,
            stats: SearchStats::default(),
            dimension,
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

    /// Search for candidates in a specific layer
    fn search_layer(
        &self,
        query: &Array1<f32>,
        entry_points: Vec<NodeIndex>,
        ef: usize,
        layer: usize,
    ) -> Vec<SearchResult> {
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();

        // Initialize with entry points
        for &ep in &entry_points {
            if let Some(document) = self.graph.node_weight(ep) {
                let distance = self.calculate_distance(query, &document.data);
                candidates.push(SearchResult {
                    id: document.metadata.id.clone(),
                    distance,
                    similarity: 1.0 / (1.0 + distance),
                });
                visited.insert(ep);
            }
        }

        let mut results = Vec::new();

        while let Some(current) = candidates.pop() {
            if results.len() >= ef {
                break;
            }

            results.push(current.clone());

            // Find the node index for this result
            let node_idx = self.graph.node_indices().find(|&idx| {
                self.graph
                    .node_weight(idx)
                    .map(|doc| doc.metadata.id == current.id)
                    .unwrap_or(false)
            });

            if let Some(idx) = node_idx {
                // Explore neighbors
                for neighbor_idx in self.graph.neighbors(idx) {
                    if visited.contains(&neighbor_idx) {
                        continue;
                    }

                    if let Some(neighbor_doc) = self.graph.node_weight(neighbor_idx) {
                        let distance = self.calculate_distance(query, &neighbor_doc.data);
                        candidates.push(SearchResult {
                            id: neighbor_doc.metadata.id.clone(),
                            distance,
                            similarity: 1.0 / (1.0 + distance),
                        });
                        visited.insert(neighbor_idx);
                    }
                }
            }
        }

        results
    }

    /// Calculate Euclidean distance between vectors
    fn calculate_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// Select neighbors using simple heuristic
    fn select_neighbors(&self, candidates: &[SearchResult], m: usize) -> Vec<String> {
        let mut selected = Vec::new();
        let mut used = HashSet::new();

        for candidate in candidates.iter().take(m) {
            if !used.contains(&candidate.id) {
                selected.push(candidate.id.clone());
                used.insert(candidate.id.clone());
            }
        }

        selected
    }

    /// Insert a new vector into the HNSW index
    fn insert_vector(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }

        let node_idx = self.graph.add_node(document.clone());
        let level = self.calculate_level();

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            return Ok(());
        }

        let mut entry_points = vec![self.entry_point.unwrap()];

        // Search from top layer down
        for current_level in (0..=level).rev() {
            let candidates = self.search_layer(
                &document.data,
                entry_points.clone(),
                self.ef_construction,
                current_level,
            );
            let neighbors = self.select_neighbors(&candidates, self.max_connections);

            // Connect to selected neighbors
            for neighbor_id in neighbors {
                if let Some(neighbor_idx) = self.graph.node_indices().find(|&idx| {
                    self.graph
                        .node_weight(idx)
                        .map(|doc| doc.metadata.id == neighbor_id)
                        .unwrap_or(false)
                }) {
                    let distance =
                        self.calculate_distance(&document.data, &self.graph[node_idx].data);
                    self.graph.add_edge(node_idx, neighbor_idx, distance);
                    self.graph.add_edge(neighbor_idx, node_idx, distance);
                }
            }

            // Update entry points for next level
            entry_points = candidates
                .iter()
                .take(self.max_connections)
                .filter_map(|result| {
                    self.graph.node_indices().find(|&idx| {
                        self.graph
                            .node_weight(idx)
                            .map(|doc| doc.metadata.id == result.id)
                            .unwrap_or(false)
                    })
                })
                .collect();
        }

        // Update entry point if this is the highest level
        if level > 0 {
            self.entry_point = Some(node_idx);
        }

        Ok(())
    }
}

impl AdvancedSearch for HNSWIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let start_time = Instant::now();

        if let Some(entry) = self.entry_point {
            let mut entry_points = vec![entry];
            let candidates = self.search_layer(query, entry_points, k * 2, 0);
            let results = candidates.into_iter().take(k).collect();

            // Update stats
            let search_time = start_time.elapsed().as_millis() as f64;
            let mut stats = self.stats.clone();
            stats.average_search_time_ms = (stats.average_search_time_ms + search_time) / 2.0;

            Ok(results)
        } else {
            Ok(vec![])
        }
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        self.insert_vector(document)?;
        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        let node_idx = self
            .graph
            .node_indices()
            .find(|&idx| {
                self.graph
                    .node_weight(idx)
                    .map(|doc| doc.metadata.id == id)
                    .unwrap_or(false)
            })
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        self.graph.remove_node(node_idx);

        if self.entry_point == Some(node_idx) {
            self.entry_point = self.graph.node_indices().next();
        }

        self.stats.total_vectors -= 1;
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
            index_size_bytes: self.graph.node_count() * self.dimension * 4, // Rough estimate
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
        let index = HNSWIndex::new(3, 16, 200);
        assert_eq!(index.dimension, 3);
        assert_eq!(index.max_connections, 16);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HNSWIndex::new(3, 4, 50);

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
        let mut index = HNSWIndex::new(3, 16, 200);
        let doc = create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 2.0]), None)
            .unwrap();

        assert!(index.insert(doc).is_err());
    }
}
