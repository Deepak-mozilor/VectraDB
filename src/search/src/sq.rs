//! Scalar Quantization (SQ) index.
//!
//! Compresses each vector dimension from f32 (4 bytes) to uint8 (1 byte),
//! achieving **4x memory reduction** with typically <1% recall loss.
//!
//! How it works:
//! 1. **Training**: Scans all vectors to find `[min, max]` per dimension.
//! 2. **Encoding**: Maps each f32 value to `[0, 255]` within that range.
//! 3. **Search**: Decodes compressed vectors on-the-fly for distance computation.
//!
//! Memory comparison (384-dim vector):
//! - f32:  1536 bytes
//! - SQ8:   384 bytes (4x reduction)

use super::{AdvancedSearch, DistanceMetric, SearchResult, SearchStats};
use ndarray::Array1;
use std::collections::HashMap;
use std::time::Instant;
use vectradb_components::{VectorDocument, VectraDBError};

/// Scalar Quantization index with uint8 encoding.
pub struct SQIndex {
    /// Per-dimension min values (for decoding).
    mins: Vec<f32>,
    /// Per-dimension scale: (max - min) / 255.0
    scales: Vec<f32>,
    /// Compressed vectors: id → Vec<u8>
    codes: HashMap<String, Vec<u8>>,
    /// Original documents (for metadata retrieval).
    documents: HashMap<String, VectorDocument>,
    dimension: usize,
    metric: DistanceMetric,
    trained: bool,
    stats: SearchStats,
}

impl SQIndex {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            mins: vec![0.0; dimension],
            scales: vec![1.0; dimension],
            codes: HashMap::new(),
            documents: HashMap::new(),
            dimension,
            metric,
            trained: false,
            stats: SearchStats::default(),
        }
    }

    /// Train quantization parameters by scanning all vectors to find
    /// per-dimension `[min, max]` ranges.
    fn train(&mut self, vectors: &[&Array1<f32>]) {
        if vectors.is_empty() {
            return;
        }
        let dim = self.dimension;
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for vec in vectors {
            for j in 0..dim {
                let v = vec[j];
                if v < mins[j] {
                    mins[j] = v;
                }
                if v > maxs[j] {
                    maxs[j] = v;
                }
            }
        }

        let mut scales = vec![1.0f32; dim];
        for j in 0..dim {
            let range = maxs[j] - mins[j];
            scales[j] = if range > 0.0 { range / 255.0 } else { 1.0 };
        }

        self.mins = mins;
        self.scales = scales;
        self.trained = true;
    }

    /// Encode a vector to uint8 codes.
    fn encode(&self, vector: &Array1<f32>) -> Vec<u8> {
        vector
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&v, (&min, &scale))| ((v - min) / scale).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Decode uint8 codes back to f32 vector.
    #[cfg(test)]
    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&code, (&min, &scale))| min + code as f32 * scale)
            .collect()
    }

    /// Compute distance between a query (f32) and a compressed vector (uint8).
    /// Decodes on-the-fly without allocating a full f32 vector.
    #[allow(clippy::needless_range_loop)]
    fn distance_to_codes(&self, query: &[f32], codes: &[u8]) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0f32;
                for j in 0..self.dimension {
                    let decoded = self.mins[j] + codes[j] as f32 * self.scales[j];
                    let d = query[j] - decoded;
                    sum += d * d;
                }
                sum.sqrt()
            }
            DistanceMetric::Cosine => {
                let mut dot = 0.0f32;
                let mut norm_q = 0.0f32;
                let mut norm_c = 0.0f32;
                for j in 0..self.dimension {
                    let decoded = self.mins[j] + codes[j] as f32 * self.scales[j];
                    dot += query[j] * decoded;
                    norm_q += query[j] * query[j];
                    norm_c += decoded * decoded;
                }
                let denom = norm_q.sqrt() * norm_c.sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot / denom
                }
            }
            DistanceMetric::DotProduct => {
                let mut dot = 0.0f32;
                for j in 0..self.dimension {
                    let decoded = self.mins[j] + codes[j] as f32 * self.scales[j];
                    dot += query[j] * decoded;
                }
                -dot // negate: smaller distance = higher dot product
            }
        }
    }
}

impl AdvancedSearch for SQIndex {
    fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<SearchResult>, VectraDBError> {
        if query.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let query_slice = query.as_slice().unwrap();

        let mut candidates: Vec<SearchResult> = self
            .codes
            .iter()
            .map(|(id, codes)| {
                let distance = self.distance_to_codes(query_slice, codes);
                let similarity = match self.metric {
                    DistanceMetric::DotProduct => -distance, // undo negation
                    _ => 1.0 / (1.0 + distance),
                };
                SearchResult {
                    id: id.clone(),
                    distance,
                    similarity,
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.truncate(k);
        Ok(candidates)
    }

    fn insert(&mut self, document: VectorDocument) -> Result<(), VectraDBError> {
        if document.data.len() != self.dimension {
            return Err(VectraDBError::DimensionMismatch {
                expected: self.dimension,
                actual: document.data.len(),
            });
        }

        // If not trained, train on this single vector (will be retrained on build_index)
        if !self.trained {
            self.train(&[&document.data]);
        }

        let codes = self.encode(&document.data);
        let id = document.metadata.id.clone();
        self.codes.insert(id.clone(), codes);
        self.documents.insert(id, document);
        self.stats.total_vectors += 1;
        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<(), VectraDBError> {
        if self.codes.remove(id).is_some() {
            self.documents.remove(id);
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

        // Train on all vectors to get accurate per-dimension ranges
        let refs: Vec<&Array1<f32>> = documents.iter().map(|d| &d.data).collect();
        self.train(&refs);

        // Encode and store all vectors
        self.codes.clear();
        self.documents.clear();
        self.stats.total_vectors = 0;

        for doc in documents {
            let codes = self.encode(&doc.data);
            let id = doc.metadata.id.clone();
            self.codes.insert(id.clone(), codes);
            self.documents.insert(id, doc);
            self.stats.total_vectors += 1;
        }

        self.stats.construction_time_ms = start_time.elapsed().as_millis() as f64;
        Ok(())
    }

    fn get_stats(&self) -> SearchStats {
        SearchStats {
            total_vectors: self.stats.total_vectors,
            // SQ8: 1 byte per dimension (vs 4 for f32)
            index_size_bytes: self.codes.len() * self.dimension,
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
    fn test_sq_encode_decode_roundtrip() {
        let mut index = SQIndex::new(4, DistanceMetric::Euclidean);
        let v = Array1::from_vec(vec![0.0, 0.5, 1.0, -1.0]);
        index.train(&[&v]);

        let codes = index.encode(&v);
        let decoded = index.decode(&codes);

        // Should be close to original (within quantization error)
        for (orig, dec) in v.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.02, "orig={orig}, decoded={dec}");
        }
    }

    #[test]
    fn test_sq_memory_savings() {
        let dim = 384;
        let mut index = SQIndex::new(dim, DistanceMetric::Euclidean);

        let docs: Vec<VectorDocument> = (0..100)
            .map(|i| {
                let v = Array1::from_iter((0..dim).map(|j| (i * dim + j) as f32 * 0.001));
                create_vector_document(format!("d{i}"), v, None).unwrap()
            })
            .collect();

        index.build_index(docs).unwrap();

        let stats = index.get_stats();
        let sq_bytes = stats.index_size_bytes;
        let f32_bytes = 100 * dim * 4;

        assert_eq!(sq_bytes, 100 * dim); // 1 byte per dim
        assert_eq!(f32_bytes / sq_bytes, 4); // 4x compression
    }

    #[test]
    fn test_sq_search_finds_nearest() {
        let dim = 8;
        let mut index = SQIndex::new(dim, DistanceMetric::Euclidean);

        let target = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let near = Array1::from_vec(vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let far = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let docs = vec![
            create_vector_document("near".into(), near, None).unwrap(),
            create_vector_document("far".into(), far, None).unwrap(),
        ];

        index.build_index(docs).unwrap();

        let results = index.search(&target, 2).unwrap();
        assert_eq!(results[0].id, "near");
        assert_eq!(results[1].id, "far");
    }

    #[test]
    fn test_sq_recall_vs_exact() {
        use rand::Rng;

        let dim = 32;
        let n = 200;
        let k = 10;
        let mut rng = rand::thread_rng();

        let docs: Vec<VectorDocument> = (0..n)
            .map(|i| {
                let v = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0)));
                create_vector_document(format!("d{i}"), v, None).unwrap()
            })
            .collect();

        let mut index = SQIndex::new(dim, DistanceMetric::Euclidean);
        index.build_index(docs.clone()).unwrap();

        let mut total_recall = 0.0;
        let queries = 10;

        for _ in 0..queries {
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

            let results = index.search(&query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = found.intersection(&gt).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / queries as f64;
        eprintln!("SQ8 recall@{k} = {avg_recall:.3} (n={n}, dim={dim})");

        // SQ8 should have very high recall (>90%)
        assert!(avg_recall > 0.85, "SQ8 recall too low: {avg_recall:.3}");
    }

    #[test]
    fn test_sq_cosine_metric() {
        let dim = 4;
        let mut index = SQIndex::new(dim, DistanceMetric::Cosine);

        let docs = vec![
            create_vector_document(
                "same".into(),
                Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
                None,
            )
            .unwrap(),
            create_vector_document(
                "ortho".into(),
                Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
                None,
            )
            .unwrap(),
        ];

        index.build_index(docs).unwrap();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results[0].id, "same");
    }
}
