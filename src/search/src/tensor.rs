//! TensorSearch: Parallel Similarity Search on Tensors
//!
//! Implements the TensorSearch paper (IEEE Big Data 2024) concepts:
//!
//! - **Basic TensorSearch**: Compare a pattern tensor against a database of
//!   reference tensors, returning the top-k most similar.
//!
//! - **Shifting TensorSearch**: Slide a pattern tensor along one dimension of a
//!   large reference tensor (template matching), returning top-k offsets.
//!
//! - **Weighted similarity**: Per-slice weights for emphasizing certain subspaces.
//!
//! - **Cache-efficient computation**: Reorders the computation so pattern data
//!   stays in CPU cache while iterating over offsets (1.3x improvement per paper).

use std::collections::HashMap;
use std::sync::RwLock;
use vectradb_components::tensor::*;
use vectradb_components::VectraDBError;

// ============================================================
// TensorSearch engine
// ============================================================

/// In-memory tensor store + search engine.
pub struct TensorSearchEngine {
    /// Stored reference tensors.
    tensors: RwLock<HashMap<String, TensorDocument>>,
}

impl TensorSearchEngine {
    pub fn new() -> Self {
        Self {
            tensors: RwLock::new(HashMap::new()),
        }
    }

    // -- CRUD --

    pub fn insert(&self, doc: TensorDocument) -> Result<(), VectraDBError> {
        let mut store = self.tensors.write().unwrap_or_else(|e| e.into_inner());
        store.insert(doc.metadata.id.clone(), doc);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<TensorDocument, VectraDBError> {
        let store = self.tensors.read().unwrap_or_else(|e| e.into_inner());
        store
            .get(id)
            .cloned()
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
    }

    pub fn delete(&self, id: &str) -> Result<(), VectraDBError> {
        let mut store = self.tensors.write().unwrap_or_else(|e| e.into_inner());
        store
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })
    }

    pub fn list_ids(&self) -> Vec<String> {
        let store = self.tensors.read().unwrap_or_else(|e| e.into_inner());
        store.keys().cloned().collect()
    }

    pub fn count(&self) -> usize {
        let store = self.tensors.read().unwrap_or_else(|e| e.into_inner());
        store.len()
    }

    // --------------------------------------------------------
    // Basic TensorSearch (paper Section III-A, Eq. 1-2)
    // --------------------------------------------------------

    /// Compare `pattern` against all stored tensors.
    /// Returns top-k most similar, sorted by similarity descending.
    ///
    /// `agg_dim`: dimension along which to aggregate weighted similarities.
    /// `weights`: optional per-slice weights along `agg_dim`.
    pub fn basic_search(
        &self,
        pattern: &TensorData,
        agg_dim: usize,
        weights: Option<&[f32]>,
        metric: TensorSimilarityMetric,
        top_k: usize,
    ) -> Result<Vec<TensorSearchResult>, VectraDBError> {
        let store = self.tensors.read().unwrap_or_else(|e| e.into_inner());

        let mut results: Vec<TensorSearchResult> = store
            .values()
            .filter(|doc| doc.tensor.shape() == pattern.shape())
            .map(|doc| {
                let sim =
                    weighted_tensor_similarity(pattern, &doc.tensor, agg_dim, weights, metric);
                TensorSearchResult {
                    id: doc.metadata.id.clone(),
                    similarity: sim,
                    offset: None,
                    metadata: doc.metadata.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        results.truncate(top_k);
        Ok(results)
    }

    // --------------------------------------------------------
    // Shifting TensorSearch (paper Section III-A, Eq. 3)
    // --------------------------------------------------------

    /// Slide `pattern` along dimension `shift_dim` of the stored tensor `reference_id`.
    /// Returns top-k offsets with highest similarity.
    ///
    /// Uses cache-efficient computation order (paper Section III-C).
    pub fn shifting_search(
        &self,
        pattern: &TensorData,
        reference_id: &str,
        shift_dim: usize,
        agg_dim: usize,
        weights: Option<&[f32]>,
        metric: TensorSimilarityMetric,
        top_k: usize,
    ) -> Result<Vec<TensorSearchResult>, VectraDBError> {
        let store = self.tensors.read().unwrap_or_else(|e| e.into_inner());
        let reference = store
            .get(reference_id)
            .ok_or_else(|| VectraDBError::VectorNotFound {
                id: reference_id.to_string(),
            })?;

        let results = shifting_search_cache_efficient(
            pattern,
            &reference.tensor,
            shift_dim,
            agg_dim,
            weights,
            metric,
            top_k,
        )?;

        // Convert to TensorSearchResult
        let out = results
            .into_iter()
            .map(|(offset, sim)| TensorSearchResult {
                id: reference_id.to_string(),
                similarity: sim,
                offset: Some(offset),
                metadata: reference.metadata.clone(),
            })
            .collect();

        Ok(out)
    }
}

impl Default for TensorSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Core similarity algorithms
// ============================================================

/// Weighted tensor similarity (paper Eq. 2).
///
/// Splits both tensors along `agg_dim`, computes per-slice similarity,
/// and aggregates with optional weights.
pub fn weighted_tensor_similarity(
    pattern: &TensorData,
    reference: &TensorData,
    agg_dim: usize,
    weights: Option<&[f32]>,
    metric: TensorSimilarityMetric,
) -> f32 {
    let num_slices = pattern.shape()[agg_dim];
    let mut total_sim = 0.0f32;
    let mut total_weight = 0.0f32;

    // Fast path for 2D tensors aggregating along dim 1 (columns)
    if pattern.rank() == 2 && agg_dim == 1 {
        let rows = pattern.shape()[0];
        for row in 0..rows {
            let p_row = pattern.row_2d(row);
            let r_row = reference.row_2d(row);
            let sim = slice_similarity(p_row, r_row, metric);
            let w = weights.map_or(1.0, |ws| ws[row]);
            total_sim += w * sim;
            total_weight += w;
        }
    } else {
        // General case: use fiber extraction
        let mut fixed = vec![0usize; pattern.rank()];
        for slice_idx in 0..num_slices {
            fixed[agg_dim] = slice_idx;
            // For the aggregation dimension, we fix slice_idx.
            // We need the full subspace. For simplicity, flatten and compute.
            let p_slice = pattern.fiber(agg_dim, &fixed);
            let r_slice = reference.fiber(agg_dim, &fixed);
            let sim = slice_similarity(&p_slice, &r_slice, metric);
            let w = weights.map_or(1.0, |ws| ws[slice_idx]);
            total_sim += w * sim;
            total_weight += w;
        }
    }

    if total_weight > 0.0 {
        total_sim / total_weight
    } else {
        0.0
    }
}

/// Cache-efficient shifting tensor search (paper Section III-C, Fig. 3b).
///
/// Instead of computing full similarity at each offset (poor cache reuse),
/// we fix one slice of the pattern and iterate over ALL offsets for that slice.
/// The pattern slice stays in L1/L2 cache across offset iterations.
pub fn shifting_search_cache_efficient(
    pattern: &TensorData,
    reference: &TensorData,
    shift_dim: usize,
    agg_dim: usize,
    weights: Option<&[f32]>,
    metric: TensorSimilarityMetric,
    top_k: usize,
) -> Result<Vec<(usize, f32)>, VectraDBError> {
    if shift_dim >= pattern.rank() || agg_dim >= pattern.rank() {
        return Err(VectraDBError::InvalidVector);
    }

    let p_extent = pattern.shape()[shift_dim];
    let r_extent = reference.shape()[shift_dim];
    if r_extent < p_extent {
        return Err(VectraDBError::DimensionMismatch {
            expected: p_extent,
            actual: r_extent,
        });
    }
    let num_positions = r_extent - p_extent + 1;
    let num_slices = pattern.shape()[agg_dim];

    // Accumulate similarity for each offset position
    let mut similarities = vec![0.0f32; num_positions];
    let mut total_weight = 0.0f32;

    // ---- Cache-efficient loop (paper Fig. 3b) ----
    // Outer: iterate over slices (pattern slice stays in cache)
    // Inner: iterate over offsets (reference accessed sequentially)
    let _ = num_slices; // used in general path via weighted_tensor_similarity

    // Fast path: 2D tensor, shift along dim 0, aggregate along dim 1
    if pattern.rank() == 2 && shift_dim == 0 && agg_dim == 1 {
        let p_rows = pattern.shape()[0];
        let cols = pattern.shape()[1];

        for col in 0..cols {
            // Extract pattern column (stays in cache)
            let mut p_col = Vec::with_capacity(p_rows);
            for row in 0..p_rows {
                p_col.push(pattern.data()[row * cols + col]);
            }

            let w = weights.map_or(1.0, |ws| ws[col]);
            total_weight += w;

            // Iterate over all offsets — reference column data accessed sequentially
            let r_cols = reference.shape()[1];
            for offset in 0..num_positions {
                let mut r_col = Vec::with_capacity(p_rows);
                for row in 0..p_rows {
                    r_col.push(reference.data()[(offset + row) * r_cols + col]);
                }
                let sim = slice_similarity(&p_col, &r_col, metric);
                similarities[offset] += w * sim;
            }
        }
    } else {
        // General case (slower but works for any rank)
        for offset in 0..num_positions {
            let sub = reference.subtensor(shift_dim, offset, p_extent)?;
            let sim = weighted_tensor_similarity(pattern, &sub, agg_dim, weights, metric);
            similarities[offset] = sim;
        }
        total_weight = 1.0; // already normalized inside weighted_tensor_similarity
    }

    // Normalize
    if total_weight > 0.0 && total_weight != 1.0 {
        for s in similarities.iter_mut() {
            *s /= total_weight;
        }
    }

    // Sort by similarity descending, return top-k (offset, similarity)
    let mut results: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results.truncate(top_k);
    Ok(results)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2d(data: Vec<f32>, rows: usize, cols: usize) -> TensorData {
        TensorData::new(data, vec![rows, cols]).unwrap()
    }

    #[test]
    fn test_basic_search() {
        let engine = TensorSearchEngine::new();

        // Insert three 2x3 reference tensors
        let r1 = make_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let r2 = make_2d(vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1], 2, 3);
        let r3 = make_2d(vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0], 2, 3);

        engine
            .insert(create_tensor_document("r1".into(), r1, None).unwrap())
            .unwrap();
        engine
            .insert(create_tensor_document("r2".into(), r2, None).unwrap())
            .unwrap();
        engine
            .insert(create_tensor_document("r3".into(), r3, None).unwrap())
            .unwrap();

        // Query with a pattern similar to r1
        let pattern = make_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let results = engine
            .basic_search(&pattern, 1, None, TensorSimilarityMetric::Cosine, 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "r1"); // exact match should be first
    }

    #[test]
    fn test_shifting_search() {
        let engine = TensorSearchEngine::new();

        // Reference: 6x3 tensor (6 rows, 3 cols)
        #[rustfmt::skip]
        let ref_data = vec![
            0.0, 0.0, 0.0,  // row 0 — noise
            0.0, 0.0, 0.0,  // row 1 — noise
            1.0, 2.0, 3.0,  // row 2 — signal start
            4.0, 5.0, 6.0,  // row 3 — signal end
            0.0, 0.0, 0.0,  // row 4 — noise
            0.0, 0.0, 0.0,  // row 5 — noise
        ];
        let reference = make_2d(ref_data, 6, 3);
        engine
            .insert(create_tensor_document("ref".into(), reference, None).unwrap())
            .unwrap();

        // Pattern: 2x3 tensor (the signal we're looking for)
        let pattern = make_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        let results = engine
            .shifting_search(
                &pattern,
                "ref",
                0,    // shift along dim 0 (rows)
                1,    // aggregate along dim 1 (cols)
                None, // equal weights
                TensorSimilarityMetric::DotProduct,
                3,
            )
            .unwrap();

        // The best match should be at offset 2 (where the signal is)
        assert_eq!(results[0].offset, Some(2));
        assert!(results[0].similarity > results[1].similarity);
    }

    #[test]
    fn test_weighted_similarity() {
        let a = make_2d(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = make_2d(vec![1.0, 0.0, 0.0, 1.0], 2, 2);

        // Equal weights
        let sim1 = weighted_tensor_similarity(
            &a,
            &b,
            1, // aggregate along cols
            None,
            TensorSimilarityMetric::DotProduct,
        );

        // Weight first row more
        let sim2 = weighted_tensor_similarity(
            &a,
            &b,
            1,
            Some(&[2.0, 0.5]),
            TensorSimilarityMetric::DotProduct,
        );

        assert!(sim1 > 0.0);
        assert!(sim2 > 0.0);
    }

    #[test]
    fn test_shifting_search_cosine_exact_match() {
        let engine = TensorSearchEngine::new();

        // Reference: rows 0-1 are the signal, rows 2-3 are different
        #[rustfmt::skip]
        let ref_data = vec![
            1.0, 0.0, 0.0,  // row 0
            0.0, 1.0, 0.0,  // row 1
            0.0, 0.0, 1.0,  // row 2
            1.0, 1.0, 1.0,  // row 3
        ];
        let reference = make_2d(ref_data, 4, 3);
        engine
            .insert(create_tensor_document("ref".into(), reference, None).unwrap())
            .unwrap();

        // Pattern matches rows 0-1 exactly
        let pattern = make_2d(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let results = engine
            .shifting_search(
                &pattern,
                "ref",
                0,
                1,
                None,
                TensorSimilarityMetric::Cosine,
                3,
            )
            .unwrap();

        assert_eq!(results[0].offset, Some(0)); // exact match at offset 0
    }

    #[test]
    fn test_crud_operations() {
        let engine = TensorSearchEngine::new();
        let t = make_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let doc = create_tensor_document("t1".into(), t, None).unwrap();

        engine.insert(doc).unwrap();
        assert_eq!(engine.count(), 1);

        let fetched = engine.get("t1").unwrap();
        assert_eq!(fetched.metadata.id, "t1");

        engine.delete("t1").unwrap();
        assert_eq!(engine.count(), 0);
        assert!(engine.get("t1").is_err());
    }
}
