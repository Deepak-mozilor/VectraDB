//! Multi-dimensional tensor types and similarity functions.
//!
//! Extends VectraDB beyond 1D vectors to support 2D matrices, 3D volumes,
//! and arbitrary-rank tensors. Based on the TensorSearch paper (IEEE Big Data 2024).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::VectraDBError;

// ============================================================
// Tensor data
// ============================================================

/// A multi-dimensional tensor stored as flat contiguous data in row-major order.
///
/// # Examples
/// ```ignore
/// // A 2×3 matrix:
/// //   [[1, 2, 3],
/// //    [4, 5, 6]]
/// let t = TensorData::new(vec![1.0,2.0,3.0,4.0,5.0,6.0], vec![2, 3]).unwrap();
/// assert_eq!(t.rank(), 2);
/// assert_eq!(t.get(&[0, 2]), 3.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorData {
    /// Create a new tensor. `data` length must equal the product of `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, VectraDBError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(VectraDBError::InvalidVector);
        }
        if shape.is_empty() {
            return Err(VectraDBError::InvalidVector);
        }
        let strides = Self::compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// Get element at the given multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> f32 {
        let flat = self.flat_index(index);
        self.data[flat]
    }

    fn flat_index(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }

    /// Extract a contiguous sub-tensor along dimension `dim` from `start` to `start+len`.
    ///
    /// For a 2D tensor of shape [R, C], `subtensor(0, 2, 3)` returns rows 2..5.
    pub fn subtensor(&self, dim: usize, start: usize, len: usize) -> Result<Self, VectraDBError> {
        if dim >= self.rank() || start + len > self.shape[dim] {
            return Err(VectraDBError::InvalidVector);
        }
        // For dim 0 (first dimension), data is contiguous
        if dim == 0 {
            let stride = self.strides[0];
            let begin = start * stride;
            let end = (start + len) * stride;
            let mut new_shape = self.shape.clone();
            new_shape[0] = len;
            return Self::new(self.data[begin..end].to_vec(), new_shape);
        }
        // General case: must copy non-contiguous data
        let mut new_shape = self.shape.clone();
        new_shape[dim] = len;
        let new_count: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_count);
        let new_strides = Self::compute_strides(&new_shape);

        for flat_new in 0..new_count {
            // Convert flat_new to multi-dim index in new shape
            let mut idx = vec![0usize; self.rank()];
            let mut rem = flat_new;
            for d in 0..self.rank() {
                idx[d] = rem / new_strides[d];
                rem %= new_strides[d];
            }
            // Adjust the sliced dimension
            idx[dim] += start;
            new_data.push(self.data[self.flat_index(&idx)]);
        }
        Self::new(new_data, new_shape)
    }

    /// Extract a 1D slice: fix all dimensions except one, vary `dim`.
    /// Returns a Vec<f32> of length `shape[dim]`.
    ///
    /// `fixed` has one entry per dimension. `fixed[dim]` is ignored.
    pub fn fiber(&self, dim: usize, fixed: &[usize]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.shape[dim]);
        let mut idx = fixed.to_vec();
        for i in 0..self.shape[dim] {
            idx[dim] = i;
            result.push(self.data[self.flat_index(&idx)]);
        }
        result
    }

    /// Get a contiguous row from a 2D tensor (fast path).
    /// Equivalent to `data[row * cols .. (row+1) * cols]`.
    pub fn row_2d(&self, row: usize) -> &[f32] {
        let cols = self.shape[1];
        let start = row * cols;
        &self.data[start..start + cols]
    }
}

// ============================================================
// Tensor metadata and document
// ============================================================

/// Metadata associated with a stored tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub id: String,
    pub shape: Vec<usize>,
    pub rank: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub tags: HashMap<String, String>,
}

/// A tensor with its metadata, stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDocument {
    pub metadata: TensorMetadata,
    pub tensor: TensorData,
}

/// Result of a tensor similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSearchResult {
    pub id: String,
    pub similarity: f32,
    /// For shifting search: the offset along the shift dimension where the match was found.
    pub offset: Option<usize>,
    pub metadata: TensorMetadata,
}

/// Create a new TensorDocument with timestamps.
pub fn create_tensor_document(
    id: String,
    tensor: TensorData,
    tags: Option<HashMap<String, String>>,
) -> Result<TensorDocument, VectraDBError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let metadata = TensorMetadata {
        id,
        shape: tensor.shape().to_vec(),
        rank: tensor.rank(),
        created_at: now,
        updated_at: now,
        tags: tags.unwrap_or_default(),
    };
    Ok(TensorDocument { metadata, tensor })
}

// ============================================================
// Similarity metrics
// ============================================================

/// Similarity metric for tensor comparison.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TensorSimilarityMetric {
    /// Sum of element-wise products.
    DotProduct,
    /// Normalized dot product (cosine similarity).
    Cosine,
    /// Pearson cross-correlation.
    CrossCorrelation,
}

/// Dot product between two equal-length slices.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity between two slices.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = dot_product(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Pearson cross-correlation between two slices.
pub fn cross_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    cov / denom
}

/// Compute similarity between two slices using the given metric.
pub fn slice_similarity(a: &[f32], b: &[f32], metric: TensorSimilarityMetric) -> f32 {
    match metric {
        TensorSimilarityMetric::DotProduct => dot_product(a, b),
        TensorSimilarityMetric::Cosine => cosine_sim(a, b),
        TensorSimilarityMetric::CrossCorrelation => cross_correlation(a, b),
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.rank(), 2);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.num_elements(), 6);
    }

    #[test]
    fn test_tensor_indexing() {
        let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_tensor_subtensor_dim0() {
        let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let sub = t.subtensor(0, 1, 2).unwrap(); // rows 1..3
        assert_eq!(sub.shape(), &[2, 2]);
        assert_eq!(sub.data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_row_2d() {
        let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.row_2d(0), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row_2d(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dot_product() {
        assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim() {
        let s = cosine_sim(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s - 1.0).abs() < 1e-6);
        let s = cosine_sim(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_cross_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cc = cross_correlation(&a, &a);
        assert!((cc - 1.0).abs() < 1e-6); // self-correlation = 1
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        assert!(TensorData::new(vec![1.0, 2.0], vec![3]).is_err());
    }
}
