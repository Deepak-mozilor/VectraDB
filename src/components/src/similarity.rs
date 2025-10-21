use crate::{SimilarityResult, VectorDocument, VectraDBError};
use ndarray::{Array1, ArrayView1};

/// Similarity calculation functions for vector comparisons

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Result<f32, VectraDBError> {
    if a.len() != b.len() {
        return Err(VectraDBError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let dot_product = a.dot(b);
    let norm_a = (a.dot(a)).sqrt();
    let norm_b = (b.dot(b)).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (norm_a * norm_b))
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Result<f32, VectraDBError> {
    if a.len() != b.len() {
        return Err(VectraDBError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let diff = a - b;
    Ok(diff.dot(&diff).sqrt())
}

/// Calculate Manhattan distance between two vectors
pub fn manhattan_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Result<f32, VectraDBError> {
    if a.len() != b.len() {
        return Err(VectraDBError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let diff = a - b;
    Ok(diff.mapv(|x| x.abs()).sum())
}

/// Calculate dot product similarity
pub fn dot_product_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Result<f32, VectraDBError> {
    if a.len() != b.len() {
        return Err(VectraDBError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    Ok(a.dot(b))
}

/// Find top-k most similar vectors using cosine similarity
pub fn find_similar_vectors_cosine(
    query_vector: &ArrayView1<f32>,
    documents: &[VectorDocument],
    top_k: usize,
) -> Result<Vec<SimilarityResult>, VectraDBError> {
    let mut results: Vec<SimilarityResult> = documents
        .iter()
        .filter_map(|doc| {
            cosine_similarity(query_vector, &doc.data.view())
                .ok()
                .map(|score| SimilarityResult {
                    id: doc.metadata.id.clone(),
                    score,
                    metadata: doc.metadata.clone(),
                })
        })
        .collect();

    // Sort by similarity score (descending)
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    
    // Take top-k results
    results.truncate(top_k);
    
    Ok(results)
}

/// Find top-k most similar vectors using Euclidean distance
pub fn find_similar_vectors_euclidean(
    query_vector: &ArrayView1<f32>,
    documents: &[VectorDocument],
    top_k: usize,
) -> Result<Vec<SimilarityResult>, VectraDBError> {
    let mut results: Vec<SimilarityResult> = documents
        .iter()
        .filter_map(|doc| {
            euclidean_distance(query_vector, &doc.data.view())
                .ok()
                .map(|distance| {
                    // Convert distance to similarity score (lower distance = higher similarity)
                    let similarity = 1.0 / (1.0 + distance);
                    SimilarityResult {
                        id: doc.metadata.id.clone(),
                        score: similarity,
                        metadata: doc.metadata.clone(),
                    }
                })
        })
        .collect();

    // Sort by similarity score (descending)
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    
    // Take top-k results
    results.truncate(top_k);
    
    Ok(results)
}

/// Calculate similarity between two vector documents
pub fn calculate_document_similarity(
    doc1: &VectorDocument,
    doc2: &VectorDocument,
    similarity_type: SimilarityType,
) -> Result<f32, VectraDBError> {
    match similarity_type {
        SimilarityType::Cosine => cosine_similarity(&doc1.data.view(), &doc2.data.view()),
        SimilarityType::Euclidean => {
            let distance = euclidean_distance(&doc1.data.view(), &doc2.data.view())?;
            Ok(1.0 / (1.0 + distance)) // Convert to similarity
        }
        SimilarityType::Manhattan => {
            let distance = manhattan_distance(&doc1.data.view(), &doc2.data.view())?;
            Ok(1.0 / (1.0 + distance)) // Convert to similarity
        }
        SimilarityType::DotProduct => dot_product_similarity(&doc1.data.view(), &doc2.data.view()),
    }
}

/// Enum for different similarity calculation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityType {
    Cosine,
    Euclidean,
    Manhattan,
    DotProduct,
}

/// Batch similarity calculation for multiple query vectors
pub fn batch_similarity_search(
    query_vectors: &[ArrayView1<f32>],
    documents: &[VectorDocument],
    top_k: usize,
    similarity_type: SimilarityType,
) -> Result<Vec<Vec<SimilarityResult>>, VectraDBError> {
    query_vectors
        .iter()
        .map(|query_vector| {
            match similarity_type {
                SimilarityType::Cosine => find_similar_vectors_cosine(query_vector, documents, top_k),
                SimilarityType::Euclidean => find_similar_vectors_euclidean(query_vector, documents, top_k),
                _ => Err(VectraDBError::InvalidVector), // Not implemented yet
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_operations::create_vector_document;
    use std::collections::HashMap;

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        
        let sim_aa = cosine_similarity(&a.view(), &b.view()).unwrap();
        let sim_ac = cosine_similarity(&a.view(), &c.view()).unwrap();
        
        assert!((sim_aa - 1.0).abs() < 1e-6);
        assert!((sim_ac - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        
        let distance = euclidean_distance(&a.view(), &b.view()).unwrap();
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_similar_vectors() {
        let docs = vec![
            create_vector_document("1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]), None).unwrap(),
            create_vector_document("2".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0]), None).unwrap(),
            create_vector_document("3".to_string(), Array1::from_vec(vec![1.0, 1.0, 0.0]), None).unwrap(),
        ];
        
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let results = find_similar_vectors_cosine(&query.view(), &docs, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1"); // Most similar
    }
}

