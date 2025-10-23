use vectradb_components::{VectorDatabase, InMemoryVectorDB};
use vectradb_search::{SearchAlgorithm, HNSWIndex, AdvancedSearch};
use ndarray::Array1;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VectraDB Basic Usage Example");
    println!("============================");

    // Example 1: Using InMemoryVectorDB
    println!("\n1. In-Memory Vector Database:");
    let mut db = InMemoryVectorDB::new();

    // Create some sample vectors
    let vectors = vec![
        ("doc1", vec![1.0, 0.0, 0.0]),
        ("doc2", vec![0.0, 1.0, 0.0]),
        ("doc3", vec![1.0, 1.0, 0.0]),
        ("doc4", vec![0.0, 0.0, 1.0]),
        ("doc5", vec![1.0, 0.0, 1.0]),
    ];

    // Insert vectors
    for (id, vector_data) in &vectors {
        let vector = Array1::from_vec(vector_data.clone());
        db.create_vector(id.to_string(), vector, None)?;
        println!("  Created vector: {}", id);
    }

    // Search for similar vectors
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let results = db.search_similar(query, 3)?;
    
    println!("  Search results for [1.0, 0.0, 0.0]:");
    for result in results {
        println!("    ID: {}, Score: {:.4}", result.id, result.score);
    }

    // Example 2: Using HNSW Index directly
    println!("\n2. HNSW Index:");
    let mut hnsw_index = HNSWIndex::new(3, 16, 200);

    // Create vector documents for HNSW
    let documents: Vec<vectradb_components::VectorDocument> = vectors
        .iter()
        .map(|(id, vector_data)| {
            vectradb_components::vector_operations::create_vector_document(
                id.to_string(),
                Array1::from_vec(vector_data.clone()),
                None,
            ).unwrap()
        })
        .collect();

    // Build index
    hnsw_index.build_index(documents)?;
    println!("  Built HNSW index with {} vectors", hnsw_index.get_stats().total_vectors);

    // Search using HNSW
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let results = hnsw_index.search(&query, 3)?;
    
    println!("  HNSW search results for [1.0, 0.0, 0.0]:");
    for result in results {
        println!("    ID: {}, Distance: {:.4}, Similarity: {:.4}", 
                result.id, result.distance, result.similarity);
    }

    // Example 3: Vector operations
    println!("\n3. Vector Operations:");
    
    // Normalize a vector
    let mut vector = Array1::from_vec(vec![3.0, 4.0, 0.0]);
    let normalized = vectradb_components::vector_operations::normalize_vector(vector)?;
    println!("  Normalized [3.0, 4.0, 0.0]: {:?}", normalized.to_vec());

    // Calculate similarity
    let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let cosine_sim = vectradb_components::similarity::cosine_similarity(&v1.view(), &v2.view())?;
    println!("  Cosine similarity between [1,0,0] and [0,1,0]: {:.4}", cosine_sim);

    let euclidean_dist = vectradb_components::similarity::euclidean_distance(&v1.view(), &v2.view())?;
    println!("  Euclidean distance between [1,0,0] and [0,1,0]: {:.4}", euclidean_dist);

    // Example 4: Database statistics
    println!("\n4. Database Statistics:");
    let stats = db.get_stats()?;
    println!("  Total vectors: {}", stats.total_vectors);
    println!("  Dimension: {}", stats.dimension);
    println!("  Memory usage: {} bytes", stats.memory_usage);

    println!("\nExample completed successfully!");
    Ok(())
}





