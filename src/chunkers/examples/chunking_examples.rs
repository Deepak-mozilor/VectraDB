use vectradb_chunkers::{
    create_chunker, production::ChunkingStrategy, production::ProductionChunker,
    production::ProductionConfig, ChunkType, ChunkingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VectraDB Chunking Methods Demo");
    println!("==============================\n");

    // Example texts for different chunking methods
    let document_text = r#"
    Artificial Intelligence (AI) has become one of the most transformative technologies of our time. 
    It encompasses a wide range of techniques and applications that enable machines to perform tasks 
    that typically require human intelligence.

    Machine learning, a subset of AI, allows computers to learn and improve from experience without 
    being explicitly programmed. Deep learning, which uses neural networks with multiple layers, 
    has been particularly successful in areas such as image recognition, natural language processing, 
    and autonomous vehicles.

    The applications of AI are vast and growing rapidly. In healthcare, AI is being used for 
    medical diagnosis, drug discovery, and personalized treatment plans. In finance, it powers 
    algorithmic trading, fraud detection, and credit scoring systems. In transportation, AI 
    enables autonomous vehicles and optimizes traffic flow.

    However, the rapid advancement of AI also raises important ethical and societal questions. 
    Issues such as bias in algorithms, job displacement, privacy concerns, and the need for 
    responsible AI development are critical topics that require careful consideration and 
    thoughtful policy responses.
    "#;

    let code_text = r#"
    use std::collections::HashMap;

    pub struct Database {
        data: HashMap<String, Vec<u8>>,
        indexes: HashMap<String, Vec<String>>,
    }

    impl Database {
        pub fn new() -> Self {
            Self {
                data: HashMap::new(),
                indexes: HashMap::new(),
            }
        }

        pub fn insert(&mut self, key: String, value: Vec<u8>) -> Result<(), String> {
            if self.data.contains_key(&key) {
                return Err("Key already exists".to_string());
            }
            
            self.data.insert(key.clone(), value);
            self.update_indexes(&key);
            Ok(())
        }

        pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
            self.data.get(key)
        }

        fn update_indexes(&mut self, key: &str) {
            // Update all relevant indexes
            for (index_name, index_keys) in self.indexes.iter_mut() {
                if self.should_index_key(key, index_name) {
                    index_keys.push(key.to_string());
                }
            }
        }

        fn should_index_key(&self, key: &str, index_name: &str) -> bool {
            // Simple indexing logic
            key.contains(index_name)
        }
    }
    "#;

    let markdown_text = r#"
    # VectraDB Documentation

    ## Overview

    VectraDB is a high-performance vector database designed for machine learning applications. 
    It provides efficient storage and retrieval of high-dimensional vectors with support for 
    various similarity metrics.

    ### Key Features

    - **High Performance**: Optimized for fast vector operations
    - **Scalable**: Handles millions of vectors with ease
    - **Flexible**: Supports multiple similarity metrics
    - **Production Ready**: Built for real-world applications

    ## Getting Started

    ### Installation

    ```bash
    cargo add vectradb
    ```

    ### Basic Usage

    ```rust
    use vectradb::Database;

    let mut db = Database::new();
    db.insert("vector1", vec![1.0, 2.0, 3.0])?;
    let result = db.search(vec![1.1, 2.1, 3.1], 5)?;
    ```

    ## API Reference

    ### Database

    The main database structure that handles all vector operations.

    | Method | Description |
    |--------|-------------|
    | `new()` | Creates a new database instance |
    | `insert()` | Adds a vector to the database |
    | `search()` | Finds similar vectors |

    ## Performance Tips

    1. Use appropriate chunk sizes for your data
    2. Enable indexing for better search performance
    3. Consider using production chunking for large datasets

    > **Note**: For production use, consider using the production chunking strategy 
    > for optimal performance and quality.
    "#;

    // Test Document Chunking
    println!("1. Document Chunking");
    println!("===================");
    let document_chunker = create_chunker("document");
    let doc_config = ChunkingConfig {
        max_chunk_size: 300,
        overlap_size: 50,
        preserve_semantics: true,
        include_metadata: true,
        custom_delimiters: None,
    };

    let doc_chunks = document_chunker.chunk(document_text, &doc_config)?;
    println!("Created {} document chunks:", doc_chunks.len());
    for (i, chunk) in doc_chunks.iter().enumerate() {
        println!(
            "  Chunk {}: {} chars, {} words",
            i + 1,
            chunk.content.len(),
            chunk.metadata.get("word_count").unwrap_or(&"0".to_string())
        );
        println!(
            "    Preview: {}...",
            &chunk.content.chars().take(100).collect::<String>()
        );
    }
    println!();

    // Test Code Chunking
    println!("2. Code Chunking");
    println!("================");
    let code_chunker = create_chunker("code");
    let code_config = ChunkingConfig {
        max_chunk_size: 500,
        overlap_size: 30,
        preserve_semantics: true,
        include_metadata: true,
        custom_delimiters: None,
    };

    let code_chunks = code_chunker.chunk(code_text, &code_config)?;
    println!("Created {} code chunks:", code_chunks.len());
    for (i, chunk) in code_chunks.iter().enumerate() {
        println!(
            "  Chunk {}: {} chars, {} lines, language: {}",
            i + 1,
            chunk.content.len(),
            chunk.metadata.get("line_count").unwrap_or(&"0".to_string()),
            chunk
                .metadata
                .get("language")
                .unwrap_or(&"unknown".to_string())
        );
        println!(
            "    Block type: {}",
            chunk
                .metadata
                .get("block_type")
                .unwrap_or(&"mixed".to_string())
        );
    }
    println!();

    // Test Markdown Chunking
    println!("3. Markdown Chunking");
    println!("====================");
    let markdown_chunker = create_chunker("markdown");
    let md_config = ChunkingConfig {
        max_chunk_size: 400,
        overlap_size: 40,
        preserve_semantics: true,
        include_metadata: true,
        custom_delimiters: None,
    };

    let md_chunks = markdown_chunker.chunk(markdown_text, &md_config)?;
    println!("Created {} markdown chunks:", md_chunks.len());
    for (i, chunk) in md_chunks.iter().enumerate() {
        println!(
            "  Chunk {}: {} chars, {} headings, {} code blocks",
            i + 1,
            chunk.content.len(),
            chunk
                .metadata
                .get("heading_count")
                .unwrap_or(&"0".to_string()),
            chunk
                .metadata
                .get("code_block_count")
                .unwrap_or(&"0".to_string())
        );
        if let Some(heading) = chunk.metadata.get("heading") {
            println!("    Section: {}", heading);
        }
    }
    println!();

    // Test Production Chunking
    println!("4. Production Chunking");
    println!("======================");
    let production_chunker = ProductionChunker::new();
    let prod_config = ProductionConfig {
        base_config: ChunkingConfig {
            max_chunk_size: 600,
            overlap_size: 60,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        },
        strategy: ChunkingStrategy::Adaptive,
        min_chunk_size: 200,
        max_chunk_size: 800,
        quality_threshold: 0.7,
        enable_quality_scoring: true,
        enable_dynamic_sizing: true,
        preserve_context: true,
        context_window_size: 50,
    };

    let prod_chunks =
        production_chunker.chunk_with_production_config(document_text, &prod_config)?;
    println!("Created {} production chunks:", prod_chunks.len());
    for (i, chunk) in prod_chunks.iter().enumerate() {
        println!(
            "  Chunk {}: {} chars, quality: {}",
            i + 1,
            chunk.content.len(),
            chunk
                .metadata
                .get("overall_quality")
                .unwrap_or(&"N/A".to_string())
        );
        println!(
            "    Content type: {}, strategy: {}",
            chunk
                .metadata
                .get("content_type")
                .unwrap_or(&"unknown".to_string()),
            chunk
                .metadata
                .get("strategy")
                .unwrap_or(&"unknown".to_string())
        );
    }
    println!();

    // Test different production strategies
    println!("5. Production Strategies Comparison");
    println!("===================================");

    let strategies = vec![
        ChunkingStrategy::Adaptive,
        ChunkingStrategy::FixedSize,
        ChunkingStrategy::Semantic,
        ChunkingStrategy::Hybrid,
    ];

    for strategy in strategies {
        let mut test_config = prod_config.clone();
        test_config.strategy = strategy.clone();

        let chunks =
            production_chunker.chunk_with_production_config(document_text, &test_config)?;
        println!(
            "  {:?}: {} chunks, avg quality: {:.3}",
            strategy,
            chunks.len(),
            chunks
                .iter()
                .filter_map(|c| c.metadata.get("overall_quality"))
                .filter_map(|q| q.parse::<f64>().ok())
                .sum::<f64>()
                / chunks.len() as f64
        );
    }
    println!();

    // Performance comparison
    println!("6. Performance Metrics");
    println!("======================");

    let start = std::time::Instant::now();
    let _ = document_chunker.chunk(document_text, &doc_config)?;
    let doc_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = code_chunker.chunk(code_text, &code_config)?;
    let code_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = markdown_chunker.chunk(markdown_text, &md_config)?;
    let md_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = production_chunker.chunk_with_production_config(document_text, &prod_config)?;
    let prod_time = start.elapsed();

    println!("  Document chunking: {:?}", doc_time);
    println!("  Code chunking: {:?}", code_time);
    println!("  Markdown chunking: {:?}", md_time);
    println!("  Production chunking: {:?}", prod_time);

    Ok(())
}
