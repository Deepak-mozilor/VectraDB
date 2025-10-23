use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod code;
pub mod document;
pub mod markdown;
pub mod production;

pub use code::CodeChunker;
pub use document::DocumentChunker;
pub use markdown::MarkdownChunker;
pub use production::ProductionChunker;

/// Represents a chunk of text with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub content: String,
    pub start_index: usize,
    pub end_index: usize,
    pub chunk_type: ChunkType,
    pub metadata: HashMap<String, String>,
}

/// Different types of chunks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChunkType {
    Document,
    Code,
    Markdown,
    Production,
}

/// Configuration for chunking operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub max_chunk_size: usize,
    pub overlap_size: usize,
    pub preserve_semantics: bool,
    pub include_metadata: bool,
    pub custom_delimiters: Option<Vec<String>>,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 1000,
            overlap_size: 100,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        }
    }
}

/// Trait for all chunking implementations
pub trait Chunker {
    fn chunk(&self, text: &str, config: &ChunkingConfig) -> Result<Vec<Chunk>>;
    fn chunk_type(&self) -> ChunkType;
}

/// Factory function to create chunkers based on content type
pub fn create_chunker(content_type: &str) -> Box<dyn Chunker> {
    match content_type.to_lowercase().as_str() {
        "document" | "text" => Box::new(DocumentChunker::new()),
        "code" | "source" => Box::new(CodeChunker::new()),
        "markdown" | "md" => Box::new(MarkdownChunker::new()),
        "production" => Box::new(ProductionChunker::new()),
        _ => Box::new(DocumentChunker::new()), // Default to document chunking
    }
}

/// Utility functions for text processing
pub mod utils {
    use super::*;

    /// Splits text into chunks with overlap
    pub fn split_with_overlap(text: &str, max_size: usize, overlap: usize) -> Vec<(usize, usize)> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        if chars.len() <= max_size {
            return vec![(0, chars.len())];
        }

        let mut start = 0;
        while start < chars.len() {
            let end = std::cmp::min(start + max_size, chars.len());
            chunks.push((start, end));

            if end >= chars.len() {
                break;
            }

            start = end - overlap;
        }

        chunks
    }

    /// Finds semantic boundaries in text
    pub fn find_semantic_boundaries(text: &str) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        for (i, char) in chars.iter().enumerate() {
            match char {
                '.' | '!' | '?' => {
                    // Sentence endings
                    boundaries.push(i + 1);
                }
                '\n' => {
                    // Line breaks
                    boundaries.push(i + 1);
                }
                ';' => {
                    // Semicolons
                    boundaries.push(i + 1);
                }
                _ => {}
            }
        }

        boundaries.push(chars.len());
        boundaries.sort();
        boundaries.dedup();
        boundaries
    }

    /// Creates metadata for a chunk
    pub fn create_chunk_metadata(
        chunk: &Chunk,
        source_info: Option<&str>,
    ) -> HashMap<String, String> {
        let mut metadata = chunk.metadata.clone();

        if let Some(source) = source_info {
            metadata.insert("source".to_string(), source.to_string());
        }

        metadata.insert("length".to_string(), chunk.content.len().to_string());
        metadata.insert("start_index".to_string(), chunk.start_index.to_string());
        metadata.insert("end_index".to_string(), chunk.end_index.to_string());

        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_config_default() {
        let config = ChunkingConfig::default();
        assert_eq!(config.max_chunk_size, 1000);
        assert_eq!(config.overlap_size, 100);
        assert!(config.preserve_semantics);
        assert!(config.include_metadata);
    }

    #[test]
    fn test_chunker_factory() {
        let document_chunker = create_chunker("document");
        assert_eq!(document_chunker.chunk_type(), ChunkType::Document);

        let code_chunker = create_chunker("code");
        assert_eq!(code_chunker.chunk_type(), ChunkType::Code);
    }

    #[test]
    fn test_split_with_overlap() {
        let text = "This is a test text that is longer than the max size.";
        let chunks = utils::split_with_overlap(text, 20, 5);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0], (0, 20));
    }

    #[test]
    fn test_find_semantic_boundaries() {
        let text = "First sentence. Second sentence! Third sentence?";
        let boundaries = utils::find_semantic_boundaries(text);

        assert!(!boundaries.is_empty());
        assert!(boundaries.contains(&15)); // End of first sentence
    }
}
