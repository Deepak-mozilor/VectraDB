use crate::{utils, Chunk, ChunkType, Chunker, ChunkingConfig};
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;

/// Document chunker for general text documents
pub struct DocumentChunker {
    sentence_regex: Regex,
    paragraph_regex: Regex,
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentChunker {
    pub fn new() -> Self {
        Self {
            sentence_regex: Regex::new(r"[.!?]+\s+").unwrap(),
            paragraph_regex: Regex::new(r"\n\s*\n").unwrap(),
        }
    }

    /// Chunks text by paragraphs first, then sentences if needed
    fn chunk_by_paragraphs(&self, text: &str, config: &ChunkingConfig) -> Vec<Chunk> {
        let paragraphs: Vec<&str> = self.paragraph_regex.split(text).collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut start_index = 0;

        for paragraph in paragraphs {
            if current_chunk.is_empty() {
                current_chunk = paragraph.to_string();
            } else if current_chunk.len() + paragraph.len() + 2 <= config.max_chunk_size {
                current_chunk.push_str("\n\n");
                current_chunk.push_str(paragraph);
            } else {
                // Save current chunk and start new one
                if !current_chunk.is_empty() {
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_index,
                        end_index: start_index + current_chunk.len(),
                        chunk_type: ChunkType::Document,
                        metadata: self.create_metadata(&current_chunk, start_index),
                    });
                    start_index += current_chunk.len();
                }
                current_chunk = paragraph.to_string();
            }
        }

        // Add the last chunk
        if !current_chunk.is_empty() {
            chunks.push(Chunk {
                content: current_chunk.clone(),
                start_index,
                end_index: start_index + current_chunk.len(),
                chunk_type: ChunkType::Document,
                metadata: self.create_metadata(&current_chunk, start_index),
            });
        }

        chunks
    }

    /// Chunks text by sentences
    fn chunk_by_sentences(&self, text: &str, config: &ChunkingConfig) -> Vec<Chunk> {
        let sentences: Vec<&str> = self.sentence_regex.split(text).collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut start_index = 0;

        for sentence in sentences {
            if current_chunk.is_empty() {
                current_chunk = sentence.to_string();
            } else if current_chunk.len() + sentence.len() < config.max_chunk_size {
                current_chunk.push_str(". ");
                current_chunk.push_str(sentence);
            } else {
                // Save current chunk and start new one
                if !current_chunk.is_empty() {
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_index,
                        end_index: start_index + current_chunk.len(),
                        chunk_type: ChunkType::Document,
                        metadata: self.create_metadata(&current_chunk, start_index),
                    });
                    start_index += current_chunk.len();
                }
                current_chunk = sentence.to_string();
            }
        }

        // Add the last chunk
        if !current_chunk.is_empty() {
            chunks.push(Chunk {
                content: current_chunk.clone(),
                start_index,
                end_index: start_index + current_chunk.len(),
                chunk_type: ChunkType::Document,
                metadata: self.create_metadata(&current_chunk, start_index),
            });
        }

        chunks
    }

    /// Creates metadata for a document chunk
    fn create_metadata(&self, content: &str, start_index: usize) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Word count
        let word_count = content.split_whitespace().count();
        metadata.insert("word_count".to_string(), word_count.to_string());

        // Sentence count
        let sentence_count = self.sentence_regex.find_iter(content).count() + 1;
        metadata.insert("sentence_count".to_string(), sentence_count.to_string());

        // Paragraph count
        let paragraph_count = self.paragraph_regex.find_iter(content).count() + 1;
        metadata.insert("paragraph_count".to_string(), paragraph_count.to_string());

        // Reading level estimation (simple)
        let avg_word_length =
            content.split_whitespace().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64;
        metadata.insert(
            "avg_word_length".to_string(),
            format!("{:.2}", avg_word_length),
        );

        metadata.insert("chunking_method".to_string(), "document".to_string());
        metadata.insert("start_index".to_string(), start_index.to_string());

        metadata
    }

    /// Applies overlap to chunks if configured
    fn apply_overlap(&self, chunks: Vec<Chunk>, config: &ChunkingConfig) -> Vec<Chunk> {
        if config.overlap_size == 0 {
            return chunks;
        }

        let mut overlapped_chunks = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let mut overlapped_chunk = chunk.clone();

            // Add overlap from previous chunk
            if i > 0 {
                let prev_chunk = &chunks[i - 1];
                let overlap_start = std::cmp::max(
                    prev_chunk.end_index.saturating_sub(config.overlap_size),
                    prev_chunk.start_index,
                );
                let overlap_text = &prev_chunk.content[overlap_start - prev_chunk.start_index..];
                overlapped_chunk.content = format!("{} {}", overlap_text, chunk.content);
                overlapped_chunk.start_index = overlap_start;
            }

            // Add overlap to next chunk
            if i < chunks.len() - 1 {
                let next_chunk = &chunks[i + 1];
                let overlap_end =
                    std::cmp::min(chunk.end_index + config.overlap_size, next_chunk.end_index);
                let overlap_text = &next_chunk.content[..overlap_end - chunk.end_index];
                overlapped_chunk.content = format!("{} {}", chunk.content, overlap_text);
                overlapped_chunk.end_index = overlap_end;
            }

            overlapped_chunks.push(overlapped_chunk);
        }

        overlapped_chunks
    }
}

impl Chunker for DocumentChunker {
    fn chunk(&self, text: &str, config: &ChunkingConfig) -> Result<Vec<Chunk>> {
        let chunks = if config.preserve_semantics {
            // Try paragraph-based chunking first
            let paragraph_chunks = self.chunk_by_paragraphs(text, config);

            // If any chunk is too large, fall back to sentence-based chunking
            if paragraph_chunks
                .iter()
                .any(|c| c.content.len() > config.max_chunk_size)
            {
                self.chunk_by_sentences(text, config)
            } else {
                paragraph_chunks
            }
        } else {
            // Simple character-based chunking
            let ranges =
                utils::split_with_overlap(text, config.max_chunk_size, config.overlap_size);
            ranges
                .into_iter()
                .map(|(start, end)| Chunk {
                    content: text[start..end].to_string(),
                    start_index: start,
                    end_index: end,
                    chunk_type: ChunkType::Document,
                    metadata: self.create_metadata(&text[start..end], start),
                })
                .collect()
        };

        Ok(self.apply_overlap(chunks, config))
    }

    fn chunk_type(&self) -> ChunkType {
        ChunkType::Document
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_chunker_creation() {
        let chunker = DocumentChunker::new();
        assert_eq!(chunker.chunk_type(), ChunkType::Document);
    }

    #[test]
    fn test_chunk_by_paragraphs() {
        let chunker = DocumentChunker::new();
        let config = ChunkingConfig {
            max_chunk_size: 100,
            overlap_size: 10,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        };

        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunker.chunk(text, &config).unwrap();

        assert!(!chunks.is_empty());
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_by_sentences() {
        let chunker = DocumentChunker::new();
        let config = ChunkingConfig {
            max_chunk_size: 50,
            overlap_size: 5,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        };

        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunker.chunk(text, &config).unwrap();

        assert!(!chunks.is_empty());
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_metadata_creation() {
        let chunker = DocumentChunker::new();
        let content = "This is a test sentence.";
        let metadata = chunker.create_metadata(content, 0);

        assert!(metadata.contains_key("word_count"));
        assert!(metadata.contains_key("sentence_count"));
        assert_eq!(metadata.get("word_count").unwrap(), "6");
    }
}
