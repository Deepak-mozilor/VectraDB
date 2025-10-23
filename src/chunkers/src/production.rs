use crate::{utils, Chunk, ChunkType, Chunker, ChunkingConfig};
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Production chunker optimized for real-world production environments
/// Includes features like dynamic sizing, quality scoring, and adaptive strategies
pub struct ProductionChunker {
    sentence_regex: Regex,
    paragraph_regex: Regex,
    url_regex: Regex,
    email_regex: Regex,
    phone_regex: Regex,
}

/// Quality metrics for production chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkQuality {
    pub readability_score: f64,
    pub information_density: f64,
    pub semantic_coherence: f64,
    pub structural_integrity: f64,
    pub overall_score: f64,
}

/// Production chunking strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Adaptive chunking based on content analysis
    Adaptive,
    /// Fixed-size chunking with overlap
    FixedSize,
    /// Semantic boundary chunking
    Semantic,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Enhanced configuration for production chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    pub base_config: ChunkingConfig,
    pub strategy: ChunkingStrategy,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub quality_threshold: f64,
    pub enable_quality_scoring: bool,
    pub enable_dynamic_sizing: bool,
    pub preserve_context: bool,
    pub context_window_size: usize,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            base_config: ChunkingConfig::default(),
            strategy: ChunkingStrategy::Adaptive,
            min_chunk_size: 200,
            max_chunk_size: 2000,
            quality_threshold: 0.7,
            enable_quality_scoring: true,
            enable_dynamic_sizing: true,
            preserve_context: true,
            context_window_size: 100,
        }
    }
}

impl Default for ProductionChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProductionChunker {
    pub fn new() -> Self {
        Self {
            sentence_regex: Regex::new(r"[.!?]+\s+").unwrap(),
            paragraph_regex: Regex::new(r"\n\s*\n").unwrap(),
            url_regex: Regex::new(r"https?://[^\s]+").unwrap(),
            email_regex: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
                .unwrap(),
            phone_regex: Regex::new(
                r"(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",
            )
            .unwrap(),
        }
    }

    /// Adaptive chunking that analyzes content and chooses the best strategy
    fn adaptive_chunk(&self, content: &str, config: &ProductionConfig) -> Vec<Chunk> {
        // Analyze content characteristics
        let content_analysis = self.analyze_content(content);

        // Choose strategy based on analysis
        let strategy = match content_analysis.content_type.as_str() {
            "structured" => ChunkingStrategy::Semantic,
            "narrative" => ChunkingStrategy::FixedSize,
            "technical" => ChunkingStrategy::Hybrid,
            _ => ChunkingStrategy::Adaptive,
        };

        // Apply chosen strategy
        match strategy {
            ChunkingStrategy::Semantic => self.semantic_chunk(content, config),
            ChunkingStrategy::FixedSize => self.fixed_size_chunk(content, config),
            ChunkingStrategy::Hybrid => self.hybrid_chunk(content, config),
            ChunkingStrategy::Adaptive => self.adaptive_chunk_recursive(content, config),
        }
    }

    /// Recursive adaptive chunking with quality feedback
    fn adaptive_chunk_recursive(&self, content: &str, config: &ProductionConfig) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut remaining_content = content.to_string();
        let mut start_index = 0;

        while !remaining_content.is_empty() {
            // Determine optimal chunk size for this segment
            let optimal_size = self.determine_optimal_chunk_size(&remaining_content, config);

            // Extract chunk of optimal size
            let (chunk_content, next_start) =
                self.extract_optimal_chunk(&remaining_content, optimal_size, config);

            // Create chunk
            let chunk = Chunk {
                content: chunk_content.clone(),
                start_index,
                end_index: start_index + chunk_content.len(),
                chunk_type: ChunkType::Production,
                metadata: self.create_production_metadata(&chunk_content, start_index, config),
            };

            // Score chunk quality
            let quality = if config.enable_quality_scoring {
                self.score_chunk_quality(&chunk)
            } else {
                ChunkQuality {
                    readability_score: 0.8,
                    information_density: 0.8,
                    semantic_coherence: 0.8,
                    structural_integrity: 0.8,
                    overall_score: 0.8,
                }
            };

            // Adjust chunk if quality is below threshold
            let final_chunk = if quality.overall_score < config.quality_threshold {
                self.improve_chunk_quality(chunk, config, &quality)
            } else {
                chunk
            };

            chunks.push(final_chunk);

            // Update indices
            start_index += next_start;
            remaining_content = remaining_content[next_start..].to_string();
        }

        chunks
    }

    /// Semantic chunking that respects content boundaries
    fn semantic_chunk(&self, content: &str, config: &ProductionConfig) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let paragraphs: Vec<&str> = self.paragraph_regex.split(content).collect();
        let mut current_chunk = String::new();
        let mut start_index = 0;

        for paragraph in paragraphs {
            if current_chunk.is_empty() {
                current_chunk = paragraph.to_string();
            } else if current_chunk.len() + paragraph.len() + 2 <= config.max_chunk_size {
                current_chunk.push_str("\n\n");
                current_chunk.push_str(paragraph);
            } else {
                // Save current chunk
                if !current_chunk.is_empty() {
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_index,
                        end_index: start_index + current_chunk.len(),
                        chunk_type: ChunkType::Production,
                        metadata: self.create_production_metadata(
                            &current_chunk,
                            start_index,
                            config,
                        ),
                    });
                    start_index += current_chunk.len();
                }
                current_chunk = paragraph.to_string();
            }
        }

        // Add remaining chunk
        if !current_chunk.is_empty() {
            let chunk_len = current_chunk.len();
            chunks.push(Chunk {
                content: current_chunk.clone(),
                start_index,
                end_index: start_index + chunk_len,
                chunk_type: ChunkType::Production,
                metadata: self.create_production_metadata(&current_chunk, start_index, config),
            });
        }

        chunks
    }

    /// Fixed-size chunking with intelligent boundaries
    fn fixed_size_chunk(&self, content: &str, config: &ProductionConfig) -> Vec<Chunk> {
        let ranges = utils::split_with_overlap(
            content,
            config.max_chunk_size,
            config.base_config.overlap_size,
        );
        let mut chunks = Vec::new();

        for (start, end) in ranges {
            let _chunk_content = &content[start..end];

            // Find better boundary if possible
            let (adjusted_start, adjusted_end) =
                self.find_better_boundary(content, start, end, config);

            let final_content = &content[adjusted_start..adjusted_end];

            chunks.push(Chunk {
                content: final_content.to_string(),
                start_index: adjusted_start,
                end_index: adjusted_end,
                chunk_type: ChunkType::Production,
                metadata: self.create_production_metadata(final_content, adjusted_start, config),
            });
        }

        chunks
    }

    /// Hybrid chunking combining multiple strategies
    fn hybrid_chunk(&self, content: &str, config: &ProductionConfig) -> Vec<Chunk> {
        // First try semantic chunking
        let semantic_chunks = self.semantic_chunk(content, config);

        // If chunks are too large, apply fixed-size chunking to large chunks
        let mut final_chunks = Vec::new();

        for chunk in semantic_chunks {
            if chunk.content.len() <= config.max_chunk_size {
                final_chunks.push(chunk);
            } else {
                // Split large chunk using fixed-size approach
                let ranges = utils::split_with_overlap(
                    &chunk.content,
                    config.max_chunk_size,
                    config.base_config.overlap_size,
                );
                for (start, end) in ranges {
                    let sub_content = &chunk.content[start..end];
                    final_chunks.push(Chunk {
                        content: sub_content.to_string(),
                        start_index: chunk.start_index + start,
                        end_index: chunk.start_index + end,
                        chunk_type: ChunkType::Production,
                        metadata: self.create_production_metadata(
                            sub_content,
                            chunk.start_index + start,
                            config,
                        ),
                    });
                }
            }
        }

        final_chunks
    }

    /// Analyzes content characteristics
    fn analyze_content(&self, content: &str) -> ContentAnalysis {
        let lines: Vec<&str> = content.lines().collect();
        let total_chars = content.len();
        let total_lines = lines.len();

        // Count different content types
        let sentence_count = self.sentence_regex.find_iter(content).count();
        let paragraph_count = self.paragraph_regex.find_iter(content).count();
        let url_count = self.url_regex.find_iter(content).count();
        let email_count = self.email_regex.find_iter(content).count();
        let phone_count = self.phone_regex.find_iter(content).count();

        // Determine content type
        let content_type = if paragraph_count > total_lines / 3 {
            "structured"
        } else if sentence_count > total_lines / 2 {
            "narrative"
        } else if url_count > 0 || email_count > 0 || phone_count > 0 {
            "technical"
        } else {
            "mixed"
        };

        ContentAnalysis {
            content_type: content_type.to_string(),
            total_chars,
            total_lines,
            sentence_count,
            paragraph_count,
            url_count,
            email_count,
            phone_count,
            avg_sentence_length: if sentence_count > 0 {
                total_chars as f64 / sentence_count as f64
            } else {
                0.0
            },
            avg_paragraph_length: if paragraph_count > 0 {
                total_chars as f64 / paragraph_count as f64
            } else {
                0.0
            },
        }
    }

    /// Determines optimal chunk size based on content analysis
    fn determine_optimal_chunk_size(&self, content: &str, config: &ProductionConfig) -> usize {
        if !config.enable_dynamic_sizing {
            return config.max_chunk_size;
        }

        let analysis = self.analyze_content(content);

        // Adjust chunk size based on content characteristics
        let base_size = config.max_chunk_size;
        let adjustment_factor = match analysis.content_type.as_str() {
            "structured" => 1.2, // Prefer larger chunks for structured content
            "narrative" => 0.8,  // Prefer smaller chunks for narrative content
            "technical" => 1.0,  // Standard size for technical content
            _ => 1.0,
        };

        let optimal_size = (base_size as f64 * adjustment_factor) as usize;

        // Ensure size is within bounds
        std::cmp::max(
            config.min_chunk_size,
            std::cmp::min(optimal_size, config.max_chunk_size),
        )
    }

    /// Extracts optimal chunk respecting semantic boundaries
    fn extract_optimal_chunk(
        &self,
        content: &str,
        target_size: usize,
        _config: &ProductionConfig,
    ) -> (String, usize) {
        let chars: Vec<char> = content.chars().collect();

        if chars.len() <= target_size {
            return (content.to_string(), chars.len());
        }

        // Find semantic boundaries near target size
        let boundaries = utils::find_semantic_boundaries(content);

        // Find the best boundary within acceptable range
        let acceptable_range =
            (target_size as f64 * 0.8) as usize..=(target_size as f64 * 1.2) as usize;

        for boundary in boundaries {
            if acceptable_range.contains(&boundary) {
                return (chars[..boundary].iter().collect(), boundary);
            }
        }

        // Fall back to target size if no good boundary found
        (chars[..target_size].iter().collect(), target_size)
    }

    /// Finds better boundaries for chunking
    fn find_better_boundary(
        &self,
        content: &str,
        start: usize,
        end: usize,
        _config: &ProductionConfig,
    ) -> (usize, usize) {
        let boundaries = utils::find_semantic_boundaries(content);

        // Look for boundaries near the current end
        let target_end = end;
        let search_range = 100; // Look within 100 characters

        for boundary in boundaries {
            if boundary >= target_end.saturating_sub(search_range)
                && boundary <= target_end + search_range
            {
                return (start, boundary);
            }
        }

        (start, end)
    }

    /// Scores chunk quality
    fn score_chunk_quality(&self, chunk: &Chunk) -> ChunkQuality {
        let readability_score = self.calculate_readability_score(&chunk.content);
        let information_density = self.calculate_information_density(&chunk.content);
        let semantic_coherence = self.calculate_semantic_coherence(&chunk.content);
        let structural_integrity = self.calculate_structural_integrity(&chunk.content);

        let overall_score =
            (readability_score + information_density + semantic_coherence + structural_integrity)
                / 4.0;

        ChunkQuality {
            readability_score,
            information_density,
            semantic_coherence,
            structural_integrity,
            overall_score,
        }
    }

    /// Calculates readability score
    fn calculate_readability_score(&self, content: &str) -> f64 {
        let words: Vec<&str> = content.split_whitespace().collect();
        let sentences: Vec<&str> = self.sentence_regex.split(content).collect();

        if words.is_empty() || sentences.is_empty() {
            return 0.0;
        }

        let avg_words_per_sentence = words.len() as f64 / sentences.len() as f64;
        let avg_syllables_per_word = words
            .iter()
            .map(|word| self.count_syllables(word))
            .sum::<usize>() as f64
            / words.len() as f64;

        // Simplified Flesch Reading Ease formula
        let score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word);

        // Normalize to 0-1 range
        (score / 100.0).clamp(0.0, 1.0)
    }

    /// Calculates information density
    fn calculate_information_density(&self, content: &str) -> f64 {
        let total_chars = content.len();
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();

        if words.is_empty() {
            return 0.0;
        }

        let vocabulary_richness = unique_words.len() as f64 / words.len() as f64;
        let content_density = words.len() as f64 / total_chars as f64;

        (vocabulary_richness + content_density) / 2.0
    }

    /// Calculates semantic coherence
    fn calculate_semantic_coherence(&self, content: &str) -> f64 {
        // Simplified semantic coherence based on sentence transitions
        let sentences: Vec<&str> = self.sentence_regex.split(content).collect();

        if sentences.len() < 2 {
            return 1.0;
        }

        // Check for transition words and sentence structure consistency
        let transition_words = [
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
        ];
        let transition_count = sentences
            .windows(2)
            .map(|pair| {
                transition_words
                    .iter()
                    .any(|&word| pair[1].to_lowercase().contains(word)) as usize
            })
            .sum::<usize>();

        transition_count as f64 / (sentences.len() - 1) as f64
    }

    /// Calculates structural integrity
    fn calculate_structural_integrity(&self, content: &str) -> f64 {
        // Check for proper sentence structure and punctuation
        let sentences: Vec<&str> = self.sentence_regex.split(content).collect();

        if sentences.is_empty() {
            return 0.0;
        }

        let proper_sentences = sentences
            .iter()
            .filter(|sentence| {
                let trimmed = sentence.trim();
                !trimmed.is_empty()
                    && (trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?'))
            })
            .count();

        proper_sentences as f64 / sentences.len() as f64
    }

    /// Counts syllables in a word (simplified)
    fn count_syllables(&self, word: &str) -> usize {
        let vowels = "aeiouy";
        let mut count = 0;
        let mut prev_was_vowel = false;

        for c in word.to_lowercase().chars() {
            let is_vowel = vowels.contains(c);
            if is_vowel && !prev_was_vowel {
                count += 1;
            }
            prev_was_vowel = is_vowel;
        }

        // Handle silent 'e'
        if word.to_lowercase().ends_with('e') && count > 1 {
            count -= 1;
        }

        std::cmp::max(1, count)
    }

    /// Improves chunk quality by adjusting boundaries
    fn improve_chunk_quality(
        &self,
        chunk: Chunk,
        _config: &ProductionConfig,
        _quality: &ChunkQuality,
    ) -> Chunk {
        // For now, return the chunk as-is
        // In a real implementation, this would try different boundaries
        chunk
    }

    /// Creates metadata for production chunks
    fn create_production_metadata(
        &self,
        content: &str,
        start_index: usize,
        config: &ProductionConfig,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Basic metrics
        let words: Vec<&str> = content.split_whitespace().collect();
        let sentences: Vec<&str> = self.sentence_regex.split(content).collect();
        let paragraphs: Vec<&str> = self.paragraph_regex.split(content).collect();

        metadata.insert("word_count".to_string(), words.len().to_string());
        metadata.insert("sentence_count".to_string(), sentences.len().to_string());
        metadata.insert("paragraph_count".to_string(), paragraphs.len().to_string());
        metadata.insert("char_count".to_string(), content.len().to_string());
        metadata.insert("start_index".to_string(), start_index.to_string());
        metadata.insert("chunking_method".to_string(), "production".to_string());
        metadata.insert("strategy".to_string(), format!("{:?}", config.strategy));

        // Content analysis
        let analysis = self.analyze_content(content);
        metadata.insert("content_type".to_string(), analysis.content_type);
        metadata.insert(
            "avg_sentence_length".to_string(),
            format!("{:.2}", analysis.avg_sentence_length),
        );
        metadata.insert(
            "avg_paragraph_length".to_string(),
            format!("{:.2}", analysis.avg_paragraph_length),
        );

        // Quality scores
        if config.enable_quality_scoring {
            let quality = self.score_chunk_quality(&Chunk {
                content: content.to_string(),
                start_index,
                end_index: start_index + content.len(),
                chunk_type: ChunkType::Production,
                metadata: HashMap::new(),
            });

            metadata.insert(
                "readability_score".to_string(),
                format!("{:.3}", quality.readability_score),
            );
            metadata.insert(
                "information_density".to_string(),
                format!("{:.3}", quality.information_density),
            );
            metadata.insert(
                "semantic_coherence".to_string(),
                format!("{:.3}", quality.semantic_coherence),
            );
            metadata.insert(
                "structural_integrity".to_string(),
                format!("{:.3}", quality.structural_integrity),
            );
            metadata.insert(
                "overall_quality".to_string(),
                format!("{:.3}", quality.overall_score),
            );
        }

        metadata
    }

    /// Applies overlap to production chunks
    fn apply_overlap(&self, chunks: Vec<Chunk>, config: &ProductionConfig) -> Vec<Chunk> {
        if config.base_config.overlap_size == 0 {
            return chunks;
        }

        let mut overlapped_chunks = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let mut overlapped_chunk = chunk.clone();

            // Add context from previous chunk if enabled
            if config.preserve_context && i > 0 {
                let prev_chunk = &chunks[i - 1];
                let context_size =
                    std::cmp::min(config.context_window_size, config.base_config.overlap_size);

                if context_size > 0 && prev_chunk.content.len() >= context_size {
                    let context = &prev_chunk.content[prev_chunk.content.len() - context_size..];
                    overlapped_chunk.content = format!("{} {}", context, chunk.content);
                }
            }

            overlapped_chunks.push(overlapped_chunk);
        }

        overlapped_chunks
    }
}

/// Content analysis result
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ContentAnalysis {
    content_type: String,
    total_chars: usize,
    total_lines: usize,
    sentence_count: usize,
    paragraph_count: usize,
    url_count: usize,
    email_count: usize,
    phone_count: usize,
    avg_sentence_length: f64,
    avg_paragraph_length: f64,
}

impl Chunker for ProductionChunker {
    fn chunk(&self, text: &str, config: &ChunkingConfig) -> Result<Vec<Chunk>> {
        // Convert to production config
        let production_config = ProductionConfig {
            base_config: config.clone(),
            ..Default::default()
        };

        self.chunk_with_production_config(text, &production_config)
    }

    fn chunk_type(&self) -> ChunkType {
        ChunkType::Production
    }
}

impl ProductionChunker {
    /// Chunks text with production-specific configuration
    pub fn chunk_with_production_config(
        &self,
        text: &str,
        config: &ProductionConfig,
    ) -> Result<Vec<Chunk>> {
        let chunks = match config.strategy {
            ChunkingStrategy::Adaptive => self.adaptive_chunk(text, config),
            ChunkingStrategy::FixedSize => self.fixed_size_chunk(text, config),
            ChunkingStrategy::Semantic => self.semantic_chunk(text, config),
            ChunkingStrategy::Hybrid => self.hybrid_chunk(text, config),
        };

        Ok(self.apply_overlap(chunks, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_chunker_creation() {
        let chunker = ProductionChunker::new();
        assert_eq!(chunker.chunk_type(), ChunkType::Production);
    }

    #[test]
    fn test_content_analysis() {
        let chunker = ProductionChunker::new();
        let content =
            "This is a test sentence. This is another sentence.\n\nThis is a new paragraph.";
        let analysis = chunker.analyze_content(content);

        assert_eq!(analysis.content_type, "narrative"); // 3 sentences > 3 lines / 2
        assert!(analysis.sentence_count >= 2);
        assert!(analysis.paragraph_count >= 1);
    }

    #[test]
    fn test_adaptive_chunking() {
        let chunker = ProductionChunker::new();
        let config = ProductionConfig::default();

        let content = "This is a test document with multiple sentences. It contains various types of content.\n\nThis is a new paragraph with more content.";
        let chunks = chunker
            .chunk_with_production_config(content, &config)
            .unwrap();

        assert!(!chunks.is_empty());
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_quality_scoring() {
        let chunker = ProductionChunker::new();
        let content =
            "This is a well-structured sentence. It has proper punctuation and good readability.";
        let quality = chunker.score_chunk_quality(&Chunk {
            content: content.to_string(),
            start_index: 0,
            end_index: content.len(),
            chunk_type: ChunkType::Production,
            metadata: HashMap::new(),
        });

        assert!(quality.overall_score > 0.0);
        assert!(quality.overall_score <= 1.0);
    }

    #[test]
    fn test_syllable_counting() {
        let chunker = ProductionChunker::new();

        assert_eq!(chunker.count_syllables("test"), 1);
        assert_eq!(chunker.count_syllables("testing"), 2);
        assert_eq!(chunker.count_syllables("beautiful"), 3);
    }
}
