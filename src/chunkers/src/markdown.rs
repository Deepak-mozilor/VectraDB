use crate::{Chunk, ChunkType, Chunker, ChunkingConfig, utils};
use anyhow::Result;
use pulldown_cmark::{Parser, Event, Tag, TagEnd};
use regex::Regex;
use std::collections::HashMap;

/// Markdown chunker for markdown documents
pub struct MarkdownChunker {
    heading_regex: Regex,
    code_block_regex: Regex,
    list_regex: Regex,
    link_regex: Regex,
    image_regex: Regex,
}

impl MarkdownChunker {
    pub fn new() -> Self {
        Self {
            heading_regex: Regex::new(r"^#{1,6}\s+").unwrap(),
            code_block_regex: Regex::new(r"^```[\s\S]*?^```").unwrap(),
            list_regex: Regex::new(r"^[\s]*[-*+]\s+").unwrap(),
            link_regex: Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap(),
            image_regex: Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)").unwrap(),
        }
    }

    /// Chunks markdown by headings (hierarchical structure)
    fn chunk_by_headings(&self, content: &str, config: &ChunkingConfig) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_section = Vec::<String>::new();
        let mut current_heading = None;
        let mut current_level = 0;
        let mut start_line = 0;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Check if this is a heading
            if self.heading_regex.is_match(trimmed) {
                // Save previous section if it exists
                if !current_section.is_empty() {
                    let section_content = current_section.join("\n");
                    chunks.push(Chunk {
                        content: section_content.clone(),
                        start_index: start_line,
                        end_index: i,
                        chunk_type: ChunkType::Markdown,
                        metadata: self.create_markdown_metadata(&section_content, start_line, "section", current_heading.as_deref()),
                    });
                    current_section.clear();
                }
                
                // Start new section
                current_heading = Some(trimmed.to_string());
                current_level = trimmed.matches('#').count();
                start_line = i;
                current_section.push(line.to_string());
            } else {
                current_section.push(line.to_string());
                
                // Check if section is getting too large
                if current_section.join("\n").len() > config.max_chunk_size {
                    let section_content = current_section.join("\n");
                    chunks.push(Chunk {
                        content: section_content.clone(),
                        start_index: start_line,
                        end_index: i,
                        chunk_type: ChunkType::Markdown,
                        metadata: self.create_markdown_metadata(&section_content, start_line, "section", current_heading.as_deref()),
                    });
                    current_section.clear();
                    start_line = i;
                }
            }
        }

        // Add remaining section
        if !current_section.is_empty() {
            let section_content = current_section.join("\n");
            chunks.push(Chunk {
                content: section_content.clone(),
                start_index: start_line,
                end_index: lines.len(),
                chunk_type: ChunkType::Markdown,
                metadata: self.create_markdown_metadata(&section_content, start_line, "section", current_heading.as_deref()),
            });
        }

        chunks
    }

    /// Chunks markdown by semantic blocks (headings, paragraphs, code blocks, lists)
    fn chunk_by_semantic_blocks(&self, content: &str, config: &ChunkingConfig) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_block = Vec::<String>::new();
        let mut current_block_type = "text";
        let mut start_line = 0;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Determine block type
            let block_type = if self.heading_regex.is_match(trimmed) {
                "heading"
            } else if trimmed.starts_with("```") {
                "code_block"
            } else if self.list_regex.is_match(line) {
                "list"
            } else if trimmed.starts_with("> ") {
                "quote"
            } else if trimmed.starts_with("|") && trimmed.contains("|") {
                "table"
            } else if trimmed.is_empty() {
                "empty"
            } else {
                "paragraph"
            };

            // If block type changed, save previous block
            if block_type != current_block_type && !current_block.is_empty() && current_block_type != "empty" {
                let block_content = current_block.join("\n");
                chunks.push(Chunk {
                    content: block_content.clone(),
                    start_index: start_line,
                    end_index: i,
                    chunk_type: ChunkType::Markdown,
                    metadata: self.create_markdown_metadata(&block_content, start_line, current_block_type, None),
                });
                current_block.clear();
                start_line = i;
            }

            current_block.push(line.to_string());
            current_block_type = block_type;

            // Check if block is getting too large
            if current_block.join("\n").len() > config.max_chunk_size {
                let block_content = current_block.join("\n");
                chunks.push(Chunk {
                    content: block_content.clone(),
                    start_index: start_line,
                    end_index: i,
                    chunk_type: ChunkType::Markdown,
                    metadata: self.create_markdown_metadata(&block_content, start_line, current_block_type, None),
                });
                current_block.clear();
                start_line = i;
            }
        }

        // Add remaining block
        if !current_block.is_empty() && current_block_type != "empty" {
            let block_content = current_block.join("\n");
            chunks.push(Chunk {
                content: block_content.clone(),
                start_index: start_line,
                end_index: lines.len(),
                chunk_type: ChunkType::Markdown,
                metadata: self.create_markdown_metadata(&block_content, start_line, current_block_type, None),
            });
        }

        chunks
    }

    /// Chunks markdown using pulldown-cmark parser for more accurate parsing
    fn chunk_by_ast(&self, content: &str, config: &ChunkingConfig) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let parser = Parser::new(content);
        let mut current_chunk = String::new();
        let mut current_heading = None;
        let mut chunk_start = 0;
        let mut in_code_block = false;
        let mut code_language = None;

        for event in parser {
            match event {
                Event::Start(Tag::Heading { level, id: _, classes: _, attrs: _ }) => {
                    // Save previous chunk if it exists
                    if !current_chunk.is_empty() {
                        chunks.push(Chunk {
                            content: current_chunk.clone(),
                            start_index: chunk_start,
                            end_index: chunk_start + current_chunk.len(),
                            chunk_type: ChunkType::Markdown,
                            metadata: self.create_markdown_metadata(&current_chunk, chunk_start, "section", current_heading.as_deref()),
                        });
                        current_chunk.clear();
                    }
                    chunk_start += current_chunk.len();
                }
                Event::Text(text) => {
                    current_chunk.push_str(&text);
                }
                Event::Code(code) => {
                    current_chunk.push_str(&format!("`{}`", code));
                }
                Event::Start(Tag::CodeBlock(lang)) => {
                    in_code_block = true;
                    let lang_str = match &lang {
                        pulldown_cmark::CodeBlockKind::Fenced(lang_name) => lang_name.as_ref().to_string(),
                        pulldown_cmark::CodeBlockKind::Indented => String::new(),
                    };
                    code_language = Some(lang_str.clone());
                    current_chunk.push_str("```");
                    if !lang_str.is_empty() {
                        current_chunk.push_str(&lang_str);
                    }
                    current_chunk.push('\n');
                }
                Event::End(TagEnd::CodeBlock) => {
                    in_code_block = false;
                    current_chunk.push_str("```");
                    code_language = None;
                }
                Event::Start(Tag::Paragraph) => {
                    if !in_code_block {
                        current_chunk.push('\n');
                    }
                }
                Event::End(TagEnd::Paragraph) => {
                    if !in_code_block {
                        current_chunk.push('\n');
                    }
                }
                Event::Start(Tag::Heading { level: _, id: _, classes: _, attrs: _ }) => {
                    if let Some(heading) = self.extract_heading_text(&current_chunk) {
                        current_heading = Some(heading);
                    }
                }
                Event::End(TagEnd::Heading(_)) => {
                    // End of heading, could save chunk here if needed
                }
                _ => {}
            }

            // Check if chunk is getting too large
            if current_chunk.len() > config.max_chunk_size {
                chunks.push(Chunk {
                    content: current_chunk.clone(),
                    start_index: chunk_start,
                    end_index: chunk_start + current_chunk.len(),
                    chunk_type: ChunkType::Markdown,
                    metadata: self.create_markdown_metadata(&current_chunk, chunk_start, "section", current_heading.as_deref()),
                });
                chunk_start += current_chunk.len();
                current_chunk.clear();
            }
        }

        // Add remaining content
        if !current_chunk.is_empty() {
            chunks.push(Chunk {
                content: current_chunk.clone(),
                start_index: chunk_start,
                end_index: chunk_start + current_chunk.len(),
                chunk_type: ChunkType::Markdown,
                metadata: self.create_markdown_metadata(&current_chunk, chunk_start, "section", current_heading.as_deref()),
            });
        }

        chunks
    }

    /// Extracts heading text from markdown content
    fn extract_heading_text(&self, content: &str) -> Option<String> {
        for line in content.lines() {
            if self.heading_regex.is_match(line.trim()) {
                return Some(line.trim().trim_start_matches('#').trim().to_string());
            }
        }
        None
    }

    /// Creates metadata for a markdown chunk
    fn create_markdown_metadata(&self, content: &str, start_line: usize, block_type: &str, heading: Option<&str>) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Basic metrics
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();
        let char_count = content.len();
        
        // Count markdown elements
        let heading_count = lines.iter().filter(|line| self.heading_regex.is_match(line.trim())).count();
        let code_block_count = self.code_block_regex.find_iter(content).count();
        let list_count = lines.iter().filter(|line| self.list_regex.is_match(line)).count();
        let link_count = self.link_regex.find_iter(content).count();
        let image_count = self.image_regex.find_iter(content).count();
        
        // Count different types of content
        let text_lines = lines.iter().filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && 
            !self.heading_regex.is_match(trimmed) && 
            !self.list_regex.is_match(line) &&
            !trimmed.starts_with("```")
        }).count();
        
        let empty_lines = lines.iter().filter(|line| line.trim().is_empty()).count();
        
        metadata.insert("block_type".to_string(), block_type.to_string());
        metadata.insert("line_count".to_string(), line_count.to_string());
        metadata.insert("char_count".to_string(), char_count.to_string());
        metadata.insert("text_lines".to_string(), text_lines.to_string());
        metadata.insert("empty_lines".to_string(), empty_lines.to_string());
        metadata.insert("heading_count".to_string(), heading_count.to_string());
        metadata.insert("code_block_count".to_string(), code_block_count.to_string());
        metadata.insert("list_count".to_string(), list_count.to_string());
        metadata.insert("link_count".to_string(), link_count.to_string());
        metadata.insert("image_count".to_string(), image_count.to_string());
        metadata.insert("start_line".to_string(), start_line.to_string());
        metadata.insert("chunking_method".to_string(), "markdown".to_string());
        
        if let Some(heading) = heading {
            metadata.insert("heading".to_string(), heading.to_string());
        }
        
        // Calculate content density
        let content_density = if line_count > 0 {
            text_lines as f64 / line_count as f64
        } else {
            0.0
        };
        metadata.insert("content_density".to_string(), format!("{:.2}", content_density));

        metadata
    }

    /// Applies overlap to markdown chunks
    fn apply_overlap(&self, chunks: Vec<Chunk>, config: &ChunkingConfig) -> Vec<Chunk> {
        if config.overlap_size == 0 {
            return chunks;
        }

        let mut overlapped_chunks = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let mut overlapped_chunk = chunk.clone();
            
            // Add overlap from previous chunk (if it's a related section)
            if i > 0 {
                let prev_chunk = &chunks[i - 1];
                if let (Some(prev_heading), Some(curr_heading)) = (
                    prev_chunk.metadata.get("heading"),
                    chunk.metadata.get("heading")
                ) {
                    // Only add overlap for related sections or if headings are similar
                    if prev_heading == curr_heading || 
                       prev_heading.contains(curr_heading) || 
                       curr_heading.contains(prev_heading) {
                        
                        let overlap_text = &prev_chunk.content[
                            std::cmp::max(0, prev_chunk.content.len().saturating_sub(config.overlap_size))..
                        ];
                        overlapped_chunk.content = format!("{}\n{}", overlap_text, chunk.content);
                    }
                }
            }
            
            overlapped_chunks.push(overlapped_chunk);
        }
        
        overlapped_chunks
    }
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, text: &str, config: &ChunkingConfig) -> Result<Vec<Chunk>> {
        let chunks = if config.preserve_semantics {
            // Try heading-based chunking first
            let heading_chunks = self.chunk_by_headings(text, config);
            
            // If no headings found, use semantic block chunking
            if heading_chunks.is_empty() || heading_chunks.len() == 1 {
                self.chunk_by_semantic_blocks(text, config)
            } else {
                heading_chunks
            }
        } else {
            // Simple line-based chunking
            let lines: Vec<&str> = text.lines().collect();
            let mut chunks = Vec::new();
            let mut current_chunk = Vec::<String>::new();
            let mut start_line = 0;
            
            for (i, line) in lines.iter().enumerate() {
                current_chunk.push(line.to_string());
                
                if current_chunk.join("\n").len() > config.max_chunk_size {
                    let chunk_content = current_chunk.join("\n");
                    chunks.push(Chunk {
                        content: chunk_content.clone(),
                        start_index: start_line,
                        end_index: i,
                        chunk_type: ChunkType::Markdown,
                        metadata: self.create_markdown_metadata(&chunk_content, start_line, "mixed", None),
                    });
                    current_chunk.clear();
                    start_line = i;
                }
            }
            
            // Add remaining lines
            if !current_chunk.is_empty() {
                let chunk_content = current_chunk.join("\n");
                chunks.push(Chunk {
                    content: chunk_content.clone(),
                    start_index: start_line,
                    end_index: lines.len(),
                    chunk_type: ChunkType::Markdown,
                    metadata: self.create_markdown_metadata(&chunk_content, start_line, "mixed", None),
                });
            }
            
            chunks
        };

        Ok(self.apply_overlap(chunks, config))
    }

    fn chunk_type(&self) -> ChunkType {
        ChunkType::Markdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_chunker_creation() {
        let chunker = MarkdownChunker::new();
        assert_eq!(chunker.chunk_type(), ChunkType::Markdown);
    }

    #[test]
    fn test_chunk_by_headings() {
        let chunker = MarkdownChunker::new();
        let config = ChunkingConfig {
            max_chunk_size: 200,
            overlap_size: 20,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        };
        
        let markdown = "# Title\n\nContent here.\n\n## Subtitle\n\nMore content.";
        let chunks = chunker.chunk(markdown, &config).unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks.len() >= 2); // Should have at least 2 chunks for 2 headings
    }

    #[test]
    fn test_chunk_by_semantic_blocks() {
        let chunker = MarkdownChunker::new();
        let config = ChunkingConfig {
            max_chunk_size: 100,
            overlap_size: 10,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        };
        
        let markdown = "## Heading\n\nThis is a paragraph.\n\n```\ncode block\n```";
        let chunks = chunker.chunk(markdown, &config).unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_markdown_metadata() {
        let chunker = MarkdownChunker::new();
        let content = "# Heading\n\nThis is content.\n\n- List item";
        let metadata = chunker.create_markdown_metadata(content, 0, "section", Some("Heading"));
        
        assert!(metadata.contains_key("heading_count"));
        assert!(metadata.contains_key("list_count"));
        assert_eq!(metadata.get("heading").unwrap(), "Heading");
        assert_eq!(metadata.get("heading_count").unwrap(), "1");
        assert_eq!(metadata.get("list_count").unwrap(), "1");
    }

    #[test]
    fn test_extract_heading_text() {
        let chunker = MarkdownChunker::new();
        
        assert_eq!(chunker.extract_heading_text("# Main Title"), Some("Main Title".to_string()));
        assert_eq!(chunker.extract_heading_text("## Sub Title"), Some("Sub Title".to_string()));
        assert_eq!(chunker.extract_heading_text("No heading here"), None);
    }
}

