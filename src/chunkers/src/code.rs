use crate::{Chunk, ChunkType, Chunker, ChunkingConfig};
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;

/// Code chunker for source code files
#[allow(dead_code)]
pub struct CodeChunker {
    function_regex: Regex,
    class_regex: Regex,
    comment_regex: Regex,
    // language_parsers: HashMap<String, Language>, // Will be added when tree-sitter is integrated
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeChunker {
    pub fn new() -> Self {
        // Note: In a real implementation, you would initialize tree-sitter languages here
        // For now, we'll use regex-based parsing

        Self {
            function_regex: Regex::new(r"(?m)^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:async\s+)?(?:function\s+|def\s+|fn\s+|func\s+|\w+\s*\([^)]*\)\s*\{)").unwrap(),
            class_regex: Regex::new(r"(?m)^\s*(?:public\s+|private\s+|protected\s+)?(?:class\s+|struct\s+|interface\s+|trait\s+|enum\s+)").unwrap(),
            comment_regex: Regex::new(r"(?m)^\s*(?://|#|\*|\/\*|\*\/|<!--)").unwrap(),
        }
    }

    /// Detects the programming language from file extension or content
    fn detect_language(&self, filename: Option<&str>, content: &str) -> String {
        if let Some(filename) = filename {
            let ext = filename.split('.').next_back().unwrap_or("").to_lowercase();
            match ext.as_str() {
                "rs" => "rust".to_string(),
                "py" => "python".to_string(),
                "js" | "jsx" => "javascript".to_string(),
                "ts" | "tsx" => "typescript".to_string(),
                "java" => "java".to_string(),
                "c" => "c".to_string(),
                "cpp" | "cc" | "cxx" => "cpp".to_string(),
                "go" => "go".to_string(),
                "rb" => "ruby".to_string(),
                "php" => "php".to_string(),
                "cs" => "csharp".to_string(),
                "kt" => "kotlin".to_string(),
                "swift" => "swift".to_string(),
                _ => "unknown".to_string(),
            }
        } else {
            // Try to detect from content
            if content.contains("fn ") && content.contains("let ") {
                "rust".to_string()
            } else if content.contains("def ") && content.contains("import ") {
                "python".to_string()
            } else if content.contains("function ") || content.contains("const ") {
                "javascript".to_string()
            } else if content.contains("public class ") {
                "java".to_string()
            } else {
                "unknown".to_string()
            }
        }
    }

    /// Chunks code by functions and classes
    fn chunk_by_structures(
        &self,
        content: &str,
        config: &ChunkingConfig,
        language: &str,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_chunk = String::new();
        let mut start_line = 0;
        let mut brace_count = 0;
        let mut in_function = false;
        let mut in_class = false;

        for (i, line) in lines.iter().enumerate() {
            let _trimmed = line.trim();

            // Detect function/class start
            let is_function_start = self.function_regex.is_match(line);
            let is_class_start = self.class_regex.is_match(line);

            if is_function_start || is_class_start {
                // Save previous chunk if it exists
                if !current_chunk.is_empty() {
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_index: current_chunk.len(),
                        end_index: current_chunk.len(),
                        chunk_type: ChunkType::Code,
                        metadata: self.create_code_metadata(
                            &current_chunk,
                            start_line,
                            language,
                            "mixed",
                        ),
                    });
                }

                // Start new chunk
                current_chunk = line.to_string();
                start_line = i;
                in_function = is_function_start;
                in_class = is_class_start;
                brace_count = 0;
            } else if !current_chunk.is_empty() {
                current_chunk.push('\n');
                current_chunk.push_str(line);

                // Count braces to detect end of function/class
                brace_count += line.matches('{').count();
                brace_count -= line.matches('}').count();

                // Check if chunk is getting too large
                if current_chunk.len() > config.max_chunk_size {
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_index: current_chunk.len(),
                        end_index: current_chunk.len(),
                        chunk_type: ChunkType::Code,
                        metadata: self.create_code_metadata(
                            &current_chunk,
                            start_line,
                            language,
                            "large",
                        ),
                    });
                    current_chunk = String::new();
                }

                // End of function/class
                if (in_function || in_class) && brace_count == 0 {
                    let structure_type = if in_function { "function" } else { "class" };
                    chunks.push(Chunk {
                        content: current_chunk.clone(),
                        start_index: current_chunk.len(),
                        end_index: current_chunk.len(),
                        chunk_type: ChunkType::Code,
                        metadata: self.create_code_metadata(
                            &current_chunk,
                            start_line,
                            language,
                            structure_type,
                        ),
                    });
                    current_chunk = String::new();
                    in_function = false;
                    in_class = false;
                }
            }
        }

        // Add remaining content
        if !current_chunk.is_empty() {
            let chunk_len = current_chunk.len();
            chunks.push(Chunk {
                content: current_chunk.clone(),
                start_index: chunk_len,
                end_index: chunk_len,
                chunk_type: ChunkType::Code,
                metadata: self.create_code_metadata(&current_chunk, start_line, language, "mixed"),
            });
        }

        chunks
    }

    /// Chunks code by logical blocks (functions, classes, imports, etc.)
    fn chunk_by_logical_blocks(
        &self,
        content: &str,
        config: &ChunkingConfig,
        language: &str,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut current_block = Vec::<String>::new();
        let mut current_block_type = "mixed";
        let mut start_line = 0;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Determine block type
            let block_type = if trimmed.starts_with("import ")
                || trimmed.starts_with("use ")
                || trimmed.starts_with("from ")
            {
                "imports"
            } else if trimmed.starts_with("//")
                || trimmed.starts_with("#")
                || trimmed.starts_with("/*")
            {
                "comments"
            } else if self.function_regex.is_match(line) {
                "function"
            } else if self.class_regex.is_match(line) {
                "class"
            } else if trimmed.starts_with("const ")
                || trimmed.starts_with("let ")
                || trimmed.starts_with("var ")
            {
                "variables"
            } else {
                "code"
            };

            // If block type changed, save previous block
            if block_type != current_block_type && !current_block.is_empty() {
                let chunk_content = current_block.join("\n");
                chunks.push(Chunk {
                    content: chunk_content.clone(),
                    start_index: start_line,
                    end_index: i,
                    chunk_type: ChunkType::Code,
                    metadata: self.create_code_metadata(
                        &current_block.join("\n"),
                        start_line,
                        language,
                        current_block_type,
                    ),
                });
                current_block.clear();
                start_line = i;
            }

            current_block.push(line.to_string());
            current_block_type = block_type;

            // Check if block is getting too large
            if current_block.join("\n").len() > config.max_chunk_size {
                let chunk_content = current_block.join("\n");
                chunks.push(Chunk {
                    content: chunk_content.clone(),
                    start_index: start_line,
                    end_index: i,
                    chunk_type: ChunkType::Code,
                    metadata: self.create_code_metadata(
                        &chunk_content,
                        start_line,
                        language,
                        current_block_type,
                    ),
                });
                current_block.clear();
                start_line = i;
            }
        }

        // Add remaining block
        if !current_block.is_empty() {
            let chunk_content = current_block.join("\n");
            chunks.push(Chunk {
                content: chunk_content.clone(),
                start_index: start_line,
                end_index: lines.len(),
                chunk_type: ChunkType::Code,
                metadata: self.create_code_metadata(
                    &chunk_content,
                    start_line,
                    language,
                    current_block_type,
                ),
            });
        }

        chunks
    }

    /// Creates metadata for a code chunk
    fn create_code_metadata(
        &self,
        content: &str,
        start_line: usize,
        language: &str,
        block_type: &str,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Basic metrics
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();
        let char_count = content.len();

        // Count different types of lines
        let comment_lines = lines
            .iter()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("//")
                    || trimmed.starts_with("#")
                    || trimmed.starts_with("/*")
                    || trimmed.starts_with("*")
            })
            .count();

        let empty_lines = lines.iter().filter(|line| line.trim().is_empty()).count();
        let code_lines = line_count - comment_lines - empty_lines;

        // Count functions and classes
        let function_count = self.function_regex.find_iter(content).count();
        let class_count = self.class_regex.find_iter(content).count();

        // Calculate complexity (simple heuristic)
        let complexity = content.matches('{').count() + content.matches('(').count();

        metadata.insert("language".to_string(), language.to_string());
        metadata.insert("block_type".to_string(), block_type.to_string());
        metadata.insert("line_count".to_string(), line_count.to_string());
        metadata.insert("char_count".to_string(), char_count.to_string());
        metadata.insert("code_lines".to_string(), code_lines.to_string());
        metadata.insert("comment_lines".to_string(), comment_lines.to_string());
        metadata.insert("empty_lines".to_string(), empty_lines.to_string());
        metadata.insert("function_count".to_string(), function_count.to_string());
        metadata.insert("class_count".to_string(), class_count.to_string());
        metadata.insert("complexity".to_string(), complexity.to_string());
        metadata.insert("start_line".to_string(), start_line.to_string());
        metadata.insert("chunking_method".to_string(), "code".to_string());

        // Calculate comment ratio
        let comment_ratio = if line_count > 0 {
            comment_lines as f64 / line_count as f64
        } else {
            0.0
        };
        metadata.insert("comment_ratio".to_string(), format!("{:.2}", comment_ratio));

        metadata
    }

    /// Applies overlap to code chunks
    fn apply_overlap(&self, chunks: Vec<Chunk>, config: &ChunkingConfig) -> Vec<Chunk> {
        if config.overlap_size == 0 {
            return chunks;
        }

        let mut overlapped_chunks = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let mut overlapped_chunk = chunk.clone();

            // Add overlap from previous chunk (if it's a related block)
            if i > 0 {
                let prev_chunk = &chunks[i - 1];
                if let (Some(prev_type), Some(curr_type)) = (
                    prev_chunk.metadata.get("block_type"),
                    chunk.metadata.get("block_type"),
                ) {
                    // Only add overlap for related blocks
                    if prev_type == curr_type
                        || (prev_type == "function" && curr_type == "code")
                        || (prev_type == "class" && curr_type == "code")
                    {
                        let overlap_lines = std::cmp::min(
                            config.overlap_size / 50, // Rough estimate: 50 chars per line
                            5,                        // Max 5 lines of overlap
                        );

                        let prev_lines: Vec<&str> = prev_chunk.content.lines().collect();
                        if prev_lines.len() >= overlap_lines {
                            let overlap_start = prev_lines.len() - overlap_lines;
                            let overlap_content = prev_lines[overlap_start..].join("\n");
                            overlapped_chunk.content =
                                format!("{}\n{}", overlap_content, chunk.content);
                        }
                    }
                }
            }

            overlapped_chunks.push(overlapped_chunk);
        }

        overlapped_chunks
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str, config: &ChunkingConfig) -> Result<Vec<Chunk>> {
        let language = self.detect_language(None, text);

        let chunks = if config.preserve_semantics {
            // Try structure-based chunking first
            let structure_chunks = self.chunk_by_structures(text, config, &language);

            // If no structures found, use logical blocks
            if structure_chunks.is_empty() || structure_chunks.len() == 1 {
                self.chunk_by_logical_blocks(text, config, &language)
            } else {
                structure_chunks
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
                        chunk_type: ChunkType::Code,
                        metadata: self.create_code_metadata(
                            &chunk_content,
                            start_line,
                            &language,
                            "mixed",
                        ),
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
                    chunk_type: ChunkType::Code,
                    metadata: self.create_code_metadata(
                        &chunk_content,
                        start_line,
                        &language,
                        "mixed",
                    ),
                });
            }

            chunks
        };

        Ok(self.apply_overlap(chunks, config))
    }

    fn chunk_type(&self) -> ChunkType {
        ChunkType::Code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_chunker_creation() {
        let chunker = CodeChunker::new();
        assert_eq!(chunker.chunk_type(), ChunkType::Code);
    }

    #[test]
    fn test_language_detection() {
        let chunker = CodeChunker::new();

        let rust_code = "fn main() { let x = 5; }";
        assert_eq!(chunker.detect_language(None, rust_code), "rust");

        let python_code = "def main(): import os";
        assert_eq!(chunker.detect_language(None, python_code), "python");
    }

    #[test]
    fn test_chunk_by_logical_blocks() {
        let chunker = CodeChunker::new();
        let config = ChunkingConfig {
            max_chunk_size: 200,
            overlap_size: 20,
            preserve_semantics: true,
            include_metadata: true,
            custom_delimiters: None,
        };

        let code = "import os\n\nclass Test:\n    def method(self):\n        pass";
        let chunks = chunker.chunk(code, &config).unwrap();

        assert!(!chunks.is_empty());
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_code_metadata() {
        let chunker = CodeChunker::new();
        let content = "// Comment\nfunction test() {\n    return 1;\n}";
        let metadata = chunker.create_code_metadata(content, 0, "javascript", "function");

        assert!(metadata.contains_key("language"));
        assert!(metadata.contains_key("function_count"));
        assert_eq!(metadata.get("language").unwrap(), "javascript");
        assert_eq!(metadata.get("function_count").unwrap(), "1");
    }
}
