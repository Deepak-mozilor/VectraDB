# VectraDB Chunkers

A comprehensive Rust library for intelligent text chunking with multiple strategies optimized for different content types and use cases.

## Features

- **Multiple Chunking Strategies**: Document, Code, Markdown, and Production-based chunking
- **Semantic Preservation**: Respects content boundaries and structure
- **Quality Scoring**: Advanced quality metrics for production environments
- **Adaptive Sizing**: Dynamic chunk size adjustment based on content analysis
- **Overlap Support**: Configurable overlap between chunks for context preservation
- **Rich Metadata**: Comprehensive metadata for each chunk
- **Production Ready**: Optimized for real-world applications

## Chunking Methods

### 1. Document Chunking
Optimized for general text documents with paragraph and sentence-based chunking.

```rust
use vectradb_chunkers::{create_chunker, ChunkingConfig};

let chunker = create_chunker("document");
let config = ChunkingConfig {
    max_chunk_size: 1000,
    overlap_size: 100,
    preserve_semantics: true,
    include_metadata: true,
    custom_delimiters: None,
};

let chunks = chunker.chunk(text, &config)?;
```

**Features:**
- Paragraph-based chunking with sentence fallback
- Word and sentence counting
- Reading level estimation
- Configurable overlap

### 2. Code Chunking
Specialized for source code with structure-aware chunking.

```rust
let chunker = create_chunker("code");
let chunks = chunker.chunk(code_text, &config)?;
```

**Features:**
- Function and class-based chunking
- Language detection (Rust, Python, JavaScript, etc.)
- Code metrics (complexity, comment ratio)
- Logical block preservation
- Import and comment grouping

### 3. Markdown Chunking
Optimized for markdown documents with heading hierarchy preservation.

```rust
let chunker = create_chunker("markdown");
let chunks = chunker.chunk(markdown_text, &config)?;
```

**Features:**
- Heading-based section chunking
- Semantic block recognition (code blocks, lists, tables)
- Markdown element counting
- Content density analysis
- AST-based parsing option

### 4. Production Chunking
Advanced chunking for production environments with quality optimization.

```rust
use vectradb_chunkers::{ProductionChunker, ProductionConfig, ChunkingStrategy};

let chunker = ProductionChunker::new();
let config = ProductionConfig {
    strategy: ChunkingStrategy::Adaptive,
    min_chunk_size: 200,
    max_chunk_size: 2000,
    quality_threshold: 0.7,
    enable_quality_scoring: true,
    enable_dynamic_sizing: true,
    preserve_context: true,
    context_window_size: 100,
    ..Default::default()
};

let chunks = chunker.chunk_with_production_config(text, &config)?;
```

**Features:**
- Multiple strategies (Adaptive, FixedSize, Semantic, Hybrid)
- Quality scoring (readability, information density, coherence)
- Dynamic chunk sizing
- Content analysis and classification
- Context preservation
- Performance optimization

## Configuration

### Basic Configuration
```rust
let config = ChunkingConfig {
    max_chunk_size: 1000,        // Maximum characters per chunk
    overlap_size: 100,           // Overlap between chunks
    preserve_semantics: true,    // Respect content boundaries
    include_metadata: true,      // Include rich metadata
    custom_delimiters: None,     // Custom boundary delimiters
};
```

### Production Configuration
```rust
let config = ProductionConfig {
    base_config: ChunkingConfig::default(),
    strategy: ChunkingStrategy::Adaptive,
    min_chunk_size: 200,
    max_chunk_size: 2000,
    quality_threshold: 0.7,
    enable_quality_scoring: true,
    enable_dynamic_sizing: true,
    preserve_context: true,
    context_window_size: 100,
};
```

## Chunk Structure

Each chunk contains:
- **Content**: The actual text content
- **Indices**: Start and end positions in the original text
- **Type**: Chunk type (Document, Code, Markdown, Production)
- **Metadata**: Rich metadata including:
  - Word/sentence/paragraph counts
  - Content analysis metrics
  - Quality scores (production chunking)
  - Language detection (code chunking)
  - Structure information (markdown chunking)

## Quality Metrics

Production chunking includes comprehensive quality scoring:

- **Readability Score**: Based on Flesch Reading Ease formula
- **Information Density**: Vocabulary richness and content density
- **Semantic Coherence**: Sentence transition and structure analysis
- **Structural Integrity**: Proper formatting and punctuation
- **Overall Score**: Weighted combination of all metrics

## Usage Examples

### Basic Usage
```rust
use vectradb_chunkers::{create_chunker, ChunkingConfig};

let chunker = create_chunker("document");
let config = ChunkingConfig::default();
let chunks = chunker.chunk("Your text here...", &config)?;

for chunk in chunks {
    println!("Chunk: {}", chunk.content);
    println!("Metadata: {:?}", chunk.metadata);
}
```

### Advanced Production Usage
```rust
use vectradb_chunkers::{ProductionChunker, ProductionConfig, ChunkingStrategy};

let chunker = ProductionChunker::new();
let config = ProductionConfig {
    strategy: ChunkingStrategy::Adaptive,
    quality_threshold: 0.8,
    enable_quality_scoring: true,
    ..Default::default()
};

let chunks = chunker.chunk_with_production_config(text, &config)?;

// Filter chunks by quality
let high_quality_chunks: Vec<_> = chunks.iter()
    .filter(|chunk| {
        chunk.metadata.get("overall_quality")
            .and_then(|q| q.parse::<f64>().ok())
            .map(|score| score >= 0.8)
            .unwrap_or(false)
    })
    .collect();
```

## Performance Considerations

- **Document Chunking**: Fastest, suitable for simple text processing
- **Code Chunking**: Moderate performance, preserves code structure
- **Markdown Chunking**: Good performance, respects document hierarchy
- **Production Chunking**: Slower but highest quality, includes analysis overhead

## Dependencies

- `regex`: Pattern matching for content analysis
- `pulldown-cmark`: Markdown parsing
- `tree-sitter`: Code parsing (optional, for advanced code analysis)
- `serde`: Serialization support
- `anyhow`: Error handling

## Examples

Run the examples to see all chunking methods in action:

```bash
cargo run --example chunking_examples
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.




