//! RAG (Retrieval-Augmented Generation) pipeline and graph-based retrieval agent.

pub mod graph_agent;
pub mod llm;
pub mod pipeline;

use thiserror::Error;

/// Errors from the RAG module.
#[derive(Error, Debug)]
pub enum RagError {
    #[error("LLM error: {0}")]
    LlmError(String),
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    #[error("Search error: {0}")]
    SearchError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub use graph_agent::{GraphAgent, GraphAgentConfig, GraphRetrievalResult, RetrievalNode};
pub use llm::{create_llm_provider, ChatMessage, ChatRole, CompletionConfig, LlmProvider};
pub use pipeline::{RagConfig, RagPipeline, RagResponse};
