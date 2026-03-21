//! Embedding model providers for VectraDB.
//!
//! Converts raw text into vectors using external embedding services.
//! Supports multiple providers through a pluggable `EmbeddingProvider` trait.
//!
//! # Supported Providers
//!
//! | Provider | Type | Example Models |
//! |----------|------|----------------|
//! | Ollama | Local | `nomic-embed-text`, `all-minilm`, `mxbai-embed-large` |
//! | OpenAI | Cloud | `text-embedding-3-small`, `text-embedding-3-large` |
//! | HuggingFace | Cloud | `sentence-transformers/all-MiniLM-L6-v2` |
//! | Cohere | Cloud | `embed-english-v3.0`, `embed-multilingual-v3.0` |

pub mod cohere;
pub mod huggingface;
pub mod ollama;
pub mod openai;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================
// Errors
// ============================================================

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Model returned unexpected dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// ============================================================
// Provider trait
// ============================================================

/// Trait for embedding model providers.
///
/// Implementors convert text into fixed-size float vectors using an
/// external model (API or local service).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text string into a vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Embed multiple texts in one call (batch).
    /// Default implementation calls `embed` sequentially.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// The vector dimension this model produces.
    fn dimension(&self) -> usize;

    /// The model name (e.g., "text-embedding-3-small").
    fn model_name(&self) -> &str;

    /// The provider name (e.g., "openai").
    fn provider_name(&self) -> &str;
}

// ============================================================
// Configuration
// ============================================================

/// Configuration for an embedding provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider name: "ollama", "openai", "huggingface", "cohere".
    pub provider: String,

    /// Model name (e.g., "text-embedding-3-small", "nomic-embed-text").
    pub model: String,

    /// API base URL (required for Ollama, optional override for others).
    pub api_url: Option<String>,

    /// API key (required for OpenAI, Cohere, HuggingFace).
    /// Can also be set via environment variables.
    pub api_key: Option<String>,

    /// Expected embedding dimension. Used to validate responses.
    pub dimension: Option<usize>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "nomic-embed-text".to_string(),
            api_url: None,
            api_key: None,
            dimension: None,
        }
    }
}

// ============================================================
// Provider factory
// ============================================================

/// Create an embedding provider from configuration.
pub fn create_provider(
    config: &EmbeddingConfig,
) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError> {
    match config.provider.to_lowercase().as_str() {
        "ollama" => {
            let url = config
                .api_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            Ok(Box::new(ollama::OllamaProvider::new(
                url,
                config.model.clone(),
                config.dimension,
            )))
        }
        "openai" => {
            let api_key = config
                .api_key
                .clone()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| {
                    EmbeddingError::ConfigError(
                        "OpenAI API key required. Set --embedding-api-key or OPENAI_API_KEY".into(),
                    )
                })?;
            let url = config
                .api_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            Ok(Box::new(openai::OpenAIProvider::new(
                url,
                api_key,
                config.model.clone(),
                config.dimension,
            )))
        }
        "huggingface" | "hf" => {
            let api_key = config
                .api_key
                .clone()
                .or_else(|| std::env::var("HF_API_KEY").ok())
                .or_else(|| std::env::var("HUGGINGFACE_API_KEY").ok())
                .ok_or_else(|| {
                    EmbeddingError::ConfigError(
                        "HuggingFace API key required. Set --embedding-api-key or HF_API_KEY"
                            .into(),
                    )
                })?;
            Ok(Box::new(huggingface::HuggingFaceProvider::new(
                api_key,
                config.model.clone(),
                config.dimension,
            )))
        }
        "cohere" => {
            let api_key = config
                .api_key
                .clone()
                .or_else(|| std::env::var("COHERE_API_KEY").ok())
                .ok_or_else(|| {
                    EmbeddingError::ConfigError(
                        "Cohere API key required. Set --embedding-api-key or COHERE_API_KEY".into(),
                    )
                })?;
            Ok(Box::new(cohere::CohereProvider::new(
                api_key,
                config.model.clone(),
                config.dimension,
            )))
        }
        other => Err(EmbeddingError::ConfigError(format!(
            "Unknown provider: '{other}'. Supported: ollama, openai, huggingface, cohere"
        ))),
    }
}
