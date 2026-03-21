//! Ollama embedding provider — local, free, no API key needed.
//!
//! Requires [Ollama](https://ollama.ai) running locally.
//!
//! ```bash
//! # Install and start Ollama, then pull a model:
//! ollama pull nomic-embed-text
//!
//! # Start VectraDB with Ollama embeddings:
//! ./vectradb-server --embedding-provider ollama --embedding-model nomic-embed-text -d 768
//! ```

use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Ollama embedding provider.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimension: Option<usize>,
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaProvider {
    pub fn new(base_url: String, model: String, dimension: Option<usize>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut results = self.embed_batch(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| EmbeddingError::ApiError("Empty response from Ollama".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!("{}/api/embed", self.base_url);
        let body = OllamaEmbedRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| EmbeddingError::ApiError(format!("Ollama API error: {e}")))?;

        let data: OllamaEmbedResponse = resp.json().await?;

        // Validate dimension if configured
        if let Some(expected) = self.dimension {
            for emb in &data.embeddings {
                if emb.len() != expected {
                    return Err(EmbeddingError::DimensionMismatch {
                        expected,
                        actual: emb.len(),
                    });
                }
            }
        }

        Ok(data.embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension.unwrap_or(768) // nomic-embed-text default
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "ollama"
    }
}
