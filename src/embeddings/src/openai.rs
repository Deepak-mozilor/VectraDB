//! OpenAI embedding provider.
//!
//! ```bash
//! # Set API key and start:
//! OPENAI_API_KEY=sk-... ./vectradb-server \
//!   --embedding-provider openai \
//!   --embedding-model text-embedding-3-small \
//!   -d 1536
//! ```
//!
//! Supported models:
//! - `text-embedding-3-small` (1536 dimensions)
//! - `text-embedding-3-large` (3072 dimensions)
//! - `text-embedding-ada-002` (1536 dimensions)

use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OpenAIProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    dimension: Option<usize>,
}

#[derive(Serialize)]
struct OpenAIEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

impl OpenAIProvider {
    pub fn new(base_url: String, api_key: String, model: String, dimension: Option<usize>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            dimension,
        }
    }

    fn default_dimension(&self) -> usize {
        match self.model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-3-small" => 1536,
            "text-embedding-ada-002" => 1536,
            _ => self.dimension.unwrap_or(1536),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut results = self.embed_batch(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| EmbeddingError::ApiError("Empty response from OpenAI".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = OpenAIEmbedRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| EmbeddingError::ApiError(format!("OpenAI API error: {e}")))?;

        let data: OpenAIEmbedResponse = resp.json().await?;

        let embeddings: Vec<Vec<f32>> = data.data.into_iter().map(|e| e.embedding).collect();

        if let Some(expected) = self.dimension {
            for emb in &embeddings {
                if emb.len() != expected {
                    return Err(EmbeddingError::DimensionMismatch {
                        expected,
                        actual: emb.len(),
                    });
                }
            }
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension.unwrap_or_else(|| self.default_dimension())
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}
