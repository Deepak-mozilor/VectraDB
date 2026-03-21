//! Cohere embedding provider.
//!
//! ```bash
//! COHERE_API_KEY=... ./vectradb-server \
//!   --embedding-provider cohere \
//!   --embedding-model embed-english-v3.0 \
//!   -d 1024
//! ```

use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct CohereProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimension: Option<usize>,
}

#[derive(Serialize)]
struct CohereEmbedRequest<'a> {
    model: &'a str,
    texts: Vec<&'a str>,
    input_type: &'a str,
}

#[derive(Deserialize)]
struct CohereEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl CohereProvider {
    pub fn new(api_key: String, model: String, dimension: Option<usize>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut results = self.embed_batch(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| EmbeddingError::ApiError("Empty response from Cohere".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = "https://api.cohere.ai/v1/embed";
        let body = CohereEmbedRequest {
            model: &self.model,
            texts: texts.to_vec(),
            input_type: "search_document",
        };

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| EmbeddingError::ApiError(format!("Cohere API error: {e}")))?;

        let data: CohereEmbedResponse = resp.json().await?;

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
        self.dimension.unwrap_or(1024)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "cohere"
    }
}
