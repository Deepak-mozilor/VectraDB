//! HuggingFace Inference API embedding provider.
//!
//! ```bash
//! HF_API_KEY=hf_... ./vectradb-server \
//!   --embedding-provider huggingface \
//!   --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
//!   -d 384
//! ```

use crate::{EmbeddingError, EmbeddingProvider};
use async_trait::async_trait;
use serde::Serialize;

pub struct HuggingFaceProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimension: Option<usize>,
}

#[derive(Serialize)]
struct HFRequest<'a> {
    inputs: Vec<&'a str>,
}

impl HuggingFaceProvider {
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
impl EmbeddingProvider for HuggingFaceProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut results = self.embed_batch(&[text]).await?;
        results
            .pop()
            .ok_or_else(|| EmbeddingError::ApiError("Empty response from HuggingFace".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/{}",
            self.model
        );
        let body = HFRequest {
            inputs: texts.to_vec(),
        };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| EmbeddingError::ApiError(format!("HuggingFace API error: {e}")))?;

        let embeddings: Vec<Vec<f32>> = resp.json().await?;

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
        self.dimension.unwrap_or(384)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "huggingface"
    }
}
