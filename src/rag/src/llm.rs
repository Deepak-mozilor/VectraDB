use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::RagError;

/// Role in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Configuration for LLM completions.
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub system_prompt: Option<String>,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 1.0,
            system_prompt: None,
        }
    }
}

/// Trait for LLM providers (Ollama, OpenAI, etc.).
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a chat completion from a list of messages.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        config: &CompletionConfig,
    ) -> Result<String, RagError>;

    /// Simple text completion from a single prompt.
    async fn complete(&self, prompt: &str, config: &CompletionConfig) -> Result<String, RagError> {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.to_string(),
        }];
        self.chat(&messages, config).await
    }

    fn model_name(&self) -> &str;
    fn provider_name(&self) -> &str;
}

// ============================================================
// Ollama LLM Provider
// ============================================================

/// LLM provider for local Ollama instance.
pub struct OllamaLlm {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaLlm {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    top_p: f32,
    num_predict: usize,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

#[async_trait]
impl LlmProvider for OllamaLlm {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        config: &CompletionConfig,
    ) -> Result<String, RagError> {
        let mut ollama_messages: Vec<OllamaChatMessage> = Vec::new();

        if let Some(sys) = &config.system_prompt {
            ollama_messages.push(OllamaChatMessage {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }

        for msg in messages {
            ollama_messages.push(OllamaChatMessage {
                role: match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                }
                .to_string(),
                content: msg.content.clone(),
            });
        }

        let request = OllamaChatRequest {
            model: self.model.clone(),
            messages: ollama_messages,
            stream: false,
            options: OllamaOptions {
                temperature: config.temperature,
                top_p: config.top_p,
                num_predict: config.max_tokens,
            },
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| RagError::LlmError(format!("Ollama HTTP error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RagError::LlmError(format!("Ollama error {status}: {body}")));
        }

        let response: OllamaChatResponse = resp
            .json()
            .await
            .map_err(|e| RagError::LlmError(format!("Ollama parse error: {e}")))?;

        Ok(response.message.content)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "ollama"
    }
}

// ============================================================
// OpenAI-compatible LLM Provider
// ============================================================

/// LLM provider for OpenAI-compatible APIs.
pub struct OpenAILlm {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAILlm {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(api_key: &str, model: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIChatMessage>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
}

#[derive(Serialize)]
struct OpenAIChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    content: String,
}

#[async_trait]
impl LlmProvider for OpenAILlm {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        config: &CompletionConfig,
    ) -> Result<String, RagError> {
        let mut openai_messages: Vec<OpenAIChatMessage> = Vec::new();

        if let Some(sys) = &config.system_prompt {
            openai_messages.push(OpenAIChatMessage {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }

        for msg in messages {
            openai_messages.push(OpenAIChatMessage {
                role: match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                }
                .to_string(),
                content: msg.content.clone(),
            });
        }

        let request = OpenAIChatRequest {
            model: self.model.clone(),
            messages: openai_messages,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
        };

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| RagError::LlmError(format!("OpenAI HTTP error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RagError::LlmError(format!("OpenAI error {status}: {body}")));
        }

        let response: OpenAIChatResponse = resp
            .json()
            .await
            .map_err(|e| RagError::LlmError(format!("OpenAI parse error: {e}")))?;

        response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| RagError::LlmError("No choices in OpenAI response".to_string()))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

/// Create an LLM provider from configuration.
pub fn create_llm_provider(
    provider: &str,
    model: &str,
    api_key: Option<&str>,
    base_url: Option<&str>,
) -> Result<Box<dyn LlmProvider>, RagError> {
    match provider {
        "ollama" => {
            let url = base_url.unwrap_or("http://localhost:11434");
            Ok(Box::new(OllamaLlm::new(url, model)))
        }
        "openai" => {
            let key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok().as_deref().map(|_| ""))
                .ok_or_else(|| RagError::ConfigError("OpenAI API key required".to_string()))?;
            // Re-read from env if the provided key was empty
            let key = if key.is_empty() {
                std::env::var("OPENAI_API_KEY")
                    .map_err(|_| RagError::ConfigError("OPENAI_API_KEY not set".to_string()))?
            } else {
                key.to_string()
            };
            if let Some(url) = base_url {
                Ok(Box::new(OpenAILlm::with_base_url(&key, model, url)))
            } else {
                Ok(Box::new(OpenAILlm::new(&key, model)))
            }
        }
        _ => Err(RagError::ConfigError(format!(
            "Unknown LLM provider: {provider}"
        ))),
    }
}
