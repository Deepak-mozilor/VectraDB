<h1 align="center">🧠 VectraDB Embeddings</h1>

<p align="center">
  <strong>Send text, get vectors — automatically.</strong><br/>
  Pluggable embedding model providers for VectraDB.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Ollama-local%20%7C%20free-10b981.svg" alt="Ollama" />
  <img src="https://img.shields.io/badge/OpenAI-cloud%20API-8B5CF6.svg" alt="OpenAI" />
  <img src="https://img.shields.io/badge/HuggingFace-cloud%20API-f59e0b.svg" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Cohere-cloud%20API-e11d48.svg" alt="Cohere" />
</p>

---

## 🤔 What is this?

Without embeddings, users must convert their text to vectors **externally** before storing them in VectraDB. With this module, VectraDB handles it automatically:

```
Before:  User → [external model] → vector → VectraDB
After:   User → "the cat sat on the mat" → VectraDB → vector → stored & searchable
```

You send raw text. VectraDB calls an embedding model, converts it to a vector, and stores it. Searching works the same way — send a text query, VectraDB embeds it and finds similar vectors.

---

## 🚀 Quick Start

### Option 1: Ollama (Local, Free, No API Key)

```bash
# 1. Install Ollama (https://ollama.ai)
# 2. Pull an embedding model
ollama pull nomic-embed-text

# 3. Start VectraDB with Ollama embeddings
./target/release/vectradb-server \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text \
  -d 768
```

### Option 2: OpenAI (Cloud)

```bash
# Start VectraDB with OpenAI embeddings
OPENAI_API_KEY=sk-... ./target/release/vectradb-server \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  -d 1536
```

### Store and Search by Text

```bash
# Store a document (auto-embeds the text)
curl -X POST http://localhost:8080/vectors/text \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "text": "The cat sat on the mat",
    "tags": {"category": "animals"}
  }'

# Search by text (auto-embeds the query)
curl -X POST http://localhost:8080/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "pets sitting on furniture",
    "top_k": 5
  }'
```

---

## 📡 Supported Providers

### Ollama (Local)

| Detail | Value |
|--------|-------|
| **Type** | Local — runs on your machine |
| **Cost** | Free |
| **API Key** | Not needed |
| **URL** | `http://localhost:11434` (default) |

**Supported models:**

| Model | Dimensions | Size | Best For |
|-------|:----------:|------|----------|
| `nomic-embed-text` | 768 | 274 MB | General purpose (recommended) |
| `all-minilm` | 384 | 46 MB | Lightweight, fast |
| `mxbai-embed-large` | 1024 | 670 MB | Higher quality |
| `snowflake-arctic-embed` | 1024 | 670 MB | Multilingual |

```bash
# Pull a model
ollama pull nomic-embed-text

# Start VectraDB
./vectradb-server \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text \
  --embedding-url http://localhost:11434 \
  -d 768
```

---

### OpenAI (Cloud)

| Detail | Value |
|--------|-------|
| **Type** | Cloud API |
| **Cost** | Paid (see [pricing](https://openai.com/pricing)) |
| **API Key** | Required (`OPENAI_API_KEY` env var or `--embedding-api-key`) |
| **URL** | `https://api.openai.com/v1` (default) |

**Supported models:**

| Model | Dimensions | Cost | Best For |
|-------|:----------:|------|----------|
| `text-embedding-3-small` | 1536 | $0.02/1M tokens | Cost-effective (recommended) |
| `text-embedding-3-large` | 3072 | $0.13/1M tokens | Highest quality |
| `text-embedding-ada-002` | 1536 | $0.10/1M tokens | Legacy |

```bash
# Option A: API key via environment variable
OPENAI_API_KEY=sk-... ./vectradb-server \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  -d 1536

# Option B: API key via CLI flag
./vectradb-server \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --embedding-api-key sk-... \
  -d 1536
```

> [!TIP]
> OpenAI-compatible APIs (Azure OpenAI, Together AI, Anyscale, etc.) work by setting `--embedding-url` to their endpoint.

---

### HuggingFace Inference API (Cloud)

| Detail | Value |
|--------|-------|
| **Type** | Cloud API |
| **Cost** | Free tier available |
| **API Key** | Required (`HF_API_KEY` env var or `--embedding-api-key`) |

**Supported models:**

| Model | Dimensions | Best For |
|-------|:----------:|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, lightweight |
| `BAAI/bge-large-en-v1.5` | 1024 | High quality English |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Balanced |

```bash
HF_API_KEY=hf_... ./vectradb-server \
  --embedding-provider huggingface \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  -d 384
```

---

### Cohere (Cloud)

| Detail | Value |
|--------|-------|
| **Type** | Cloud API |
| **Cost** | Free tier available |
| **API Key** | Required (`COHERE_API_KEY` env var or `--embedding-api-key`) |

**Supported models:**

| Model | Dimensions | Best For |
|-------|:----------:|----------|
| `embed-english-v3.0` | 1024 | English text |
| `embed-multilingual-v3.0` | 1024 | 100+ languages |
| `embed-english-light-v3.0` | 384 | Fast, lightweight |

```bash
COHERE_API_KEY=... ./vectradb-server \
  --embedding-provider cohere \
  --embedding-model embed-english-v3.0 \
  -d 1024
```

---

## 🌐 API Endpoints

All text endpoints require the server to be started with `--embedding-provider`. Without it, they return `501 Not Implemented`.

### POST /embed

Embed text and return the raw vector. Useful for debugging or when you want to handle storage yourself.

```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

**Response:**
```json
{
  "vector": [0.123, -0.456, 0.789, ...],
  "dimension": 768,
  "model": "nomic-embed-text",
  "provider": "ollama"
}
```

---

### POST /vectors/text

Embed text and store it as a vector. The original text is saved in the `_text` tag for later retrieval.

```bash
curl -X POST http://localhost:8080/vectors/text \
  -H "Content-Type: application/json" \
  -d '{
    "id": "article-42",
    "text": "Retrieval-Augmented Generation combines search with LLMs",
    "tags": {"category": "ai", "source": "blog"}
  }'
```

**Response:** Same as `POST /vectors` — returns the full vector document.

---

### POST /search/text

Embed a text query and search for similar vectors. Supports metadata filtering.

```bash
# Simple search
curl -X POST http://localhost:8080/search/text \
  -H "Content-Type: application/json" \
  -d '{"text": "how do RAG systems work?", "top_k": 5}'

# Search with filter
curl -X POST http://localhost:8080/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "how do RAG systems work?",
    "top_k": 5,
    "filter": {
      "must": [{"key": "category", "value": "ai"}],
      "must_not": [{"key": "status", "value": "archived"}]
    }
  }'
```

**Response:** Same as `POST /search` — returns scored results with metadata.

---

### POST /vectors/text/batch

Embed and store multiple texts in a single request. Uses batch embedding (one API call) for efficiency.

```bash
curl -X POST http://localhost:8080/vectors/text/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "d1", "text": "First document about cats", "tags": {"topic": "animals"}},
      {"id": "d2", "text": "Second document about dogs", "tags": {"topic": "animals"}},
      {"id": "d3", "text": "Third document about physics", "tags": {"topic": "science"}}
    ]
  }'
```

**Response:**
```json
{
  "created": 3,
  "errors": []
}
```

---

## ⚙️ CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--embedding-provider` | Provider: `ollama`, `openai`, `huggingface`, `cohere` | *(disabled)* |
| `--embedding-model` | Model name | `nomic-embed-text` |
| `--embedding-url` | Custom API URL | Provider default |
| `--embedding-api-key` | API key | From env var |

**Environment variables for API keys:**

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| HuggingFace | `HF_API_KEY` or `HUGGINGFACE_API_KEY` |
| Cohere | `COHERE_API_KEY` |

> [!IMPORTANT]
> The `-d` (dimension) flag must match the model's output dimension. If they don't match, vectors will be rejected with a dimension mismatch error.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  REST API    │────▶│  EmbeddingProvider│────▶│  Model Service  │
│  /vectors/text│     │  (trait)          │     │  (Ollama/OpenAI │
│  /search/text │     │                  │     │   /HF/Cohere)   │
└──────┬───────┘     └────────┬─────────┘     └─────────────────┘
       │                      │
       │                      ▼
       │              ┌──────────────┐
       │              │  Vec<f32>    │
       │              │  (embedding) │
       │              └──────┬───────┘
       │                     │
       ▼                     ▼
┌─────────────────────────────────┐
│         VectraDB Storage        │
│   (Sled + Search Index)         │
└─────────────────────────────────┘
```

The `EmbeddingProvider` trait:

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
    fn provider_name(&self) -> &str;
}
```

### Adding a Custom Provider

Implement the `EmbeddingProvider` trait and add it to the `create_provider()` factory in `lib.rs`:

```rust
pub struct MyProvider { /* ... */ }

#[async_trait]
impl EmbeddingProvider for MyProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Call your embedding API here
    }
    fn dimension(&self) -> usize { 768 }
    fn model_name(&self) -> &str { "my-model" }
    fn provider_name(&self) -> &str { "my-provider" }
}
```

---

## 📚 Examples

### RAG Pipeline (Retrieval-Augmented Generation)

```bash
# 1. Start with Ollama
ollama pull nomic-embed-text
./vectradb-server --embedding-provider ollama -d 768

# 2. Index your documents
for file in docs/*.txt; do
  id=$(basename "$file" .txt)
  text=$(cat "$file")
  curl -s -X POST http://localhost:8080/vectors/text \
    -H "Content-Type: application/json" \
    -d "{\"id\": \"$id\", \"text\": $(echo "$text" | jq -Rs .)}"
done

# 3. Search with a user question
curl -X POST http://localhost:8080/search/text \
  -d '{"text": "What is the return policy?", "top_k": 3}'
```

### Semantic Deduplication

```bash
# Embed all documents, then search each against others
# Documents with similarity > 0.95 are likely duplicates
curl -X POST http://localhost:8080/search/text \
  -d '{"text": "duplicate check content here", "top_k": 5}' \
  | jq '.results[] | select(.score > 0.95)'
```

---

<p align="center">
  <a href="../README.md">← Back to README</a> •
  <a href="../ARCHITECTURE.md">Architecture →</a>
</p>
