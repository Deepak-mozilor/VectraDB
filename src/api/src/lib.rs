use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    middleware,
    response::Json,
    routing::{delete, get, post, put},
    Router,
};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use vectradb_components::{
    filter::{FilterCondition, MetadataFilter},
    DatabaseStats, SimilarityResult, VectorDatabase, VectraDBError,
};
use vectradb_storage::{BatchInsertResult, DatabaseConfig, PersistentVectorDB};

pub mod auth;
pub mod metrics;
pub mod rate_limit;
pub use auth::AuthConfig;
pub use rate_limit::{RateLimitConfig, RateLimiter};

/// API server state
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<PersistentVectorDB>>,
    /// Optional embedding provider for text-based endpoints.
    pub embedder: Option<Arc<dyn vectradb_embeddings::EmbeddingProvider>>,
    /// Authentication configuration.
    pub auth: Arc<AuthConfig>,
    /// Rate limiter.
    pub rate_limiter: Arc<RateLimiter>,
    /// Prometheus metrics handle (None = metrics disabled).
    pub metrics_handle: Option<metrics_exporter_prometheus::PrometheusHandle>,
    /// GPU distance engine (optional, requires `gpu` feature).
    #[cfg(feature = "gpu")]
    pub gpu: Option<Arc<vectradb_search::gpu::GpuDistanceEngine>>,
    /// TF-IDF index for sparse text retrieval.
    pub tfidf: Option<Arc<RwLock<vectradb_tfidf::TfIdfIndex>>>,
    /// RAG pipeline.
    pub rag_pipeline: Option<Arc<vectradb_rag::RagPipeline>>,
    /// Graph-based retrieval agent.
    pub graph_agent: Option<Arc<vectradb_rag::GraphAgent>>,
}

/// Request/Response types for API endpoints

#[derive(Debug, Deserialize)]
pub struct CreateVectorRequest {
    pub id: String,
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateVectorRequest {
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertVectorRequest {
    pub vector: Vec<f32>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct BatchCreateRequest {
    pub vectors: Vec<CreateVectorRequest>,
}

/// A key-value tag condition for filtering.
#[derive(Debug, Deserialize)]
pub struct TagCondition {
    pub key: String,
    pub value: String,
}

/// Filter for narrowing search results by metadata tags.
///
/// - `must`: All conditions must match (AND)
/// - `must_not`: None of these conditions should match (AND NOT)
/// - `should`: At least one condition must match (OR)
#[derive(Debug, Deserialize)]
pub struct SearchFilter {
    pub must: Option<Vec<TagCondition>>,
    pub must_not: Option<Vec<TagCondition>>,
    pub should: Option<Vec<TagCondition>>,
}

impl SearchFilter {
    /// Convert to the internal MetadataFilter type.
    fn to_metadata_filter(&self) -> Option<MetadataFilter> {
        let mut parts = Vec::new();

        if let Some(must) = &self.must {
            for c in must {
                parts.push(MetadataFilter::Condition(FilterCondition::Equals {
                    key: c.key.clone(),
                    value: c.value.clone(),
                }));
            }
        }

        if let Some(must_not) = &self.must_not {
            for c in must_not {
                parts.push(MetadataFilter::Condition(FilterCondition::NotEquals {
                    key: c.key.clone(),
                    value: c.value.clone(),
                }));
            }
        }

        if let Some(should) = &self.should {
            if !should.is_empty() {
                let or_parts: Vec<MetadataFilter> = should
                    .iter()
                    .map(|c| {
                        MetadataFilter::Condition(FilterCondition::Equals {
                            key: c.key.clone(),
                            value: c.value.clone(),
                        })
                    })
                    .collect();
                parts.push(MetadataFilter::Or(or_parts));
            }
        }

        match parts.len() {
            0 => None,
            1 => Some(parts.remove(0)),
            _ => Some(MetadataFilter::And(parts)),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub top_k: Option<usize>,
    /// Optional per-query ef_search override (higher = better recall, slower).
    pub ef_search: Option<usize>,
    /// Optional metadata filter.
    pub filter: Option<SearchFilter>,
}

#[derive(Debug, Serialize)]
pub struct VectorResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SimilarityResult>,
    pub total_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

/// Create the API router
pub fn create_router(state: AppState) -> Router {
    let auth_state = state.auth.clone();
    let rate_limiter = state.rate_limiter.clone();

    Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/vectors", post(create_vector))
        .route("/vectors/batch", post(batch_create_vectors))
        .route("/vectors/:id", get(get_vector))
        .route("/vectors/:id", put(update_vector))
        .route("/vectors/:id", delete(delete_vector))
        .route("/vectors/:id/upsert", put(upsert_vector))
        .route("/search", post(search_vectors))
        .route("/search/gpu", post(search_gpu_handler))
        .route("/vectors", get(list_vectors))
        // Text-based endpoints (require embedding provider)
        .route("/embed", post(embed_text))
        .route("/vectors/text", post(create_vector_from_text))
        .route("/search/text", post(search_by_text))
        .route("/vectors/text/batch", post(batch_create_from_text))
        // Threshold search
        .route("/search/threshold", post(search_threshold_handler))
        // Hybrid search (dense + sparse)
        .route("/search/hybrid", post(search_hybrid_handler))
        // TF-IDF endpoints
        .route("/tfidf/index", post(tfidf_index_handler))
        .route("/tfidf/search", post(tfidf_search_handler))
        .route("/tfidf/batch", post(tfidf_batch_handler))
        // RAG endpoints
        .route("/rag/query", post(rag_query_handler))
        .route("/rag/agent", post(rag_agent_handler))
        // Evaluation endpoint
        .route("/eval/run", post(eval_run_handler))
        // Prometheus metrics endpoint
        .route("/metrics", get(metrics::metrics_handler))
        // Metrics recording middleware (innermost — records after response)
        .layer(middleware::from_fn(metrics::metrics_middleware))
        // Auth middleware (checks Bearer token on all routes except /health)
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::auth_middleware,
        ))
        // Rate limiting (outermost — runs before auth, exempt /health)
        .layer(middleware::from_fn_with_state(
            rate_limiter,
            rate_limit::rate_limit_middleware,
        ))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024)) // 256 MB for batch inserts
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                ])
                .allow_headers(Any),
        )
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> Result<Json<HashMap<String, String>>, StatusCode> {
    let mut response = HashMap::new();
    response.insert("status".to_string(), "healthy".to_string());
    response.insert("service".to_string(), "vectradb-api".to_string());
    Ok(Json(response))
}

/// Get database statistics
async fn get_stats(
    State(state): State<AppState>,
) -> Result<Json<DatabaseStats>, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    match db.get_stats() {
        Ok(stats) => Ok(Json(stats)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Validate vector data at the API boundary
fn validate_request_vector(values: &[f32]) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if values.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid vector".to_string(),
                message: "Vector must not be empty".to_string(),
            }),
        ));
    }
    for &v in values {
        if !v.is_finite() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Invalid vector".to_string(),
                    message: "Vector contains NaN or infinite values".to_string(),
                }),
            ));
        }
    }
    Ok(())
}

/// Create a new vector
async fn create_vector(
    State(state): State<AppState>,
    Json(request): Json<CreateVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let id = request.id.clone();
    let dim = request.vector.len();
    let tags = request.tags.clone().unwrap_or_default();
    let vector = Array1::from_vec(request.vector);

    let mut db = state.db.write().await;
    match db.create_vector(id.clone(), vector.clone(), request.tags) {
        Ok(_) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Ok(Json(VectorResponse {
                id,
                vector: vector.to_vec(),
                dimension: dim,
                created_at: now,
                updated_at: now,
                tags,
            }))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to create vector".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Batch create vectors (single flush at the end)
async fn batch_create_vectors(
    State(state): State<AppState>,
    Json(request): Json<BatchCreateRequest>,
) -> Result<Json<BatchInsertResult>, (StatusCode, Json<ErrorResponse>)> {
    if request.vectors.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Empty batch".to_string(),
                message: "vectors array must not be empty".to_string(),
            }),
        ));
    }

    for v in &request.vectors {
        validate_request_vector(&v.vector)?;
    }

    let batch: Vec<_> = request
        .vectors
        .into_iter()
        .map(|v| (v.id, Array1::from_vec(v.vector), v.tags))
        .collect();

    let mut db = state.db.write().await;
    match db.batch_create_vectors(batch) {
        Ok(result) => Ok(Json(result)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Batch insert failed".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Get a vector by ID
async fn get_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    match db.get_vector(&id) {
        Ok(document) => Ok(Json(VectorResponse {
            id: document.metadata.id,
            vector: document.data.to_vec(),
            dimension: document.metadata.dimension,
            created_at: document.metadata.created_at,
            updated_at: document.metadata.updated_at,
            tags: document.metadata.tags,
        })),
        Err(VectraDBError::VectorNotFound { .. }) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Vector not found".to_string(),
                message: format!("Vector with ID '{}' not found", id),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Update an existing vector
async fn update_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let vector = Array1::from_vec(request.vector);

    let mut db = state.db.write().await;
    match db.update_vector(&id, vector, request.tags) {
        Ok(_) => {
            // Fetch the updated vector to return complete information
            match db.get_vector(&id) {
                Ok(document) => Ok(Json(VectorResponse {
                    id: document.metadata.id,
                    vector: document.data.to_vec(),
                    dimension: document.metadata.dimension,
                    created_at: document.metadata.created_at,
                    updated_at: document.metadata.updated_at,
                    tags: document.metadata.tags,
                })),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to fetch updated vector".to_string(),
                        message: e.to_string(),
                    }),
                )),
            }
        }
        Err(VectraDBError::VectorNotFound { .. }) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Vector not found".to_string(),
                message: format!("Vector with ID '{}' not found", id),
            }),
        )),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to update vector".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Delete a vector by ID
async fn delete_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let mut db = state.db.write().await;
    match db.delete_vector(&id) {
        Ok(_) => Ok(StatusCode::NO_CONTENT),
        Err(VectraDBError::VectorNotFound { .. }) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Vector not found".to_string(),
                message: format!("Vector with ID '{}' not found", id),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Upsert a vector (insert or update)
async fn upsert_vector(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpsertVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let vector = Array1::from_vec(request.vector);

    let mut db = state.db.write().await;
    match db.upsert_vector(id.clone(), vector, request.tags) {
        Ok(_) => {
            // Fetch the upserted vector to return complete information
            match db.get_vector(&id) {
                Ok(document) => Ok(Json(VectorResponse {
                    id: document.metadata.id,
                    vector: document.data.to_vec(),
                    dimension: document.metadata.dimension,
                    created_at: document.metadata.created_at,
                    updated_at: document.metadata.updated_at,
                    tags: document.metadata.tags,
                })),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to fetch upserted vector".to_string(),
                        message: e.to_string(),
                    }),
                )),
            }
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to upsert vector".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// Search for similar vectors
async fn search_vectors(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_request_vector(&request.vector)?;
    let vector = Array1::from_vec(request.vector);
    let top_k = request.top_k.unwrap_or(10).clamp(1, 10000);
    let ef_search = request.ef_search;

    let metadata_filter = request.filter.as_ref().and_then(|f| f.to_metadata_filter());

    let start_time = std::time::Instant::now();
    let db = state.db.read().await;

    let search_result = if metadata_filter.is_some() {
        db.search_with_filter(vector, top_k, metadata_filter.as_ref())
    } else {
        // Use GPU reranking when available for better recall
        #[cfg(feature = "gpu")]
        {
            if let Some(gpu) = &state.gpu {
                let metric = db.config().index_config.metric;
                let rerank_ef = ef_search.unwrap_or(500).max(top_k * 10);
                db.search_gpu_rerank(vector, top_k, rerank_ef, gpu, metric)
            } else if let Some(ef) = ef_search {
                db.search_similar_with_ef(vector, top_k, ef)
            } else {
                db.search_similar(vector, top_k)
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            if let Some(ef) = ef_search {
                db.search_similar_with_ef(vector, top_k, ef)
            } else {
                db.search_similar(vector, top_k)
            }
        }
    };

    metrics::record_search_query();

    match search_result {
        Ok(results) => {
            let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
            Ok(Json(SearchResponse {
                results,
                total_time_ms: total_time,
            }))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Search failed".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// GPU brute-force search (100% recall, requires `gpu` feature)
async fn search_gpu_handler(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (state, request);
        #[allow(clippy::needless_return)]
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: "GPU not available".to_string(),
                message: "Server was not built with --features gpu".to_string(),
            }),
        ));
    }

    #[cfg(feature = "gpu")]
    {
        validate_request_vector(&request.vector)?;
        let vector = Array1::from_vec(request.vector);
        let top_k = request.top_k.unwrap_or(10).clamp(1, 10000);

        let gpu = match &state.gpu {
            Some(g) => g.clone(),
            None => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: "GPU not initialized".to_string(),
                        message: "No GPU adapter found on this system".to_string(),
                    }),
                ));
            }
        };

        let start_time = std::time::Instant::now();
        let db = state.db.read().await;
        let metric = db.config().index_config.metric;

        match db.search_gpu(vector, top_k, &gpu, metric) {
            Ok(results) => {
                let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
                Ok(Json(SearchResponse {
                    results,
                    total_time_ms: total_time,
                }))
            }
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "GPU search failed".to_string(),
                    message: e.to_string(),
                }),
            )),
        }
    }
}

/// List all vector IDs
async fn list_vectors(
    State(state): State<AppState>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().await;
    match db.list_vectors() {
        Ok(ids) => Ok(Json(ids)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// ============================================================
// Text-based endpoints (require embedding provider)
// ============================================================

#[derive(Debug, Deserialize)]
pub struct EmbedTextRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedTextResponse {
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub model: String,
    pub provider: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateVectorFromTextRequest {
    pub id: String,
    pub text: String,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct SearchByTextRequest {
    pub text: String,
    pub top_k: Option<usize>,
    pub filter: Option<SearchFilter>,
}

#[derive(Debug, Deserialize)]
pub struct BatchTextDocument {
    pub id: String,
    pub text: String,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct BatchCreateFromTextRequest {
    pub documents: Vec<BatchTextDocument>,
}

#[derive(Debug, Serialize)]
pub struct BatchCreateResponse {
    pub created: usize,
    pub errors: Vec<String>,
}

/// Helper: get the embedder or return 501.
fn require_embedder(
    state: &AppState,
) -> Result<&Arc<dyn vectradb_embeddings::EmbeddingProvider>, (StatusCode, Json<ErrorResponse>)> {
    state.embedder.as_ref().ok_or_else(|| {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: "Embeddings not configured".to_string(),
                message: "Start the server with --embedding-provider to enable text endpoints"
                    .to_string(),
            }),
        )
    })
}

/// POST /embed — embed text, return the vector
async fn embed_text(
    State(state): State<AppState>,
    Json(request): Json<EmbedTextRequest>,
) -> Result<Json<EmbedTextResponse>, (StatusCode, Json<ErrorResponse>)> {
    let embedder = require_embedder(&state)?;

    let vector = embedder.embed(&request.text).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Embedding failed".to_string(),
                message: e.to_string(),
            }),
        )
    })?;

    Ok(Json(EmbedTextResponse {
        dimension: vector.len(),
        vector,
        model: embedder.model_name().to_string(),
        provider: embedder.provider_name().to_string(),
    }))
}

/// POST /vectors/text — embed text and store as a vector
async fn create_vector_from_text(
    State(state): State<AppState>,
    Json(request): Json<CreateVectorFromTextRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let embedder = require_embedder(&state)?;

    let vector_data = embedder.embed(&request.text).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Embedding failed".to_string(),
                message: e.to_string(),
            }),
        )
    })?;

    let vector = Array1::from_vec(vector_data);
    let mut db = state.db.write().await;

    // Store the original text in tags
    let mut tags = request.tags.unwrap_or_default();
    tags.insert("_text".to_string(), request.text);

    match db.create_vector(request.id.clone(), vector, Some(tags)) {
        Ok(_) => match db.get_vector(&request.id) {
            Ok(doc) => Ok(Json(VectorResponse {
                id: doc.metadata.id,
                vector: doc.data.to_vec(),
                dimension: doc.metadata.dimension,
                created_at: doc.metadata.created_at,
                updated_at: doc.metadata.updated_at,
                tags: doc.metadata.tags,
            })),
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to fetch created vector".to_string(),
                    message: e.to_string(),
                }),
            )),
        },
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to create vector".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// POST /search/text — embed text and search for similar vectors
async fn search_by_text(
    State(state): State<AppState>,
    Json(request): Json<SearchByTextRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let embedder = require_embedder(&state)?;

    let vector_data = embedder.embed(&request.text).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Embedding failed".to_string(),
                message: e.to_string(),
            }),
        )
    })?;

    let vector = Array1::from_vec(vector_data);
    let top_k = request.top_k.unwrap_or(10).clamp(1, 10000);
    let metadata_filter = request.filter.as_ref().and_then(|f| f.to_metadata_filter());

    let start_time = std::time::Instant::now();
    let db = state.db.read().await;

    let search_result = if metadata_filter.is_some() {
        db.search_with_filter(vector, top_k, metadata_filter.as_ref())
    } else {
        db.search_similar(vector, top_k)
    };

    match search_result {
        Ok(results) => {
            let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
            Ok(Json(SearchResponse {
                results,
                total_time_ms: total_time,
            }))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Search failed".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

/// POST /vectors/text/batch — embed and store multiple texts at once
async fn batch_create_from_text(
    State(state): State<AppState>,
    Json(request): Json<BatchCreateFromTextRequest>,
) -> Result<Json<BatchCreateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let embedder = require_embedder(&state)?;

    let texts: Vec<&str> = request.documents.iter().map(|d| d.text.as_str()).collect();

    let embeddings = embedder.embed_batch(&texts).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Batch embedding failed".to_string(),
                message: e.to_string(),
            }),
        )
    })?;

    let mut db = state.db.write().await;
    let mut created = 0;
    let mut errors = Vec::new();

    for (doc, embedding) in request.documents.iter().zip(embeddings.into_iter()) {
        let vector = Array1::from_vec(embedding);
        let mut tags = doc.tags.clone().unwrap_or_default();
        tags.insert("_text".to_string(), doc.text.clone());

        match db.create_vector(doc.id.clone(), vector, Some(tags)) {
            Ok(_) => created += 1,
            Err(e) => errors.push(format!("{}: {}", doc.id, e)),
        }
    }

    Ok(Json(BatchCreateResponse { created, errors }))
}

/// Start the API server
pub async fn start_server(
    config: DatabaseConfig,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize database
    let db = PersistentVectorDB::new(config).await?;
    let state = AppState {
        db: Arc::new(RwLock::new(db)),
        embedder: None,
        auth: Arc::new(AuthConfig::disabled()),
        rate_limiter: Arc::new(RateLimiter::new(RateLimitConfig::disabled())),
        metrics_handle: None,
        #[cfg(feature = "gpu")]
        gpu: None,
        tfidf: None,
        rag_pipeline: None,
        graph_agent: None,
    };

    // Create router
    let app = create_router(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("VectraDB API server running on http://0.0.0.0:{}", port);

    axum::serve(listener, app).await?;
    Ok(())
}

// ============================================================
// New request/response types
// ============================================================

#[derive(Debug, Deserialize)]
pub struct ThresholdSearchRequest {
    pub vector: Vec<f32>,
    pub min_similarity: f32,
    pub max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    pub text: String,
    pub vector: Option<Vec<f32>>,
    pub top_k: Option<usize>,
    pub dense_candidates: Option<usize>,
    pub sparse_candidates: Option<usize>,
    pub fusion: Option<String>, // "rrf" or "weighted"
    pub dense_weight: Option<f32>,
    pub sparse_weight: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct TfIdfIndexRequest {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct TfIdfBatchRequest {
    pub documents: Vec<TfIdfIndexRequest>,
}

#[derive(Debug, Deserialize)]
pub struct TfIdfSearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RagQueryRequest {
    pub question: String,
    pub top_k: Option<usize>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RagAgentRequest {
    pub question: String,
    pub max_depth: Option<usize>,
    pub branch_factor: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct EvalRunRequest {
    pub queries: Vec<EvalQueryInput>,
    pub k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct EvalQueryInput {
    pub vector: Vec<f32>,
    pub relevant_ids: Vec<String>,
    pub relevance_scores: Option<Vec<f32>>,
}

// ============================================================
// New handler implementations
// ============================================================

async fn search_threshold_handler(
    State(state): State<AppState>,
    Json(req): Json<ThresholdSearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let query_vector = Array1::from_vec(req.vector);
    let max_results = req.max_results.unwrap_or(100);

    let db = state.db.read().await;
    let results = db
        .search_by_threshold(query_vector, req.min_similarity, max_results)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "search_error".to_string(),
                    message: e.to_string(),
                }),
            )
        })?;

    Ok(Json(SearchResponse {
        results,
        total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn search_hybrid_handler(
    State(state): State<AppState>,
    Json(req): Json<HybridSearchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let top_k = req.top_k.unwrap_or(10);
    let dense_candidates = req.dense_candidates.unwrap_or(top_k * 3);
    let sparse_candidates = req.sparse_candidates.unwrap_or(top_k * 3);

    // Get query vector: either provided or embed the text
    let query_vector = if let Some(v) = req.vector {
        Array1::from_vec(v)
    } else if let Some(embedder) = &state.embedder {
        let vec = embedder.embed(&req.text).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "embedding_error".to_string(),
                    message: e.to_string(),
                }),
            )
        })?;
        Array1::from_vec(vec)
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "missing_vector".to_string(),
                message: "Either provide 'vector' or configure an embedding provider".to_string(),
            }),
        ));
    };

    let fusion = match req.fusion.as_deref() {
        Some("weighted") => vectradb_storage::FusionMethod::WeightedSum {
            dense_weight: req.dense_weight.unwrap_or(0.7),
            sparse_weight: req.sparse_weight.unwrap_or(0.3),
        },
        _ => vectradb_storage::FusionMethod::ReciprocalRankFusion { k: 60 },
    };

    let db = state.db.read().await;
    let results = db
        .search_hybrid(
            query_vector,
            &req.text,
            top_k,
            dense_candidates,
            sparse_candidates,
            &fusion,
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "search_error".to_string(),
                    message: e.to_string(),
                }),
            )
        })?;

    Ok(Json(serde_json::json!({
        "results": results,
        "total_time_ms": start.elapsed().as_secs_f64() * 1000.0,
    })))
}

async fn tfidf_index_handler(
    State(state): State<AppState>,
    Json(req): Json<TfIdfIndexRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let tfidf = state.tfidf.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "tfidf_disabled".to_string(),
                message: "TF-IDF indexing is not enabled".to_string(),
            }),
        )
    })?;

    tfidf.write().await.add_document(&req.id, &req.text);
    Ok(Json(serde_json::json!({"status": "indexed", "id": req.id})))
}

async fn tfidf_batch_handler(
    State(state): State<AppState>,
    Json(req): Json<TfIdfBatchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let tfidf = state.tfidf.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "tfidf_disabled".to_string(),
                message: "TF-IDF indexing is not enabled".to_string(),
            }),
        )
    })?;

    let count = req.documents.len();
    let mut tfidf = tfidf.write().await;
    for doc in req.documents {
        tfidf.add_document(&doc.id, &doc.text);
    }

    Ok(Json(
        serde_json::json!({"status": "indexed", "count": count}),
    ))
}

async fn tfidf_search_handler(
    State(state): State<AppState>,
    Json(req): Json<TfIdfSearchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let tfidf = state.tfidf.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "tfidf_disabled".to_string(),
                message: "TF-IDF indexing is not enabled".to_string(),
            }),
        )
    })?;

    let top_k = req.top_k.unwrap_or(10);
    let results = tfidf.read().await.search(&req.query, top_k);

    Ok(Json(serde_json::json!({"results": results})))
}

async fn rag_query_handler(
    State(state): State<AppState>,
    Json(req): Json<RagQueryRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let rag = state.rag_pipeline.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "rag_disabled".to_string(),
                message: "RAG pipeline is not configured. Set --llm-provider and --enable-rag"
                    .to_string(),
            }),
        )
    })?;

    let response = rag.query(&req.question).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "rag_error".to_string(),
                message: e.to_string(),
            }),
        )
    })?;

    Ok(Json(serde_json::json!(response)))
}

async fn rag_agent_handler(
    State(state): State<AppState>,
    Json(req): Json<RagAgentRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let agent = state.graph_agent.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "agent_disabled".to_string(),
                message: "Graph agent is not configured. Set --llm-provider and --enable-rag"
                    .to_string(),
            }),
        )
    })?;

    let response = agent.query(&req.question).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "agent_error".to_string(),
                message: e.to_string(),
            }),
        )
    })?;

    Ok(Json(serde_json::json!(response)))
}

async fn eval_run_handler(
    State(state): State<AppState>,
    Json(req): Json<EvalRunRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let k = req.k.unwrap_or(10);
    let db = state.db.read().await;

    let mut eval_results = Vec::new();
    for (i, q) in req.queries.iter().enumerate() {
        let query_vector = Array1::from_vec(q.vector.clone());
        let start = std::time::Instant::now();
        let results = db.search_similar(query_vector, k).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "search_error".to_string(),
                    message: e.to_string(),
                }),
            )
        })?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let retrieved_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        let gt = vectradb_eval::QueryGroundTruth {
            query_id: format!("q{i}"),
            relevant_ids: q.relevant_ids.clone(),
            relevance_scores: q.relevance_scores.clone(),
        };

        eval_results.push((retrieved_ids, gt, latency_ms));
    }

    let report = vectradb_eval::Evaluator::evaluate(&eval_results, k);
    Ok(Json(serde_json::json!(report)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_vector_request() {
        let request = CreateVectorRequest {
            id: "test_id".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            tags: Some(HashMap::from([(
                "category".to_string(),
                "test".to_string(),
            )])),
        };

        assert_eq!(request.id, "test_id");
        assert_eq!(request.vector.len(), 3);
    }

    #[tokio::test]
    async fn test_search_request() {
        let request = SearchRequest {
            vector: vec![1.0, 2.0, 3.0],
            top_k: Some(5),
            ef_search: None,
            filter: None,
        };

        assert_eq!(request.vector.len(), 3);
        assert_eq!(request.top_k, Some(5));
    }
}

use tower_http::cors::{CorsLayer, Any};

let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods(Any)
    .allow_headers(Any);

let app = router.layer(cors);