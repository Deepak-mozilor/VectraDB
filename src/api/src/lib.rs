use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
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

/// API server state
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<PersistentVectorDB>>,
    /// Optional embedding provider for text-based endpoints.
    pub embedder: Option<Arc<dyn vectradb_embeddings::EmbeddingProvider>>,
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
        .route("/vectors", get(list_vectors))
        // Text-based endpoints (require embedding provider)
        .route("/embed", post(embed_text))
        .route("/vectors/text", post(create_vector_from_text))
        .route("/search/text", post(search_by_text))
        .route("/vectors/text/batch", post(batch_create_from_text))
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
    } else if let Some(ef) = ef_search {
        db.search_similar_with_ef(vector, top_k, ef)
    } else {
        db.search_similar(vector, top_k)
    };

    match search_result {
        Ok(results) => {
            let total_time = start_time.elapsed().as_secs_f64() * 1000.0; // Convert to milliseconds
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
    };

    // Create router
    let app = create_router(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("VectraDB API server running on http://0.0.0.0:{}", port);

    axum::serve(listener, app).await?;
    Ok(())
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
            filter: None,
        };

        assert_eq!(request.vector.len(), 3);
        assert_eq!(request.top_k, Some(5));
    }
}
