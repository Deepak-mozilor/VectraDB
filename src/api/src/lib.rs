use vectradb_components::{VectorDatabase, VectorDocument, VectraDBError, SimilarityResult, DatabaseStats};
use vectradb_storage::{PersistentVectorDB, DatabaseConfig};
use vectradb_search::SearchAlgorithm;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, delete, put},
    Router,
};
use tower_http::cors::CorsLayer;

/// API server state
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<PersistentVectorDB>>,
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
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub top_k: Option<usize>,
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
        .route("/vectors/:id", get(get_vector))
        .route("/vectors/:id", put(update_vector))
        .route("/vectors/:id", delete(delete_vector))
        .route("/vectors/:id/upsert", put(upsert_vector))
        .route("/search", post(search_vectors))
        .route("/vectors", get(list_vectors))
        .layer(CorsLayer::permissive())
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

/// Create a new vector
async fn create_vector(
    State(state): State<AppState>,
    Json(request): Json<CreateVectorRequest>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let vector = Array1::from_vec(request.vector);
    
    let mut db = state.db.write().await;
    match db.create_vector(request.id.clone(), vector, request.tags) {
        Ok(_) => {
            // Fetch the created vector to return complete information
            match db.get_vector(&request.id) {
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
                        error: "Failed to fetch created vector".to_string(),
                        message: e.to_string(),
                    }),
                )),
            }
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
    let vector = Array1::from_vec(request.vector);
    let top_k = request.top_k.unwrap_or(10);
    
    let start_time = std::time::Instant::now();
    let db = state.db.read().await;
    
    match db.search_similar(vector, top_k) {
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

/// Start the API server
pub async fn start_server(config: DatabaseConfig, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize database
    let db = PersistentVectorDB::new(config).await?;
    let state = AppState {
        db: Arc::new(RwLock::new(db)),
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
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_create_vector_request() {
        let request = CreateVectorRequest {
            id: "test_id".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            tags: Some(HashMap::from([("category".to_string(), "test".to_string())])),
        };
        
        assert_eq!(request.id, "test_id");
        assert_eq!(request.vector.len(), 3);
    }

    #[tokio::test]
    async fn test_search_request() {
        let request = SearchRequest {
            vector: vec![1.0, 2.0, 3.0],
            top_k: Some(5),
        };
        
        assert_eq!(request.vector.len(), 3);
        assert_eq!(request.top_k, Some(5));
    }
}

