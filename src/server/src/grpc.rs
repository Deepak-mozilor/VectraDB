use tonic::{Request, Response, Status};
use vectradb_storage::PersistentVectorDB;
use vectradb_components::VectorDatabase;
use ndarray::Array1;
use std::sync::Arc;
use tokio::sync::RwLock;

// Include generated proto code
pub mod vectradb {
    tonic::include_proto!("vectradb");
}

use vectradb::{
    vectra_db_server::{VectraDb, VectraDbServer},
    CreateVectorRequest, VectorResponse, GetVectorRequest, UpdateVectorRequest,
    DeleteVectorRequest, DeleteVectorResponse, UpsertVectorRequest, SearchRequest,
    SearchResponse, ListVectorsRequest, ListVectorsResponse, GetStatsRequest,
    StatsResponse, HealthCheckRequest, HealthCheckResponse, SimilarityResult,
    VectorMetadata,
};

pub struct VectraDbService {
    db: Arc<RwLock<PersistentVectorDB>>,
}

impl VectraDbService {
    pub fn new(db: Arc<RwLock<PersistentVectorDB>>) -> Self {
        Self { db }
    }

    pub fn into_service(self) -> VectraDbServer<Self> {
        VectraDbServer::new(self)
    }
}

#[tonic::async_trait]
impl VectraDb for VectraDbService {
    async fn create_vector(
        &self,
        request: Request<CreateVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        let tags = if req.tags.is_empty() {
            None
        } else {
            Some(req.tags)
        };

        let mut db = self.db.write().await;
        db.create_vector(req.id.clone(), vector, tags)
            .map_err(|e| Status::internal(e.to_string()))?;

        // Fetch created vector
        let document = db.get_vector(&req.id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(VectorResponse {
            id: document.metadata.id,
            vector: document.data.to_vec(),
            dimension: document.metadata.dimension as u64,
            created_at: document.metadata.created_at,
            updated_at: document.metadata.updated_at,
            tags: document.metadata.tags,
        }))
    }

    async fn get_vector(
        &self,
        request: Request<GetVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let db = self.db.read().await;
        
        let document = db.get_vector(&req.id)
            .map_err(|e| Status::not_found(e.to_string()))?;

        Ok(Response::new(VectorResponse {
            id: document.metadata.id,
            vector: document.data.to_vec(),
            dimension: document.metadata.dimension as u64,
            created_at: document.metadata.created_at,
            updated_at: document.metadata.updated_at,
            tags: document.metadata.tags,
        }))
    }

    async fn update_vector(
        &self,
        request: Request<UpdateVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        let tags = if req.tags.is_empty() {
            None
        } else {
            Some(req.tags)
        };

        let mut db = self.db.write().await;
        db.update_vector(&req.id, vector, tags)
            .map_err(|e| Status::internal(e.to_string()))?;

        let document = db.get_vector(&req.id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(VectorResponse {
            id: document.metadata.id,
            vector: document.data.to_vec(),
            dimension: document.metadata.dimension as u64,
            created_at: document.metadata.created_at,
            updated_at: document.metadata.updated_at,
            tags: document.metadata.tags,
        }))
    }

    async fn delete_vector(
        &self,
        request: Request<DeleteVectorRequest>,
    ) -> Result<Response<DeleteVectorResponse>, Status> {
        let req = request.into_inner();
        let mut db = self.db.write().await;
        
        db.delete_vector(&req.id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DeleteVectorResponse { success: true }))
    }

    async fn upsert_vector(
        &self,
        request: Request<UpsertVectorRequest>,
    ) -> Result<Response<VectorResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        let tags = if req.tags.is_empty() {
            None
        } else {
            Some(req.tags)
        };

        let mut db = self.db.write().await;
        db.upsert_vector(req.id.clone(), vector, tags)
            .map_err(|e| Status::internal(e.to_string()))?;

        let document = db.get_vector(&req.id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(VectorResponse {
            id: document.metadata.id,
            vector: document.data.to_vec(),
            dimension: document.metadata.dimension as u64,
            created_at: document.metadata.created_at,
            updated_at: document.metadata.updated_at,
            tags: document.metadata.tags,
        }))
    }

    async fn search_similar(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let vector = Array1::from_vec(req.vector);
        let top_k = req.top_k as usize;

        let start_time = std::time::Instant::now();
        let db = self.db.read().await;

        let results = db.search_similar(vector, top_k)
            .map_err(|e| Status::internal(e.to_string()))?;

        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let grpc_results = results
            .into_iter()
            .map(|r| SimilarityResult {
                id: r.id.clone(),
                score: r.score,
                metadata: Some(VectorMetadata {
                    id: r.metadata.id,
                    dimension: r.metadata.dimension as u64,
                    created_at: r.metadata.created_at,
                    updated_at: r.metadata.updated_at,
                    tags: r.metadata.tags,
                }),
            })
            .collect();

        Ok(Response::new(SearchResponse {
            results: grpc_results,
            total_time_ms,
        }))
    }

    async fn list_vectors(
        &self,
        _request: Request<ListVectorsRequest>,
    ) -> Result<Response<ListVectorsResponse>, Status> {
        let db = self.db.read().await;
        let ids = db.list_vectors()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ListVectorsResponse { ids }))
    }

    async fn get_stats(
        &self,
        _request: Request<GetStatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let db = self.db.read().await;
        let stats = db.get_stats()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(StatsResponse {
            total_vectors: stats.total_vectors as u64,
            dimension: stats.dimension as u64,
            memory_usage: stats.memory_usage,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            status: "healthy".to_string(),
            service: "vectradb-grpc".to_string(),
        }))
    }
}
