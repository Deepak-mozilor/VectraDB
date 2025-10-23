use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sled::{Db, Tree};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use vectradb_components::{
    DatabaseStats, VectorDatabase, VectorDocument, VectorMetadata, VectraDBError,
};
use vectradb_search::{
    AdvancedSearch, HNSWIndex, LSHIndex, PQIndex, SearchAlgorithm, SearchConfig,
};

/// Persistent vector database with multiple indexing strategies
pub struct PersistentVectorDB {
    storage: Arc<Db>,
    vectors_tree: Tree,
    metadata_tree: Tree,
    index: Box<dyn AdvancedSearch + Send + Sync>,
    config: DatabaseConfig,
    stats: DatabaseStats,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub data_dir: String,
    pub search_algorithm: SearchAlgorithm,
    pub index_config: SearchConfig,
    pub auto_flush: bool,
    pub cache_size: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            data_dir: "./vectradb_data".to_string(),
            search_algorithm: SearchAlgorithm::HNSW,
            index_config: SearchConfig::default(),
            auto_flush: true,
            cache_size: 1000,
        }
    }
}

impl PersistentVectorDB {
    /// Create a new persistent vector database
    pub async fn new(config: DatabaseConfig) -> Result<Self, VectraDBError> {
        let db = sled::open(&config.data_dir)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        let vectors_tree = db
            .open_tree("vectors")
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        let metadata_tree = db
            .open_tree("metadata")
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Create search index based on configuration
        let index: Box<dyn AdvancedSearch + Send + Sync> = match config.search_algorithm {
            SearchAlgorithm::HNSW => Box::new(HNSWIndex::new(
                config.index_config.dimension.unwrap_or(384),
                config.index_config.m,
                config.index_config.ef_construction,
            )),
            SearchAlgorithm::LSH => Box::new(LSHIndex::new(
                config.index_config.dimension.unwrap_or(384),
                config.index_config.num_hashes,
            )),
            SearchAlgorithm::PQ => Box::new(PQIndex::new(
                config.index_config.dimension.unwrap_or(384),
                config.index_config.num_subspaces.unwrap_or(8),
                config.index_config.codes_per_subspace.unwrap_or(256),
            )),
            _ => {
                return Err(VectraDBError::DatabaseError(anyhow::anyhow!(
                    "Unsupported search algorithm"
                )))
            }
        };

        let mut db_instance = Self {
            storage: Arc::new(db),
            vectors_tree,
            metadata_tree,
            index,
            config,
            stats: DatabaseStats::default(),
        };

        // Load existing data and rebuild index
        db_instance.rebuild_index().await?;

        Ok(db_instance)
    }

    /// Rebuild the search index from persistent storage
    async fn rebuild_index(&mut self) -> Result<(), VectraDBError> {
        let mut documents = Vec::new();

        for result in self.vectors_tree.iter() {
            let (id_bytes, vector_bytes) =
                result.map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            let id = String::from_utf8(id_bytes.to_vec())
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            // Get metadata
            let metadata_bytes = self
                .metadata_tree
                .get(&id)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
                .ok_or_else(|| VectraDBError::VectorNotFound { id: id.clone() })?;

            let metadata: VectorMetadata = bincode::deserialize(&metadata_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            // Deserialize vector data
            let data: Array1<f32> = bincode::deserialize(&vector_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            let document = VectorDocument { metadata, data };
            documents.push(document);
        }

        // Build index with loaded documents
        self.index.build_index(documents)?;

        // Update stats
        self.stats.total_vectors = self.vectors_tree.len();

        Ok(())
    }

    /// Serialize and store vector data
    async fn store_vector(&self, id: &str, document: &VectorDocument) -> Result<(), VectraDBError> {
        // Serialize vector data
        let vector_bytes = bincode::serialize(&document.data)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        let metadata_bytes = bincode::serialize(&document.metadata)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Store in database
        self.vectors_tree
            .insert(id.as_bytes(), vector_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        self.metadata_tree
            .insert(id.as_bytes(), metadata_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Flush if auto-flush is enabled
        if self.config.auto_flush {
            self.storage
                .flush_async()
                .await
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        }

        Ok(())
    }

    /// Serialize and store vector data (sync version)
    fn store_vector_sync(&self, id: &str, document: &VectorDocument) -> Result<(), VectraDBError> {
        // Serialize vector data
        let vector_bytes = bincode::serialize(&document.data)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        let metadata_bytes = bincode::serialize(&document.metadata)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Store in database
        self.vectors_tree
            .insert(id.as_bytes(), vector_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        self.metadata_tree
            .insert(id.as_bytes(), metadata_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Flush if auto-flush is enabled
        if self.config.auto_flush {
            self.storage
                .flush()
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        }

        Ok(())
    }

    /// Load vector from persistent storage
    async fn load_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        // Load metadata
        let metadata_bytes = self
            .metadata_tree
            .get(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        let metadata: VectorMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Load vector data
        let vector_bytes = self
            .vectors_tree
            .get(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        let data: Array1<f32> = bincode::deserialize(&vector_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        Ok(VectorDocument { metadata, data })
    }

    /// Load vector from persistent storage (sync version)
    fn load_vector_sync(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        // Load metadata
        let metadata_bytes = self
            .metadata_tree
            .get(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        let metadata: VectorMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        // Load vector data
        let vector_bytes = self
            .vectors_tree
            .get(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
            .ok_or_else(|| VectraDBError::VectorNotFound { id: id.to_string() })?;

        let data: Array1<f32> = bincode::deserialize(&vector_bytes)
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        Ok(VectorDocument { metadata, data })
    }

    /// Remove vector from persistent storage
    async fn remove_stored_vector(&self, id: &str) -> Result<(), VectraDBError> {
        self.vectors_tree
            .remove(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        self.metadata_tree
            .remove(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        if self.config.auto_flush {
            self.storage
                .flush_async()
                .await
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        }

        Ok(())
    }

    /// Remove vector from persistent storage (sync version)
    fn remove_stored_vector_sync(&self, id: &str) -> Result<(), VectraDBError> {
        self.vectors_tree
            .remove(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        self.metadata_tree
            .remove(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

        if self.config.auto_flush {
            self.storage
                .flush()
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        }

        Ok(())
    }
}

impl VectorDatabase for PersistentVectorDB {
    fn create_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        // Check if vector already exists
        if self
            .vectors_tree
            .contains_key(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
        {
            return Err(VectraDBError::VectorNotFound { id });
        }

        let document = vectradb_components::vector_operations::create_vector_document(
            id.clone(),
            vector,
            tags,
        )?;

        // Store in index
        self.index.insert(document.clone())?;

        // Store in persistent storage (sync operation)
        self.store_vector_sync(&id, &document)?;

        self.stats.total_vectors += 1;
        Ok(())
    }

    fn get_vector(&self, id: &str) -> Result<VectorDocument, VectraDBError> {
        self.load_vector_sync(id)
    }

    fn update_vector(
        &mut self,
        id: &str,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        // Load existing document
        let existing_doc = self.load_vector_sync(id)?;

        // Update document
        let updated_doc = vectradb_components::vector_operations::update_vector_document(
            existing_doc,
            vector,
            tags,
        )?;

        // Update in index
        self.index.update(id, updated_doc.clone())?;

        // Update in persistent storage
        self.store_vector_sync(id, &updated_doc)?;

        Ok(())
    }

    fn delete_vector(&mut self, id: &str) -> Result<(), VectraDBError> {
        // Remove from index
        self.index.remove(id)?;

        // Remove from persistent storage
        self.remove_stored_vector_sync(id)?;

        self.stats.total_vectors -= 1;
        Ok(())
    }

    fn upsert_vector(
        &mut self,
        id: String,
        vector: Array1<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> Result<(), VectraDBError> {
        if self
            .vectors_tree
            .contains_key(id.as_bytes())
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?
        {
            self.update_vector(&id, vector, tags)
        } else {
            self.create_vector(id, vector, tags)
        }
    }

    fn search_similar(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
    ) -> Result<Vec<vectradb_components::SimilarityResult>, VectraDBError> {
        let search_results = self.index.search(&query_vector, top_k)?;

        let similarity_results: Vec<vectradb_components::SimilarityResult> = search_results
            .into_iter()
            .map(|result| {
                let id = result.id.clone();
                vectradb_components::SimilarityResult {
                    id: result.id,
                    score: result.similarity,
                    metadata: vectradb_components::VectorMetadata {
                        id,
                        dimension: 0, // Will be filled from actual document
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    },
                }
            })
            .collect();

        Ok(similarity_results)
    }

    fn list_vectors(&self) -> Result<Vec<String>, VectraDBError> {
        let mut ids = Vec::new();

        for result in self.vectors_tree.iter() {
            let (id_bytes, _) =
                result.map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            let id = String::from_utf8(id_bytes.to_vec())
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            ids.push(id);
        }

        Ok(ids)
    }

    fn get_stats(&self) -> Result<DatabaseStats, VectraDBError> {
        let index_stats = self.index.get_stats();

        Ok(DatabaseStats {
            total_vectors: self.stats.total_vectors,
            dimension: self.config.index_config.dimension.unwrap_or(384),
            memory_usage: index_stats.index_size_bytes as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_persistent_db_creation() {
        let temp_dir = tempdir().unwrap();
        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };

        let db = PersistentVectorDB::new(config).await;
        assert!(db.is_ok());
    }

    #[tokio::test]
    async fn test_persistent_db_operations() {
        let temp_dir = tempdir().unwrap();
        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };

        let mut db = PersistentVectorDB::new(config).await.unwrap();

        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(db
            .create_vector("test_id".to_string(), vector, None)
            .is_ok());
        assert!(db.get_vector("test_id").is_ok());
        assert!(db.delete_vector("test_id").is_ok());
    }
}
