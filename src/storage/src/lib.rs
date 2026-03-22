use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sled::{Db, Tree};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use vectradb_components::{
    filter::MetadataFilter, DatabaseStats, VectorDatabase, VectorDocument, VectorMetadata,
    VectraDBError,
};
use vectradb_search::{
    AdvancedSearch, ES4DConfig, ES4DIndex, HNSWIndex, IVFConfig, IVFIndex, LSHIndex, PQIndex,
    SQIndex, SearchAlgorithm, SearchConfig,
};

/// Fusion method for hybrid search combining dense and sparse scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionMethod {
    /// score = w_dense * dense_score + w_sparse * sparse_score
    WeightedSum {
        dense_weight: f32,
        sparse_weight: f32,
    },
    /// Reciprocal Rank Fusion: score = sum(1 / (k + rank))
    ReciprocalRankFusion { k: usize },
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::ReciprocalRankFusion { k: 60 }
    }
}

/// Result of a hybrid search combining dense and sparse retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    pub id: String,
    pub dense_score: f32,
    pub sparse_score: f32,
    pub combined_score: f32,
    pub metadata: VectorMetadata,
    pub matched_terms: Vec<String>,
}

/// Persistent vector database with multiple indexing strategies
pub struct PersistentVectorDB {
    storage: Arc<Db>,
    vectors_tree: Tree,
    metadata_tree: Tree,
    index: Box<dyn AdvancedSearch + Send + Sync>,
    config: DatabaseConfig,
    stats: DatabaseStats,
    tfidf_index: Option<vectradb_tfidf::TfIdfIndex>,
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
    /// Get a reference to the database configuration.
    pub fn config(&self) -> &DatabaseConfig {
        &self.config
    }

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
                config.index_config.search_ef,
                config.index_config.metric,
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
            SearchAlgorithm::ES4D => Box::new(ES4DIndex::new(ES4DConfig {
                dimension: config.index_config.dimension.unwrap_or(384),
                shard_length: config.index_config.shard_length.unwrap_or(64),
                m: config.index_config.m,
                ef_construction: config.index_config.ef_construction,
                search_ef: config.index_config.search_ef,
                ..Default::default()
            })),
            SearchAlgorithm::SQ => Box::new(SQIndex::new(
                config.index_config.dimension.unwrap_or(384),
                config.index_config.metric,
            )),
            SearchAlgorithm::IVF => Box::new(IVFIndex::new(IVFConfig {
                dimension: config.index_config.dimension.unwrap_or(384),
                nlist: config.index_config.ivf_nlist.unwrap_or(256),
                nprobe: config.index_config.ivf_nprobe.unwrap_or(16),
                metric: config.index_config.metric,
            })),
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
            tfidf_index: None,
        };

        // Load existing data and rebuild index
        db_instance.rebuild_index().await?;

        Ok(db_instance)
    }

    /// Rebuild the search index from persistent storage.
    /// Loads all vectors from Sled and feeds them to `build_index()`.
    async fn rebuild_index(&mut self) -> Result<(), VectraDBError> {
        let start = Instant::now();
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

        let count = documents.len();
        if count == 0 {
            return Ok(());
        }

        // Build index with loaded documents
        self.index.build_index(documents)?;

        // Update stats
        self.stats.total_vectors = self.vectors_tree.len();

        let elapsed = start.elapsed();
        eprintln!(
            "Index rebuilt: {} vectors in {:.2}s ({:.0} vectors/sec)",
            count,
            elapsed.as_secs_f64(),
            count as f64 / elapsed.as_secs_f64().max(0.001),
        );

        Ok(())
    }

    /// Serialize and store vector data
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

        self.stats.total_vectors = self.stats.total_vectors.saturating_sub(1);
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
                // Fetch actual metadata from storage when available
                let metadata = self
                    .load_vector_sync(&result.id)
                    .map(|doc| doc.metadata)
                    .unwrap_or_else(|_| vectradb_components::VectorMetadata {
                        id: result.id.clone(),
                        dimension: 0,
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    });
                vectradb_components::SimilarityResult {
                    id: result.id,
                    score: result.similarity,
                    metadata,
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

// ---- Filtered search (not part of the VectorDatabase trait) ----

impl PersistentVectorDB {
    /// Search with a per-query ef override for HNSW/ES4D.
    pub fn search_similar_with_ef(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        ef: usize,
    ) -> Result<Vec<vectradb_components::SimilarityResult>, VectraDBError> {
        let search_results = self.index.search_with_ef(&query_vector, top_k, ef)?;

        let similarity_results: Vec<vectradb_components::SimilarityResult> = search_results
            .into_iter()
            .map(|result| {
                let metadata = self
                    .load_vector_sync(&result.id)
                    .map(|doc| doc.metadata)
                    .unwrap_or_else(|_| vectradb_components::VectorMetadata {
                        id: result.id.clone(),
                        dimension: 0,
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    });
                vectradb_components::SimilarityResult {
                    id: result.id,
                    score: result.similarity,
                    metadata,
                }
            })
            .collect();

        Ok(similarity_results)
    }

    /// Search for similar vectors with metadata filtering.
    ///
    /// Over-fetches from the search index, then filters by metadata tags.
    /// This avoids modifying the search algorithms while still providing
    /// accurate filtered results.
    ///
    /// # Arguments
    /// * `query_vector` - The query vector
    /// * `top_k` - Number of results to return after filtering
    /// * `filter` - Optional metadata filter (must/must_not/should conditions)
    pub fn search_with_filter(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<vectradb_components::SimilarityResult>, VectraDBError> {
        // No filter → delegate to standard search
        let filter = match filter {
            Some(f) => f,
            None => return self.search_similar(query_vector, top_k),
        };

        // Over-fetch: request more candidates than needed since some will be filtered out.
        // Start with 10x, which handles filters that discard up to 90% of results.
        let oversample_factor = 10;
        let fetch_count = (top_k * oversample_factor).max(100);

        let search_results = self.index.search(&query_vector, fetch_count)?;

        let mut filtered = Vec::with_capacity(top_k);

        for result in search_results {
            // Fetch full metadata from persistent storage
            let metadata = match self.load_vector_sync(&result.id) {
                Ok(doc) => doc.metadata,
                Err(_) => continue,
            };

            // Apply filter
            if filter.matches(&metadata.tags) {
                filtered.push(vectradb_components::SimilarityResult {
                    id: result.id,
                    score: result.similarity,
                    metadata,
                });
                if filtered.len() >= top_k {
                    break;
                }
            }
        }

        Ok(filtered)
    }

    /// Insert multiple vectors in a single batch with one flush at the end.
    ///
    /// Much faster than calling `create_vector` in a loop because it defers
    /// the disk flush until all vectors are inserted.
    #[allow(clippy::type_complexity)]
    pub fn batch_create_vectors(
        &mut self,
        vectors: Vec<(String, Array1<f32>, Option<HashMap<String, String>>)>,
    ) -> Result<BatchInsertResult, VectraDBError> {
        let total = vectors.len();
        let mut inserted = 0usize;
        let mut errors: Vec<String> = Vec::new();

        for (id, vector, tags) in vectors {
            let document = match vectradb_components::vector_operations::create_vector_document(
                id.clone(),
                vector,
                tags,
            ) {
                Ok(doc) => doc,
                Err(e) => {
                    errors.push(format!("{}: {}", id, e));
                    continue;
                }
            };

            // Insert into index
            if let Err(e) = self.index.insert(document.clone()) {
                errors.push(format!("{}: {}", id, e));
                continue;
            }

            // Store to disk (without flushing)
            let vector_bytes = bincode::serialize(&document.data)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
            let metadata_bytes = bincode::serialize(&document.metadata)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            self.vectors_tree
                .insert(id.as_bytes(), vector_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
            self.metadata_tree
                .insert(id.as_bytes(), metadata_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            inserted += 1;
        }

        self.stats.total_vectors = self.vectors_tree.len();

        // Single flush at the end
        if self.config.auto_flush {
            self.storage
                .flush()
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        }

        Ok(BatchInsertResult {
            total,
            inserted,
            errors,
        })
    }

    /// Search by similarity threshold: returns all results with score >= min_similarity.
    pub fn search_by_threshold(
        &self,
        query_vector: Array1<f32>,
        min_similarity: f32,
        max_results: usize,
    ) -> Result<Vec<vectradb_components::SimilarityResult>, VectraDBError> {
        let search_results =
            self.index
                .search_by_threshold(&query_vector, min_similarity, max_results)?;

        let similarity_results = search_results
            .into_iter()
            .map(|result| {
                let metadata = self
                    .load_vector_sync(&result.id)
                    .map(|doc| doc.metadata)
                    .unwrap_or_else(|_| vectradb_components::VectorMetadata {
                        id: result.id.clone(),
                        dimension: 0,
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    });
                vectradb_components::SimilarityResult {
                    id: result.id,
                    score: result.similarity,
                    metadata,
                }
            })
            .collect();

        Ok(similarity_results)
    }

    /// Enable TF-IDF indexing for hybrid search.
    pub fn enable_tfidf(&mut self, config: vectradb_tfidf::TfIdfConfig) {
        self.tfidf_index = Some(vectradb_tfidf::TfIdfIndex::new(config));
    }

    /// Index a document's text for TF-IDF sparse retrieval.
    pub fn index_text(&mut self, id: &str, text: &str) {
        if let Some(tfidf) = &mut self.tfidf_index {
            tfidf.add_document(id, text);
        }
    }

    /// Remove a document from the TF-IDF index.
    pub fn remove_text(&mut self, id: &str) {
        if let Some(tfidf) = &mut self.tfidf_index {
            tfidf.remove_document(id);
        }
    }

    /// Search the TF-IDF index (sparse text search only).
    pub fn search_text(
        &self,
        query: &str,
        top_k: usize,
    ) -> Vec<vectradb_tfidf::SparseSearchResult> {
        match &self.tfidf_index {
            Some(tfidf) => tfidf.search(query, top_k),
            None => vec![],
        }
    }

    /// Hybrid search combining dense vector similarity and sparse TF-IDF.
    pub fn search_hybrid(
        &self,
        query_vector: Array1<f32>,
        query_text: &str,
        top_k: usize,
        dense_candidates: usize,
        sparse_candidates: usize,
        fusion: &FusionMethod,
    ) -> Result<Vec<HybridSearchResult>, VectraDBError> {
        // Dense search
        let dense_results = self.index.search(&query_vector, dense_candidates)?;

        // Sparse search
        let sparse_results = match &self.tfidf_index {
            Some(tfidf) => tfidf.search(query_text, sparse_candidates),
            None => vec![],
        };

        // Fuse results
        let fused = match fusion {
            FusionMethod::WeightedSum {
                dense_weight,
                sparse_weight,
            } => self.fuse_weighted_sum(
                &dense_results,
                &sparse_results,
                *dense_weight,
                *sparse_weight,
            ),
            FusionMethod::ReciprocalRankFusion { k } => {
                self.fuse_rrf(&dense_results, &sparse_results, *k)
            }
        };

        // Sort and truncate
        let mut results = fused;
        results.sort_by(|a, b| b.combined_score.total_cmp(&a.combined_score));
        results.truncate(top_k);

        Ok(results)
    }

    fn fuse_weighted_sum(
        &self,
        dense: &[vectradb_search::SearchResult],
        sparse: &[vectradb_tfidf::SparseSearchResult],
        w_dense: f32,
        w_sparse: f32,
    ) -> Vec<HybridSearchResult> {
        let mut scores: HashMap<String, (f32, f32, Vec<String>)> = HashMap::new();

        for r in dense {
            scores.entry(r.id.clone()).or_insert((0.0, 0.0, vec![])).0 = r.similarity;
        }
        for r in sparse {
            let entry = scores.entry(r.id.clone()).or_insert((0.0, 0.0, vec![]));
            entry.1 = r.score;
            entry.2 = r.matched_terms.clone();
        }

        scores
            .into_iter()
            .map(|(id, (dense_score, sparse_score, matched_terms))| {
                let combined = w_dense * dense_score + w_sparse * sparse_score;
                let metadata = self
                    .load_vector_sync(&id)
                    .map(|doc| doc.metadata)
                    .unwrap_or_else(|_| VectorMetadata {
                        id: id.clone(),
                        dimension: 0,
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    });
                HybridSearchResult {
                    id,
                    dense_score,
                    sparse_score,
                    combined_score: combined,
                    metadata,
                    matched_terms,
                }
            })
            .collect()
    }

    fn fuse_rrf(
        &self,
        dense: &[vectradb_search::SearchResult],
        sparse: &[vectradb_tfidf::SparseSearchResult],
        k: usize,
    ) -> Vec<HybridSearchResult> {
        let mut rrf_scores: HashMap<String, (f32, f32, f32, Vec<String>)> = HashMap::new();

        // Dense RRF contribution
        for (rank, r) in dense.iter().enumerate() {
            let rrf = 1.0 / (k + rank + 1) as f32;
            let entry = rrf_scores
                .entry(r.id.clone())
                .or_insert((0.0, r.similarity, 0.0, vec![]));
            entry.0 += rrf;
        }

        // Sparse RRF contribution
        for (rank, r) in sparse.iter().enumerate() {
            let rrf = 1.0 / (k + rank + 1) as f32;
            let entry = rrf_scores
                .entry(r.id.clone())
                .or_insert((0.0, 0.0, r.score, vec![]));
            entry.0 += rrf;
            entry.2 = r.score;
            entry.3 = r.matched_terms.clone();
        }

        rrf_scores
            .into_iter()
            .map(
                |(id, (combined_score, dense_score, sparse_score, matched_terms))| {
                    let metadata = self
                        .load_vector_sync(&id)
                        .map(|doc| doc.metadata)
                        .unwrap_or_else(|_| VectorMetadata {
                            id: id.clone(),
                            dimension: 0,
                            created_at: 0,
                            updated_at: 0,
                            tags: HashMap::new(),
                        });
                    HybridSearchResult {
                        id,
                        dense_score,
                        sparse_score,
                        combined_score,
                        metadata,
                        matched_terms,
                    }
                },
            )
            .collect()
    }

    /// Manually flush all pending writes to disk.
    pub fn flush(&self) -> Result<(), VectraDBError> {
        self.storage
            .flush()
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;
        Ok(())
    }

    /// Hybrid GPU search: HNSW/ES4D fetches candidates on CPU, GPU re-ranks exactly.
    ///
    /// This combines the speed of graph-based search with GPU-exact distances,
    /// achieving near-100% recall at HNSW-like latency.
    #[cfg(feature = "gpu")]
    pub fn search_gpu_rerank(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        rerank_ef: usize,
        gpu: &vectradb_search::gpu::GpuDistanceEngine,
        metric: vectradb_search::DistanceMetric,
    ) -> Result<Vec<vectradb_components::SimilarityResult>, VectraDBError> {
        let search_results =
            self.index
                .search_gpu_rerank(&query_vector, top_k, rerank_ef, gpu, metric)?;

        let similarity_results = search_results
            .into_iter()
            .map(|result| {
                let metadata = self
                    .load_vector_sync(&result.id)
                    .map(|doc| doc.metadata)
                    .unwrap_or_else(|_| vectradb_components::VectorMetadata {
                        id: result.id.clone(),
                        dimension: 0,
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    });
                vectradb_components::SimilarityResult {
                    id: result.id,
                    score: result.similarity,
                    metadata,
                }
            })
            .collect();

        Ok(similarity_results)
    }

    /// GPU brute-force search: compute distances on GPU for all stored vectors.
    ///
    /// This bypasses the HNSW index and instead sends all vectors to the GPU
    /// for parallel distance computation, achieving 100% recall.
    #[cfg(feature = "gpu")]
    pub fn search_gpu(
        &self,
        query_vector: Array1<f32>,
        top_k: usize,
        gpu: &vectradb_search::gpu::GpuDistanceEngine,
        metric: vectradb_search::DistanceMetric,
    ) -> Result<Vec<vectradb_components::SimilarityResult>, VectraDBError> {
        let dim = query_vector.len();
        let query_slice = query_vector
            .as_slice()
            .ok_or_else(|| VectraDBError::DatabaseError(anyhow::anyhow!("non-contiguous query")))?;

        // Collect all stored vectors into a flat buffer + id list
        let mut ids: Vec<String> = Vec::new();
        let mut flat_data: Vec<f32> = Vec::new();

        for entry in self.vectors_tree.iter() {
            let (id_bytes, vec_bytes) =
                entry.map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            let id = String::from_utf8(id_bytes.to_vec())
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            let data: Array1<f32> = bincode::deserialize(&vec_bytes)
                .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!(e)))?;

            if let Some(s) = data.as_slice() {
                flat_data.extend_from_slice(s);
                ids.push(id);
            }
        }

        if ids.is_empty() {
            return Ok(vec![]);
        }

        // Run GPU distance computation
        let distances = gpu.batch_distances(query_slice, &flat_data, dim, metric);

        // Build (index, distance) pairs and sort
        let mut scored: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(top_k);

        // Convert to SimilarityResult with metadata
        let results = scored
            .into_iter()
            .map(|(i, dist)| {
                let id = &ids[i];
                let similarity = match metric {
                    vectradb_search::DistanceMetric::Cosine => 1.0 - dist,
                    vectradb_search::DistanceMetric::DotProduct => -dist,
                    vectradb_search::DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
                };
                let metadata = self
                    .load_vector_sync(id)
                    .map(|doc| doc.metadata)
                    .unwrap_or_else(|_| vectradb_components::VectorMetadata {
                        id: id.clone(),
                        dimension: 0,
                        created_at: 0,
                        updated_at: 0,
                        tags: HashMap::new(),
                    });
                vectradb_components::SimilarityResult {
                    id: id.clone(),
                    score: similarity,
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }
}

/// Result of a batch insert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertResult {
    pub total: usize,
    pub inserted: usize,
    pub errors: Vec<String>,
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

        // Create config with matching dimension
        let search_config = SearchConfig {
            dimension: Some(3),
            ..Default::default()
        };

        let config = DatabaseConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            index_config: search_config,
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
