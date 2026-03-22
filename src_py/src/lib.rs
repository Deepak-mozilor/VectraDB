#![allow(non_local_definitions)]

use ndarray::Array1;
use pyo3::prelude::*;
use std::collections::HashMap;
use vectradb_components::{DatabaseStats, SimilarityResult, VectorDatabase, VectorDocument};
use vectradb_search::SearchAlgorithm;
use vectradb_storage::{DatabaseConfig, PersistentVectorDB};

/// Python wrapper for VectraDB
#[pyclass]
pub struct VectraDB {
    db: PersistentVectorDB,
    #[cfg(feature = "gpu")]
    gpu: Option<vectradb_search::gpu::GpuDistanceEngine>,
}

#[pymethods]
impl VectraDB {
    /// Create a new VectraDB instance
    #[new]
    #[pyo3(signature = (data_dir=None, search_algorithm=None, dimension=None, search_ef=None))]
    pub fn new(
        data_dir: Option<String>,
        search_algorithm: Option<String>,
        dimension: Option<usize>,
        search_ef: Option<usize>,
    ) -> PyResult<Self> {
        let mut index_config = vectradb_search::SearchConfig::default();
        if let Some(dim) = dimension {
            index_config.dimension = Some(dim);
        }
        if let Some(ef) = search_ef {
            index_config.search_ef = ef;
        }
        let config = DatabaseConfig {
            data_dir: data_dir.unwrap_or_else(|| "./vectradb_data".to_string()),
            search_algorithm: match search_algorithm
                .unwrap_or_else(|| "hnsw".to_string())
                .as_str()
            {
                "hnsw" => SearchAlgorithm::HNSW,
                "lsh" => SearchAlgorithm::LSH,
                "pq" => SearchAlgorithm::PQ,
                "es4d" => SearchAlgorithm::ES4D,
                _ => SearchAlgorithm::HNSW,
            },
            index_config,
            ..Default::default()
        };

        // For now, we'll use a blocking approach with tokio runtime
        // In a production environment, you'd want to handle async properly
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let db = rt
            .block_on(PersistentVectorDB::new(config))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        #[cfg(feature = "gpu")]
        let gpu = vectradb_search::gpu::GpuDistanceEngine::new(100_000);

        Ok(Self {
            db,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }

    /// Create a new vector
    pub fn create_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let array_vector = Array1::from_vec(vector);
        self.db
            .create_vector(id, array_vector, tags)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    /// Get a vector by ID
    pub fn get_vector(&self, id: &str) -> PyResult<PyVectorDocument> {
        let document = self
            .db
            .get_vector(id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;

        Ok(PyVectorDocument::from(document))
    }

    /// Update an existing vector
    pub fn update_vector(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let array_vector = Array1::from_vec(vector);
        self.db
            .update_vector(id, array_vector, tags)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    /// Delete a vector by ID
    pub fn delete_vector(&mut self, id: &str) -> PyResult<()> {
        self.db
            .delete_vector(id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;
        Ok(())
    }

    /// Upsert a vector (insert or update)
    pub fn upsert_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        let array_vector = Array1::from_vec(vector);
        self.db
            .upsert_vector(id, array_vector, tags)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }

    /// Search for similar vectors (releases GIL for parallel search)
    pub fn search_similar(
        &self,
        py: Python<'_>,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
    ) -> PyResult<Vec<PySimilarityResult>> {
        let array_query = Array1::from_vec(query_vector);
        let top_k = top_k.unwrap_or(10);

        // Release Python GIL during the Rust search — enables true parallel search
        let results = py.allow_threads(|| {
            self.db.search_similar(array_query, top_k)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let py_results: Vec<PySimilarityResult> =
            results.into_iter().map(PySimilarityResult::from).collect();

        Ok(py_results)
    }

    /// Batch create vectors (much faster — single flush at end, releases GIL)
    pub fn batch_create(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        tags: Option<Vec<Option<HashMap<String, String>>>>,
    ) -> PyResult<(usize, usize)> {
        let batch: Vec<_> = ids
            .into_iter()
            .zip(vectors.into_iter())
            .enumerate()
            .map(|(i, (id, vec))| {
                let t = tags.as_ref().and_then(|ts| ts.get(i).cloned()).flatten();
                (id, Array1::from_vec(vec), t)
            })
            .collect();

        let result = py.allow_threads(|| {
            self.db.batch_create_vectors(batch)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok((result.inserted, result.total))
    }

    /// Search with GPU reranking (HNSW candidates → GPU exact distances)
    #[cfg(feature = "gpu")]
    #[pyo3(signature = (query_vector, top_k=None, rerank_ef=None))]
    pub fn search_gpu(
        &self,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
        rerank_ef: Option<usize>,
    ) -> PyResult<Vec<PySimilarityResult>> {
        let gpu = self.gpu.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No GPU adapter found")
        })?;

        let array_query = Array1::from_vec(query_vector);
        let top_k = top_k.unwrap_or(10);
        let rerank_ef = rerank_ef.unwrap_or(500);
        let metric = self.db.config().index_config.metric;

        let results = self
            .db
            .search_gpu_rerank(array_query, top_k, rerank_ef, gpu, metric)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(results.into_iter().map(PySimilarityResult::from).collect())
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        { self.gpu.is_some() }
        #[cfg(not(feature = "gpu"))]
        { false }
    }

    /// List all vector IDs
    pub fn list_vectors(&self) -> PyResult<Vec<String>> {
        self.db
            .list_vectors()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get database statistics
    pub fn get_stats(&self) -> PyResult<PyDatabaseStats> {
        let stats = self
            .db
            .get_stats()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(PyDatabaseStats::from(stats))
    }
}

/// Python wrapper for VectorDocument
#[pyclass]
#[derive(Clone)]
pub struct PyVectorDocument {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub vector: Vec<f32>,
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get)]
    pub created_at: u64,
    #[pyo3(get)]
    pub updated_at: u64,
    #[pyo3(get)]
    pub tags: HashMap<String, String>,
}

impl From<VectorDocument> for PyVectorDocument {
    fn from(doc: VectorDocument) -> Self {
        Self {
            id: doc.metadata.id,
            vector: doc.data.to_vec(),
            dimension: doc.metadata.dimension,
            created_at: doc.metadata.created_at,
            updated_at: doc.metadata.updated_at,
            tags: doc.metadata.tags,
        }
    }
}

/// Python wrapper for SimilarityResult
#[pyclass]
#[derive(Clone)]
pub struct PySimilarityResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub tags: HashMap<String, String>,
}

impl From<SimilarityResult> for PySimilarityResult {
    fn from(result: SimilarityResult) -> Self {
        Self {
            id: result.id,
            score: result.score,
            tags: result.metadata.tags,
        }
    }
}

/// Python wrapper for DatabaseStats
#[pyclass]
#[derive(Clone)]
pub struct PyDatabaseStats {
    #[pyo3(get)]
    pub total_vectors: usize,
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get)]
    pub memory_usage: u64,
}

impl From<DatabaseStats> for PyDatabaseStats {
    fn from(stats: DatabaseStats) -> Self {
        Self {
            total_vectors: stats.total_vectors,
            dimension: stats.dimension,
            memory_usage: stats.memory_usage,
        }
    }
}

/// Python module definition
#[pymodule]
fn vectradb_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VectraDB>()?;
    m.add_class::<PyVectorDocument>()?;
    m.add_class::<PySimilarityResult>()?;
    m.add_class::<PyDatabaseStats>()?;

    // Add version info
    m.add("__version__", "0.1.0")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_vector_document_conversion() {
        let doc = VectorDocument {
            metadata: vectradb_components::VectorMetadata {
                id: "test".to_string(),
                dimension: 3,
                created_at: 1234567890,
                updated_at: 1234567890,
                tags: HashMap::new(),
            },
            data: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        };

        let py_doc = PyVectorDocument::from(doc);
        assert_eq!(py_doc.id, "test");
        assert_eq!(py_doc.dimension, 3);
        assert_eq!(py_doc.vector, vec![1.0, 2.0, 3.0]);
    }
}
