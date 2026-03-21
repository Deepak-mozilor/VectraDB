//! GPU-accelerated vector operations using wgpu compute shaders.
//!
//! Provides hardware-accelerated similarity computations (cosine, euclidean,
//! dot product) and batch search via WGSL compute shaders. Falls back gracefully
//! if no GPU adapter is available.

use crate::VectraDBError;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ============================================================
// WGSL compute shaders
// ============================================================

const SIMILARITY_SHADER: &str = r#"
// Similarity compute shader
// Computes pairwise similarity between a query vector and N document vectors.
//
// Binding layout:
//   @group(0) @binding(0) params:    { dim, n_docs, metric, _pad }
//   @group(0) @binding(1) query:     array<f32>   (length = dim)
//   @group(0) @binding(2) documents: array<f32>   (length = dim * n_docs, row-major)
//   @group(0) @binding(3) results:   array<f32>   (length = n_docs, output)
//
// metric: 0 = cosine, 1 = euclidean, 2 = dot product

struct Params {
    dim: u32,
    n_docs: u32,
    metric: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> documents: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(64)
fn similarity_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let doc_idx = gid.x;
    if doc_idx >= params.n_docs {
        return;
    }

    let dim = params.dim;
    let offset = doc_idx * dim;

    var dot_val: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    var diff_sq: f32 = 0.0;

    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let a = query[i];
        let b = documents[offset + i];
        dot_val = dot_val + a * b;
        norm_a = norm_a + a * a;
        norm_b = norm_b + b * b;
        let d = a - b;
        diff_sq = diff_sq + d * d;
    }

    var result: f32;
    if params.metric == 0u {
        // Cosine similarity
        let denom = sqrt(norm_a) * sqrt(norm_b);
        if denom == 0.0 {
            result = 0.0;
        } else {
            result = dot_val / denom;
        }
    } else if params.metric == 1u {
        // Euclidean distance -> similarity = 1 / (1 + dist)
        let dist = sqrt(diff_sq);
        result = 1.0 / (1.0 + dist);
    } else {
        // Dot product
        result = dot_val;
    }

    results[doc_idx] = result;
}
"#;

const BATCH_SIMILARITY_SHADER: &str = r#"
// Batch similarity: computes similarity for Q queries x N documents.
//
// Binding layout:
//   @group(0) @binding(0) params:    { dim, n_docs, n_queries, metric }
//   @group(0) @binding(1) queries:   array<f32>   (length = dim * n_queries)
//   @group(0) @binding(2) documents: array<f32>   (length = dim * n_docs)
//   @group(0) @binding(3) results:   array<f32>   (length = n_queries * n_docs)

struct Params {
    dim: u32,
    n_docs: u32,
    n_queries: u32,
    metric: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> documents: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(8, 8)
fn batch_similarity_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.y;
    let doc_idx = gid.x;
    if query_idx >= params.n_queries || doc_idx >= params.n_docs {
        return;
    }

    let dim = params.dim;
    let q_offset = query_idx * dim;
    let d_offset = doc_idx * dim;

    var dot_val: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    var diff_sq: f32 = 0.0;

    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let a = queries[q_offset + i];
        let b = documents[d_offset + i];
        dot_val = dot_val + a * b;
        norm_a = norm_a + a * a;
        norm_b = norm_b + b * b;
        let d = a - b;
        diff_sq = diff_sq + d * d;
    }

    var result: f32;
    if params.metric == 0u {
        let denom = sqrt(norm_a) * sqrt(norm_b);
        if denom == 0.0 {
            result = 0.0;
        } else {
            result = dot_val / denom;
        }
    } else if params.metric == 1u {
        let dist = sqrt(diff_sq);
        result = 1.0 / (1.0 + dist);
    } else {
        result = dot_val;
    }

    results[query_idx * params.n_docs + doc_idx] = result;
}
"#;

// ============================================================
// Types
// ============================================================

/// GPU similarity metric (maps to shader metric param).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuMetric {
    Cosine = 0,
    Euclidean = 1,
    DotProduct = 2,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SimilarityParams {
    dim: u32,
    n_docs: u32,
    metric: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BatchParams {
    dim: u32,
    n_docs: u32,
    n_queries: u32,
    metric: u32,
}

/// GPU device info returned by [`GpuAccelerator::device_info`].
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

// ============================================================
// GpuAccelerator
// ============================================================

/// Manages a wgpu device and provides GPU-accelerated vector operations.
///
/// # Example
/// ```ignore
/// use vectradb_components::gpu::{GpuAccelerator, GpuMetric};
///
/// let gpu = GpuAccelerator::new().expect("no GPU available");
/// println!("Using GPU: {:?}", gpu.device_info());
///
/// let query = vec![1.0, 0.0, 0.0];
/// let docs = vec![1.0, 0.0, 0.0,  0.0, 1.0, 0.0]; // 2 docs, dim=3
/// let scores = gpu.similarity(&query, &docs, 3, GpuMetric::Cosine).unwrap();
/// ```
pub struct GpuAccelerator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    similarity_pipeline: wgpu::ComputePipeline,
    batch_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    batch_bind_group_layout: wgpu::BindGroupLayout,
    info: GpuDeviceInfo,
}

impl GpuAccelerator {
    /// Initialize the GPU accelerator. Returns an error if no suitable GPU is found.
    pub fn new() -> Result<Self, VectraDBError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, VectraDBError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                VectraDBError::DatabaseError(anyhow::anyhow!("No suitable GPU adapter found"))
            })?;

        let adapter_info = adapter.get_info();
        let info = GpuDeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("VectraDB GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!("GPU device error: {e}")))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // --- Similarity pipeline ---
        let sim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("similarity_shader"),
            source: wgpu::ShaderSource::Wgsl(SIMILARITY_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("similarity_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform, false),
                bgl_entry(
                    1,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                ),
                bgl_entry(
                    2,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    false,
                ),
                bgl_entry(
                    3,
                    wgpu::BufferBindingType::Storage { read_only: false },
                    false,
                ),
            ],
        });

        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("similarity_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let similarity_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("similarity_pipeline"),
                layout: Some(&sim_pipeline_layout),
                module: &sim_module,
                entry_point: Some("similarity_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Batch pipeline ---
        let batch_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch_similarity_shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_SIMILARITY_SHADER.into()),
        });

        let batch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("batch_bgl"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform, false),
                    bgl_entry(
                        1,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                    bgl_entry(
                        2,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                    bgl_entry(
                        3,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                ],
            });

        let batch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("batch_pl"),
                bind_group_layouts: &[&batch_bind_group_layout],
                push_constant_ranges: &[],
            });

        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batch_pipeline"),
            layout: Some(&batch_pipeline_layout),
            module: &batch_module,
            entry_point: Some("batch_similarity_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            similarity_pipeline,
            batch_pipeline,
            bind_group_layout,
            batch_bind_group_layout,
            info,
        })
    }

    /// Returns information about the GPU device in use.
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    /// Compute similarity scores between a single query vector and N document vectors.
    ///
    /// * `query`     – flat f32 slice of length `dim`
    /// * `documents` – flat f32 slice of length `dim * N` (row-major)
    /// * `dim`       – vector dimensionality
    /// * `metric`    – similarity metric to use
    ///
    /// Returns a `Vec<f32>` of length N with one score per document.
    pub fn similarity(
        &self,
        query: &[f32],
        documents: &[f32],
        dim: usize,
        metric: GpuMetric,
    ) -> Result<Vec<f32>, VectraDBError> {
        if query.len() != dim {
            return Err(VectraDBError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            });
        }
        if documents.is_empty() || documents.len() % dim != 0 {
            return Err(VectraDBError::InvalidVector);
        }

        let n_docs = documents.len() / dim;

        let params = SimilarityParams {
            dim: dim as u32,
            n_docs: n_docs as u32,
            metric: metric as u32,
            _pad: 0,
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let query_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("query"),
                contents: bytemuck::cast_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let docs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("documents"),
                contents: bytemuck::cast_slice(documents),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let result_size = (n_docs * std::mem::size_of::<f32>()) as u64;
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("similarity_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: docs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.similarity_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n_docs + 63) / 64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging_buf, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        self.read_buffer(&staging_buf, n_docs)
    }

    /// Compute similarity scores for multiple queries against N documents in one GPU dispatch.
    ///
    /// * `queries`   – flat f32 slice of length `dim * Q`
    /// * `documents` – flat f32 slice of length `dim * N`
    /// * `dim`       – vector dimensionality
    /// * `metric`    – similarity metric
    ///
    /// Returns a `Vec<Vec<f32>>` of shape `[Q][N]`.
    pub fn batch_similarity(
        &self,
        queries: &[f32],
        documents: &[f32],
        dim: usize,
        metric: GpuMetric,
    ) -> Result<Vec<Vec<f32>>, VectraDBError> {
        if queries.is_empty() || queries.len() % dim != 0 {
            return Err(VectraDBError::InvalidVector);
        }
        if documents.is_empty() || documents.len() % dim != 0 {
            return Err(VectraDBError::InvalidVector);
        }

        let n_queries = queries.len() / dim;
        let n_docs = documents.len() / dim;

        let params = BatchParams {
            dim: dim as u32,
            n_docs: n_docs as u32,
            n_queries: n_queries as u32,
            metric: metric as u32,
        };

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let queries_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("queries"),
                contents: bytemuck::cast_slice(queries),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let docs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_documents"),
                contents: bytemuck::cast_slice(documents),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let total_results = n_queries * n_docs;
        let result_size = (total_results * std::mem::size_of::<f32>()) as u64;
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_results"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_staging"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_bg"),
            layout: &self.batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: docs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n_docs + 7) / 8) as u32, ((n_queries + 7) / 8) as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging_buf, 0, result_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let flat = self.read_buffer(&staging_buf, total_results)?;
        Ok(flat.chunks(n_docs).map(|c| c.to_vec()).collect())
    }

    /// Read back f32 results from a staging buffer.
    fn read_buffer(
        &self,
        staging_buf: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>, VectraDBError> {
        let buf_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!("GPU recv error: {e}")))?
            .map_err(|e| VectraDBError::DatabaseError(anyhow::anyhow!("GPU map error: {e}")))?;

        let data = buf_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(results)
    }

    /// Check if GPU acceleration is available on this system.
    pub fn is_available() -> bool {
        Self::new().is_ok()
    }
}

// ============================================================
// Helpers
// ============================================================

fn bgl_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    _has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<GpuAccelerator> {
        GpuAccelerator::new().ok()
    }

    #[test]
    fn test_gpu_cosine_similarity() {
        let Some(gpu) = try_gpu() else {
            eprintln!("Skipping GPU test: no GPU available");
            return;
        };
        println!("GPU: {:?}", gpu.device_info());

        let query = vec![1.0, 0.0, 0.0];
        let docs = vec![
            1.0, 0.0, 0.0, // doc0: identical
            0.0, 1.0, 0.0, // doc1: orthogonal
            1.0, 1.0, 0.0, // doc2: 45 degrees
        ];

        let scores = gpu.similarity(&query, &docs, 3, GpuMetric::Cosine).unwrap();
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < 1e-5, "identical should be ~1.0");
        assert!(scores[1].abs() < 1e-5, "orthogonal should be ~0.0");
        assert!((scores[2] - 0.7071).abs() < 1e-3, "45deg should be ~0.707");
    }

    #[test]
    fn test_gpu_euclidean_similarity() {
        let Some(gpu) = try_gpu() else { return };

        let query = vec![0.0, 0.0];
        let docs = vec![
            0.0, 0.0, // distance 0
            3.0, 4.0, // distance 5
        ];

        let scores = gpu
            .similarity(&query, &docs, 2, GpuMetric::Euclidean)
            .unwrap();
        assert!((scores[0] - 1.0).abs() < 1e-5); // 1/(1+0)
        assert!((scores[1] - 1.0 / 6.0).abs() < 1e-5); // 1/(1+5)
    }

    #[test]
    fn test_gpu_dot_product() {
        let Some(gpu) = try_gpu() else { return };

        let query = vec![1.0, 2.0, 3.0];
        let docs = vec![4.0, 5.0, 6.0]; // dot = 32
        let scores = gpu
            .similarity(&query, &docs, 3, GpuMetric::DotProduct)
            .unwrap();
        assert!((scores[0] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_gpu_batch_similarity() {
        let Some(gpu) = try_gpu() else { return };

        let queries = vec![
            1.0, 0.0, 0.0, // query 0
            0.0, 1.0, 0.0, // query 1
        ];
        let docs = vec![
            1.0, 0.0, 0.0, // doc 0
            0.0, 1.0, 0.0, // doc 1
        ];

        let results = gpu
            .batch_similarity(&queries, &docs, 3, GpuMetric::Cosine)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        // query0 vs doc0 = 1.0, query0 vs doc1 = 0.0
        assert!((results[0][0] - 1.0).abs() < 1e-5);
        assert!(results[0][1].abs() < 1e-5);
        // query1 vs doc0 = 0.0, query1 vs doc1 = 1.0
        assert!(results[1][0].abs() < 1e-5);
        assert!((results[1][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_dimension_mismatch() {
        let Some(gpu) = try_gpu() else { return };
        let result = gpu.similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3, GpuMetric::Cosine);
        assert!(result.is_err());
    }
}
