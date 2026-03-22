//! GPU-accelerated batch distance computation via wgpu compute shaders.
//!
//! Provides a `GpuDistanceEngine` that computes distances between a query
//! vector and a batch of candidate vectors on the GPU. This is useful for
//! re-ranking or brute-force search on large datasets.
//!
//! Enable with the `gpu` cargo feature:
//! ```toml
//! vectradb-search = { path = "src/search", features = ["gpu"] }
//! ```

use super::DistanceMetric;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// GPU distance computation engine.
pub struct GpuDistanceEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_l2: wgpu::ComputePipeline,
    pipeline_cosine: wgpu::ComputePipeline,
    pipeline_dot: wgpu::ComputePipeline,
    max_batch: usize,
}

/// Parameters passed to the compute shader via a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    dim: u32,
    n: u32,
}

impl GpuDistanceEngine {
    /// Initialise the GPU device and compile compute pipelines.
    /// Returns `None` if no suitable GPU adapter is found.
    pub fn new(max_batch: usize) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vectradb-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .ok()?;

        let shader_src = include_str!("distance.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distance_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_src)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance_bgl"),
            entries: &[
                // params (uniform)
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                // query (storage, read-only)
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                // data (storage, read-only)
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                // output (storage, read-write)
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipeline_l2 = make_pipeline("distance_l2");
        let pipeline_cosine = make_pipeline("distance_cosine");
        let pipeline_dot = make_pipeline("distance_dot");

        Some(Self {
            device,
            queue,
            pipeline_l2,
            pipeline_cosine,
            pipeline_dot,
            max_batch,
        })
    }

    /// Compute distances from `query` to every row in `data` on the GPU.
    ///
    /// `data` is a flat slice of `n * dim` floats (row-major).
    /// Returns a Vec of `n` distances.
    pub fn batch_distances(
        &self,
        query: &[f32],
        data: &[f32],
        dim: usize,
        metric: DistanceMetric,
    ) -> Vec<f32> {
        let n = data.len() / dim;
        assert_eq!(data.len(), n * dim);
        assert_eq!(query.len(), dim);

        let pipeline = match metric {
            DistanceMetric::Euclidean => &self.pipeline_l2,
            DistanceMetric::Cosine => &self.pipeline_cosine,
            DistanceMetric::DotProduct => &self.pipeline_dot,
        };

        let params = Params {
            dim: dim as u32,
            n: n as u32,
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

        let data_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("data"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (n * std::mem::size_of::<f32>()) as u64;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("distance_bg"),
            layout: &bgl,
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
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("distance_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("distance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging_buf.unmap();

        result
    }

    /// Returns the adapter/device name for logging.
    pub fn device_name(&self) -> String {
        format!("{:?}", self.device.limits())
    }

    /// Maximum batch size this engine was configured for.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_l2_distance() {
        let engine = match GpuDistanceEngine::new(1024) {
            Some(e) => e,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let dim = 4;
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        // Two candidate vectors
        let data = vec![
            1.0, 0.0, 0.0, 0.0, // identical to query → distance 0
            0.0, 1.0, 0.0, 0.0, // distance √2 ≈ 1.414
        ];

        let dists = engine.batch_distances(&query, &data, dim, DistanceMetric::Euclidean);
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 0.0).abs() < 1e-5, "d0 = {}", dists[0]);
        assert!(
            (dists[1] - 2.0_f32.sqrt()).abs() < 1e-4,
            "d1 = {}",
            dists[1]
        );
    }

    #[test]
    fn test_gpu_cosine_distance() {
        let engine = match GpuDistanceEngine::new(1024) {
            Some(e) => e,
            None => {
                eprintln!("No GPU adapter found, skipping test");
                return;
            }
        };

        let dim = 3;
        let query = vec![1.0f32, 0.0, 0.0];
        let data = vec![
            1.0, 0.0, 0.0, // same direction → cos dist 0
            0.0, 1.0, 0.0, // orthogonal → cos dist 1
        ];

        let dists = engine.batch_distances(&query, &data, dim, DistanceMetric::Cosine);
        assert_eq!(dists.len(), 2);
        assert!(dists[0].abs() < 1e-5, "d0 = {}", dists[0]);
        assert!((dists[1] - 1.0).abs() < 1e-5, "d1 = {}", dists[1]);
    }
}
