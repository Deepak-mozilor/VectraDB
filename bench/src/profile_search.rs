use ndarray::Array1;
use std::time::Instant;
use vectradb_components::vector_operations::create_vector_document;
use vectradb_search::{hnsw::HNSWIndex, AdvancedSearch, DistanceMetric};

fn main() {
    let dim = 128;
    let n = 10_000usize;
    let n_queries = 10_000usize;

    let mut rng_state: u64 = 42;
    let mut rand_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut gen_vec = || -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| rand_f32()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9;
        v.iter().map(|x| x / norm).collect()
    };

    eprintln!("Building HNSW index ({n} vectors, dim={dim})...");
    let mut index = HNSWIndex::new(dim, 16, 200, 500, DistanceMetric::Euclidean);
    for i in 0..n {
        let v = gen_vec();
        let doc = create_vector_document(format!("v-{i}"), Array1::from_vec(v), None).unwrap();
        index.insert(doc).unwrap();
    }
    eprintln!("Index built.");

    let queries: Vec<Array1<f32>> = (0..n_queries)
        .map(|_| Array1::from_vec(gen_vec()))
        .collect();

    // Warmup
    for q in &queries[..100] {
        let _ = index.search(q, 10);
    }

    eprintln!("Running {n_queries} searches...");
    let t0 = Instant::now();
    for q in &queries {
        let _ = index.search(q, 10).unwrap();
    }
    let elapsed = t0.elapsed();
    let qps = n_queries as f64 / elapsed.as_secs_f64();
    let avg_us = elapsed.as_micros() as f64 / n_queries as f64;
    let avg_ms = avg_us / 1000.0;
    eprintln!("Done: {qps:.0} QPS, avg latency {avg_us:.1}us ({avg_ms:.3}ms)");
}
