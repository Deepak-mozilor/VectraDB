// Batch distance computation shader for VectraDB.
//
// Each workgroup thread computes the distance between the query vector
// and one candidate vector from the data buffer.

struct Params {
    dim: u32,
    n:   u32,
}

@group(0) @binding(0) var<uniform>           params:  Params;
@group(0) @binding(1) var<storage, read>     query:   array<f32>;
@group(0) @binding(2) var<storage, read>     data:    array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// L2 (Euclidean) distance
@compute @workgroup_size(64)
fn distance_l2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n {
        return;
    }
    let base = idx * params.dim;
    var sum: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let diff = query[d] - data[base + d];
        sum = sum + diff * diff;
    }
    output[idx] = sqrt(sum);
}

// Cosine distance: 1 - (a·b / (|a|·|b|))
@compute @workgroup_size(64)
fn distance_cosine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n {
        return;
    }
    let base = idx * params.dim;
    var dot_ab: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let a = query[d];
        let b = data[base + d];
        dot_ab = dot_ab + a * b;
        norm_a = norm_a + a * a;
        norm_b = norm_b + b * b;
    }
    let denom = sqrt(norm_a) * sqrt(norm_b);
    if denom == 0.0 {
        output[idx] = 1.0;
    } else {
        output[idx] = 1.0 - dot_ab / denom;
    }
}

// Negative dot product (so smaller = more similar)
@compute @workgroup_size(64)
fn distance_dot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n {
        return;
    }
    let base = idx * params.dim;
    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        dot = dot + query[d] * data[base + d];
    }
    output[idx] = -dot;
}
