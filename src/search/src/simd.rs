//! SIMD-accelerated distance functions.
//!
//! Provides hardware-accelerated L2 distance, dot product, and cosine distance
//! using platform-specific SIMD intrinsics with automatic fallback.
//!
//! Performance hierarchy (fastest to slowest):
//! 1. AVX2 (x86_64) — processes 8 floats per cycle
//! 2. SSE (x86_64) — processes 4 floats per cycle
//! 3. NEON (aarch64/ARM) — processes 4 floats per cycle
//! 4. Scalar fallback — 4-wide loop unrolling
//!
//! The best available implementation is selected at compile time via `cfg` attributes.

// ============================================================
// Public API — auto-dispatches to best available impl
// ============================================================

/// L2 (Euclidean) distance between two f32 slices.
#[inline]
pub fn simd_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    simd_l2_squared(a, b).sqrt()
}

/// Squared L2 distance (avoids sqrt for comparisons).
#[inline]
pub fn simd_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_l2_squared(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { sse_l2_squared(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_l2_squared(a, b) };
    }
    #[allow(unreachable_code)]
    scalar_l2_squared(a, b)
}

/// Dot product of two f32 slices.
#[inline]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_dot(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { sse_dot(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_dot(a, b) };
    }
    #[allow(unreachable_code)]
    scalar_dot(a, b)
}

/// Cosine distance: 1.0 - cosine_similarity.
#[inline]
pub fn simd_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_cosine_distance(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { sse_cosine_distance(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_cosine_distance(a, b) };
    }
    #[allow(unreachable_code)]
    scalar_cosine_distance(a, b)
}

// ============================================================
// Scalar fallback (4-wide unrolled)
// ============================================================

#[inline]
fn scalar_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        s0 += d * d;
    }
    s0 + s1 + s2 + s3
}

#[inline]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        s0 += a[j] * b[j];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
    }
    for i in (chunks * 4)..n {
        s0 += a[i] * b[i];
    }
    s0 + s1 + s2 + s3
}

#[inline]
fn scalar_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut dot: f32 = 0.0;
    let mut na: f32 = 0.0;
    let mut nb: f32 = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

// ============================================================
// x86_64 AVX2 (8-wide, 256-bit)
// ============================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let chunks = n / 8;

    for i in 0..chunks {
        let j = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(j));
        let vb = _mm256_loadu_ps(b.as_ptr().add(j));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum of 8 floats
    let mut result = hsum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..n {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let chunks = n / 8;

    for i in 0..chunks {
        let j = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(j));
        let vb = _mm256_loadu_ps(b.as_ptr().add(j));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = hsum_avx2(sum);
    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut vdot = _mm256_setzero_ps();
    let mut vna = _mm256_setzero_ps();
    let mut vnb = _mm256_setzero_ps();
    let chunks = n / 8;

    for i in 0..chunks {
        let j = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(j));
        let vb = _mm256_loadu_ps(b.as_ptr().add(j));
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna = _mm256_fmadd_ps(va, va, vna);
        vnb = _mm256_fmadd_ps(vb, vb, vnb);
    }

    let mut dot = hsum_avx2(vdot);
    let mut na = hsum_avx2(vna);
    let mut nb = hsum_avx2(vnb);

    for i in (chunks * 8)..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

/// Horizontal sum of __m256 (8 floats → 1 float).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

// ============================================================
// x86_64 SSE (4-wide, 128-bit)
// ============================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn sse_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut sum = _mm_setzero_ps();
    let chunks = n / 4;

    for i in 0..chunks {
        let j = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(j));
        let vb = _mm_loadu_ps(b.as_ptr().add(j));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    let mut result = hsum_sse(sum);
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn sse_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut sum = _mm_setzero_ps();
    let chunks = n / 4;

    for i in 0..chunks {
        let j = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(j));
        let vb = _mm_loadu_ps(b.as_ptr().add(j));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }

    let mut result = hsum_sse(sum);
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn sse_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut vdot = _mm_setzero_ps();
    let mut vna = _mm_setzero_ps();
    let mut vnb = _mm_setzero_ps();
    let chunks = n / 4;

    for i in 0..chunks {
        let j = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(j));
        let vb = _mm_loadu_ps(b.as_ptr().add(j));
        vdot = _mm_add_ps(vdot, _mm_mul_ps(va, vb));
        vna = _mm_add_ps(vna, _mm_mul_ps(va, va));
        vnb = _mm_add_ps(vnb, _mm_mul_ps(vb, vb));
    }

    let mut dot = hsum_sse(vdot);
    let mut na = hsum_sse(vna);
    let mut nb = hsum_sse(vnb);

    for i in (chunks * 4)..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

/// Horizontal sum of __m128 (4 floats → 1 float).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn hsum_sse(v: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;
    let shuf = _mm_movehdup_ps(v);
    let sum = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(sum, sum);
    let sum2 = _mm_add_ss(sum, shuf2);
    _mm_cvtss_f32(sum2)
}

// ============================================================
// aarch64 NEON (4-wide, 128-bit)
// ============================================================

#[cfg(target_arch = "aarch64")]
unsafe fn neon_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let chunks = n / 4;

    for i in 0..chunks {
        let j = i * 4;
        let va = vld1q_f32(a.as_ptr().add(j));
        let vb = vld1q_f32(b.as_ptr().add(j));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }

    let mut result = vaddvq_f32(sum);
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let chunks = n / 4;

    for i in 0..chunks {
        let j = i * 4;
        let va = vld1q_f32(a.as_ptr().add(j));
        let vb = vld1q_f32(b.as_ptr().add(j));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }
    result
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut vdot = vdupq_n_f32(0.0);
    let mut vna = vdupq_n_f32(0.0);
    let mut vnb = vdupq_n_f32(0.0);
    let chunks = n / 4;

    for i in 0..chunks {
        let j = i * 4;
        let va = vld1q_f32(a.as_ptr().add(j));
        let vb = vld1q_f32(b.as_ptr().add(j));
        vdot = vfmaq_f32(vdot, va, vb);
        vna = vfmaq_f32(vna, va, va);
        vnb = vfmaq_f32(vnb, vb, vb);
    }

    let mut dot = vaddvq_f32(vdot);
    let mut na = vaddvq_f32(vna);
    let mut nb = vaddvq_f32(vnb);

    for i in (chunks * 4)..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, eps: f32) {
        assert!(
            (a - b).abs() < eps,
            "expected {b}, got {a}, diff={}",
            (a - b).abs()
        );
    }

    #[test]
    fn test_l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_close(simd_l2_distance(&a, &b), 5.0, 1e-5);
    }

    #[test]
    fn test_l2_squared_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (3^2 + 3^2 + 3^2) = 27
        assert_close(simd_l2_squared(&a, &b), 27.0, 1e-5);
    }

    #[test]
    fn test_dot_product_known() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        // 5 + 12 + 21 + 32 = 70
        assert_close(simd_dot(&a, &b), 70.0, 1e-5);
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_close(simd_cosine_distance(&a, &a), 0.0, 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert_close(simd_cosine_distance(&a, &b), 1.0, 1e-5);
    }

    #[test]
    fn test_high_dimensional() {
        // 384-dim vectors (common embedding size)
        let a: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01 + 0.001).collect();

        let l2 = simd_l2_distance(&a, &b);
        let dot = simd_dot(&a, &b);
        let cos = simd_cosine_distance(&a, &b);

        // Compare against scalar reference
        let ref_l2 = scalar_l2_squared(&a, &b).sqrt();
        let ref_dot = scalar_dot(&a, &b);
        let ref_cos = scalar_cosine_distance(&a, &b);

        assert_close(l2, ref_l2, 1e-3);
        assert_close(dot, ref_dot, 1e-1); // dot product of large vectors has larger absolute error
        assert_close(cos, ref_cos, 1e-5);
    }

    #[test]
    fn test_non_aligned_lengths() {
        // Test with lengths that aren't multiples of 4 or 8
        for len in [1, 2, 3, 5, 7, 9, 13, 17, 31, 33, 63, 65, 127, 129] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();

            let l2 = simd_l2_squared(&a, &b);
            let ref_l2 = scalar_l2_squared(&a, &b);
            assert_close(l2, ref_l2, 1e-3);
        }
    }

    #[test]
    fn test_zero_vectors() {
        let a = vec![0.0; 64];
        let b = vec![0.0; 64];
        assert_close(simd_l2_distance(&a, &b), 0.0, 1e-6);
        assert_close(simd_dot(&a, &b), 0.0, 1e-6);
        assert_close(simd_cosine_distance(&a, &b), 1.0, 1e-6); // undefined, returns 1
    }
}
