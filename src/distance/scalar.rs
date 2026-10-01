//! Portable reference kernels.
//!
//! These are the definitions every SIMD kernel is tested against, so they are
//! written as the plainest possible loops. The f32 loops accumulate
//! sequentially and are therefore not auto-vectorised (LLVM may not reassociate
//! floating-point sums); the integer loops may be auto-vectorised by LLVM,
//! which is allowed because integer addition is associative. The benchmark
//! report states this when it compares against them.

/// Squared Euclidean distance.
pub fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Inner product.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// Cosine distance `1 - cos(a, b)`, computed in one pass. Zero vectors are at
/// distance 1 from everything.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut ab, mut aa, mut bb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    super::cosine_from_parts(ab, aa, bb)
}

/// Inner product of int8 vectors, exact (accumulated in i64, returned as i64).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    let n = a.len().min(b.len());
    let mut s = 0i64;
    for i in 0..n {
        s += a[i] as i64 * b[i] as i64;
    }
    s
}

/// Squared Euclidean distance of int8 vectors, exact.
pub fn l2sq_i8(a: &[i8], b: &[i8]) -> u64 {
    let n = a.len().min(b.len());
    let mut s = 0u64;
    for i in 0..n {
        let d = (a[i] as i32 - b[i] as i32).unsigned_abs() as u64;
        s += d * d;
    }
    s
}

/// Hamming distance between bit strings stored as u64 words.
pub fn hamming(a: &[u64], b: &[u64]) -> u64 {
    let n = a.len().min(b.len());
    let mut s = 0u64;
    for i in 0..n {
        s += (a[i] ^ b[i]).count_ones() as u64;
    }
    s
}
