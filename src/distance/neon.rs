//! Arm NEON (AdvSIMD) kernels.
//!
//! NEON is part of the AArch64 baseline, so the f32 kernels need no runtime
//! check. The int8 kernels come in two flavours: one using the `SDOT`/`UDOT`
//! instructions (FEAT_DotProd, detected at run time) and one using widening
//! multiplies that works on any AArch64 core.
//!
//! All loads are unaligned (`vld1q_*` has no alignment requirement) and the
//! tails are finished with scalar code, so any length and any offset is valid.
//! Lengths are taken as `min(a.len(), b.len())`; callers pass equal lengths.

use core::arch::aarch64::*;

/// Squared Euclidean distance, 16 lanes per iteration in four accumulators.
pub fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 4 floats at index i..i+4 with i + 4 <= n, so it
    // stays inside both slices.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
            i += 16;
        }
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc0 = vfmaq_f32(acc0, d, d);
            i += 4;
        }
        let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            s += d * d;
            i += 1;
        }
        s
    }
}

/// Inner product.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: as in `l2sq_f32`.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        s
    }
}

/// Cosine distance in one pass over both inputs.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: as in `l2sq_f32`.
    unsafe {
        let (mut ab0, mut ab1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut aa0, mut aa1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut bb0, mut bb1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0;
        while i + 8 <= n {
            let x0 = vld1q_f32(pa.add(i));
            let y0 = vld1q_f32(pb.add(i));
            let x1 = vld1q_f32(pa.add(i + 4));
            let y1 = vld1q_f32(pb.add(i + 4));
            ab0 = vfmaq_f32(ab0, x0, y0);
            aa0 = vfmaq_f32(aa0, x0, x0);
            bb0 = vfmaq_f32(bb0, y0, y0);
            ab1 = vfmaq_f32(ab1, x1, y1);
            aa1 = vfmaq_f32(aa1, x1, x1);
            bb1 = vfmaq_f32(bb1, y1, y1);
            i += 8;
        }
        if i + 4 <= n {
            let x = vld1q_f32(pa.add(i));
            let y = vld1q_f32(pb.add(i));
            ab0 = vfmaq_f32(ab0, x, y);
            aa0 = vfmaq_f32(aa0, x, x);
            bb0 = vfmaq_f32(bb0, y, y);
            i += 4;
        }
        let mut ab = vaddvq_f32(vaddq_f32(ab0, ab1));
        let mut aa = vaddvq_f32(vaddq_f32(aa0, aa1));
        let mut bb = vaddvq_f32(vaddq_f32(bb0, bb1));
        while i < n {
            let (x, y) = (*pa.add(i), *pb.add(i));
            ab += x * y;
            aa += x * x;
            bb += y * y;
            i += 1;
        }
        super::cosine_from_parts(ab, aa, bb)
    }
}

/// Elements per block before the 32-bit lane accumulators are widened to 64
/// bits. Chosen so that no lane can overflow for any input values: a lane of
/// the SDOT accumulator gains at most 4 * 128 * 128 = 65536 per 16 elements.
const I8_BLOCK: usize = 1 << 16;

/// int8 inner product with `SDOT` (FEAT_DotProd).
///
/// # Safety
/// The CPU must support FEAT_DotProd.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_i8_dotprod(a: &[i8], b: &[i8]) -> i64 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut total = 0i64;
    let mut start = 0;
    while start < n {
        let end = (start + I8_BLOCK).min(n);
        let mut i = start;
        // SAFETY: loads read 16 bytes at i..i+16 with i + 16 <= end <= n.
        unsafe {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            while i + 32 <= end {
                acc0 = vdotq_s32(acc0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
                acc1 = vdotq_s32(acc1, vld1q_s8(pa.add(i + 16)), vld1q_s8(pb.add(i + 16)));
                i += 32;
            }
            if i + 16 <= end {
                acc0 = vdotq_s32(acc0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
                i += 16;
            }
            total += vaddlvq_s32(acc0) + vaddlvq_s32(acc1);
            while i < end {
                total += *pa.add(i) as i64 * *pb.add(i) as i64;
                i += 1;
            }
        }
        start = end;
    }
    total
}

/// int8 inner product with widening multiplies (any AArch64 core).
pub fn dot_i8_widening(a: &[i8], b: &[i8]) -> i64 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut total = 0i64;
    let mut start = 0;
    while start < n {
        let end = (start + I8_BLOCK).min(n);
        let mut i = start;
        // SAFETY: loads read 16 bytes at i..i+16 with i + 16 <= end <= n.
        // Each i16 product is at most 128 * 128 = 16384, and `vpadalq_s16`
        // adds pairs of them into i32 lanes, so nothing overflows in a block.
        unsafe {
            let mut acc = vdupq_n_s32(0);
            while i + 16 <= end {
                let x = vld1q_s8(pa.add(i));
                let y = vld1q_s8(pb.add(i));
                acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
                acc = vpadalq_s16(acc, vmull_high_s8(x, y));
                i += 16;
            }
            total += vaddlvq_s32(acc);
            while i < end {
                total += *pa.add(i) as i64 * *pb.add(i) as i64;
                i += 1;
            }
        }
        start = end;
    }
    total
}

/// int8 squared Euclidean distance with `UDOT` on the absolute differences.
///
/// # Safety
/// The CPU must support FEAT_DotProd.
#[target_feature(enable = "dotprod")]
pub unsafe fn l2sq_i8_dotprod(a: &[i8], b: &[i8]) -> u64 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut total = 0u64;
    let mut start = 0;
    while start < n {
        // |a - b| <= 255, so a lane gains at most 4 * 255^2 = 260100 per 16
        // elements; 2^16 elements keep each u32 lane below 2^31.
        let end = (start + I8_BLOCK).min(n);
        let mut i = start;
        // SAFETY: loads read 16 bytes at i..i+16 with i + 16 <= end <= n.
        unsafe {
            let mut acc0 = vdupq_n_u32(0);
            let mut acc1 = vdupq_n_u32(0);
            while i + 32 <= end {
                let d0 = vreinterpretq_u8_s8(vabdq_s8(vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i))));
                let d1 = vreinterpretq_u8_s8(vabdq_s8(
                    vld1q_s8(pa.add(i + 16)),
                    vld1q_s8(pb.add(i + 16)),
                ));
                acc0 = vdotq_u32(acc0, d0, d0);
                acc1 = vdotq_u32(acc1, d1, d1);
                i += 32;
            }
            if i + 16 <= end {
                let d = vreinterpretq_u8_s8(vabdq_s8(vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i))));
                acc0 = vdotq_u32(acc0, d, d);
                i += 16;
            }
            total += vaddlvq_u32(acc0) + vaddlvq_u32(acc1);
            while i < end {
                let d = (*pa.add(i) as i32 - *pb.add(i) as i32).unsigned_abs() as u64;
                total += d * d;
                i += 1;
            }
        }
        start = end;
    }
    total
}

/// int8 squared Euclidean distance with widening arithmetic (any AArch64 core).
pub fn l2sq_i8_widening(a: &[i8], b: &[i8]) -> u64 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut total = 0u64;
    let mut start = 0;
    while start < n {
        let end = (start + I8_BLOCK).min(n);
        let mut i = start;
        // SAFETY: loads read 16 bytes at i..i+16 with i + 16 <= end <= n.
        // Differences are formed in i16 (|d| <= 255) and squared into u32
        // lanes; a lane gains at most 4 * 255^2 per 16 elements.
        unsafe {
            let mut acc = vdupq_n_u32(0);
            while i + 16 <= end {
                let x = vld1q_s8(pa.add(i));
                let y = vld1q_s8(pb.add(i));
                let dl = vabdl_s8(vget_low_s8(x), vget_low_s8(y));
                let dh = vabdl_high_s8(x, y);
                let dl = vreinterpretq_u16_s16(dl);
                let dh = vreinterpretq_u16_s16(dh);
                acc = vmlal_u16(acc, vget_low_u16(dl), vget_low_u16(dl));
                acc = vmlal_high_u16(acc, dl, dl);
                acc = vmlal_u16(acc, vget_low_u16(dh), vget_low_u16(dh));
                acc = vmlal_high_u16(acc, dh, dh);
                i += 16;
            }
            total += vaddlvq_u32(acc);
            while i < end {
                let d = (*pa.add(i) as i32 - *pb.add(i) as i32).unsigned_abs() as u64;
                total += d * d;
                i += 1;
            }
        }
        start = end;
    }
    total
}

/// Hamming distance with `CNT` (population count per byte).
pub fn hamming(a: &[u64], b: &[u64]) -> u64 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr() as *const u8, b.as_ptr() as *const u8);
    let mut total = 0u64;
    let mut i = 0; // index in u64 words
    // SAFETY: each iteration loads 16 bytes = words i, i+1 with i + 2 <= n.
    // A u16 lane gains at most 16 per iteration; blocks of 2048 iterations
    // keep it below 2^16.
    unsafe {
        while i + 2 <= n {
            let end = (i + 2 * 2048).min(n);
            let mut acc = vdupq_n_u16(0);
            while i + 2 <= end {
                let x = veorq_u8(vld1q_u8(pa.add(i * 8)), vld1q_u8(pb.add(i * 8)));
                acc = vpadalq_u8(acc, vcntq_u8(x));
                i += 2;
            }
            total += vaddlvq_u16(acc) as u64;
        }
    }
    while i < n {
        total += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    total
}
