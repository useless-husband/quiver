//! x86-64 AVX2 + FMA kernels (selected at run time when the CPU has both).
//!
//! This path exists so that the index is fast on the Linux CI runners and on
//! x86 servers; it is deliberately smaller than the NEON module (no int8 L2,
//! Hamming uses the scalar `popcnt` loop).

use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: loads read 8 floats at i..i+8 with i + 8 <= n.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            i += 16;
        }
        if i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc0 = _mm256_fmadd_ps(d, d, acc0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            s += d * d;
            i += 1;
        }
        s
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: as above.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            i += 16;
        }
        if i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        s
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: as above.
    unsafe {
        let mut ab = _mm256_setzero_ps();
        let mut aa = _mm256_setzero_ps();
        let mut bb = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(pa.add(i));
            let y = _mm256_loadu_ps(pb.add(i));
            ab = _mm256_fmadd_ps(x, y, ab);
            aa = _mm256_fmadd_ps(x, x, aa);
            bb = _mm256_fmadd_ps(y, y, bb);
            i += 8;
        }
        let (mut sab, mut saa, mut sbb) = (hsum256(ab), hsum256(aa), hsum256(bb));
        while i < n {
            let (x, y) = (*pa.add(i), *pb.add(i));
            sab += x * y;
            saa += x * x;
            sbb += y * y;
            i += 1;
        }
        super::cosine_from_parts(sab, saa, sbb)
    }
}

/// int8 inner product: sign-extend to i16 and use `vpmaddwd`.
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut total = 0i64;
    let mut start = 0;
    while start < n {
        // Each i32 lane gains at most 2 * 16384 per 16 elements.
        let end = (start + (1 << 16)).min(n);
        let mut i = start;
        // SAFETY: loads read 16 bytes at i..i+16 with i + 16 <= end <= n.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            while i + 16 <= end {
                let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
                let y = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
                i += 16;
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
            total += lanes.iter().map(|&v| v as i64).sum::<i64>();
            while i < end {
                total += *pa.add(i) as i64 * *pb.add(i) as i64;
                i += 1;
            }
        }
        start = end;
    }
    total
}
