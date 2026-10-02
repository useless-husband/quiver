//! CRC-32C (Castagnoli), used to checksum WAL records and segment sections.
//!
//! Uses the ARMv8 `CRC32CX` instruction or SSE4.2 `crc32` when available and
//! a byte-wise table otherwise. All three produce identical results (tested).

const POLY: u32 = 0x82F6_3B78; // reflected Castagnoli polynomial

const TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// Table-driven reference implementation.
pub fn crc32c_table(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in data {
        c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_arm(crc: u32, data: &[u8]) -> u32 {
    use core::arch::aarch64::{__crc32cb, __crc32cd};
    let mut c = !crc;
    let (chunks, rest) = data.as_chunks::<8>();
    for ch in chunks {
        c = __crc32cd(c, u64::from_le_bytes(*ch));
    }
    for &b in rest {
        c = __crc32cb(c, b);
    }
    !c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_x86(crc: u32, data: &[u8]) -> u32 {
    use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
    let mut c = !crc as u64;
    let (chunks, rest) = data.as_chunks::<8>();
    for ch in chunks {
        c = _mm_crc32_u64(c, u64::from_le_bytes(*ch));
    }
    let mut c = c as u32;
    for &b in rest {
        c = _mm_crc32_u8(c, b);
    }
    !c
}

/// Extends `crc` (0 for a fresh checksum) with `data`.
pub fn crc32c_append(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        // SAFETY: the CPU supports the CRC extension (checked above).
        return unsafe { crc32c_arm(crc, data) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.2") {
        // SAFETY: the CPU supports SSE4.2 (checked above).
        return unsafe { crc32c_x86(crc, data) };
    }
    crc32c_table(crc, data)
}

pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_append(0, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn known_vectors_and_paths_agree() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c_table(0, b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let n = rng.below(300);
            let data: Vec<u8> = (0..n).map(|_| rng.next_u64() as u8).collect();
            let split = rng.below(n + 1);
            let whole = crc32c(&data);
            assert_eq!(whole, crc32c_table(0, &data));
            assert_eq!(crc32c_append(crc32c(&data[..split]), &data[split..]), whole);
        }
    }
}
