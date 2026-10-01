//! A small deterministic PRNG (xoshiro256** seeded through SplitMix64).
//!
//! The index needs randomness for HNSW levels and k-means initialisation, and
//! the tests need reproducible random inputs; neither needs cryptographic
//! quality, and a seedable generator keeps every run reproducible.

/// SplitMix64 step; also used on its own to derive values from a counter.
#[inline]
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut s = [0u64; 4];
        for w in s.iter_mut() {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            *w = splitmix64(x);
        }
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    pub fn f32(&mut self) -> f32 {
        self.f64() as f32
    }

    /// Uniform in `0..n` (`n > 0`), using Lemire's multiply-shift reduction.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    pub fn bool(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_roughly_uniform() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut buckets = [0usize; 10];
        for _ in 0..100_000 {
            buckets[a.below(10)] += 1;
        }
        for &c in &buckets {
            assert!((9_000..11_000).contains(&c), "{buckets:?}");
        }
        let mean: f64 = (0..100_000).map(|_| a.normal() as f64).sum::<f64>() / 100_000.0;
        assert!(mean.abs() < 0.02, "mean {mean}");
    }
}
