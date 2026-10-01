//! int8 scalar quantization with per-vector scaling.
//!
//! Each (prepared, and for L2 mean-centred) vector `x` is stored as
//! `x ≈ s · q` with `q ∈ [-127, 127]^d` and `s = max|x_i| / 127`. The query is
//! quantized the same way, so every distance is one integer dot product,
//! which NEON computes with `SDOT` (16 multiply-accumulates per instruction):
//!
//! * inner product: `x·y ≈ s_x s_y (q_x·q_y)`,
//! * L2: `|x̂ - ŷ|² = s_x²|q_x|² + s_y²|q_y|² - 2 s_x s_y (q_x·q_y)`, using
//!   the norms of the *dequantized* vectors so the estimate is exactly the
//!   distance between the two reconstructions (never negative).
//!
//! Centring on the dataset mean is applied for L2 only (it does not change L2
//! distances and spends the 8 bits on the spread rather than the offset).
//! For inner product it would change the ranking, so it is not done.

use crate::distance::{Kernels, Metric, kernels};
use crate::index::Space;

pub struct Sq8 {
    dim: usize,
    metric: Metric,
    center: Vec<f32>,
    codes: Vec<i8>,
    scale: Vec<f32>,
    qnorm: Vec<f32>,
}

/// A query quantized for [`Sq8`].
pub struct Sq8Query {
    q: Vec<i8>,
    s: f32,
    n: f32,
}

fn quantize(x: &[f32], center: &[f32], out: &mut [i8]) -> (f32, f32) {
    let mut max = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let c = v - center.get(i).copied().unwrap_or(0.0);
        max = max.max(c.abs());
    }
    if max == 0.0 || !max.is_finite() {
        out.fill(0);
        return (0.0, 0.0);
    }
    let s = max / 127.0;
    let inv = 1.0 / s;
    let mut n = 0i64;
    for (i, &v) in x.iter().enumerate() {
        let c = v - center.get(i).copied().unwrap_or(0.0);
        let q = (c * inv).round().clamp(-127.0, 127.0) as i8;
        out[i] = q;
        n += q as i64 * q as i64;
    }
    (s, s * s * n as f32)
}

impl Sq8 {
    /// Quantizes `rows * dim` prepared vectors.
    pub fn build(data: &[f32], dim: usize, metric: Metric) -> Sq8 {
        let rows = data.len() / dim;
        let mut center = Vec::new();
        if metric == Metric::L2 && rows > 0 {
            let mut sum = vec![0.0f64; dim];
            for v in data.chunks_exact(dim) {
                for (s, &x) in sum.iter_mut().zip(v) {
                    *s += x as f64;
                }
            }
            center = sum.iter().map(|&s| (s / rows as f64) as f32).collect();
        }
        let mut sq = Sq8 {
            dim,
            metric,
            center,
            codes: vec![0; rows * dim],
            scale: vec![0.0; rows],
            qnorm: vec![0.0; rows],
        };
        for (i, v) in data.chunks_exact(dim).enumerate() {
            let (s, n) = quantize(v, &sq.center, &mut sq.codes[i * dim..(i + 1) * dim]);
            sq.scale[i] = s;
            sq.qnorm[i] = n;
        }
        sq
    }

    /// Reassembles from stored parts (a segment file).
    pub fn from_parts(
        dim: usize,
        metric: Metric,
        center: Vec<f32>,
        codes: Vec<i8>,
        scale: Vec<f32>,
    ) -> Option<Sq8> {
        let rows = scale.len();
        if codes.len() != rows * dim || !(center.is_empty() || center.len() == dim) {
            return None;
        }
        let qnorm = (0..rows)
            .map(|i| {
                let q = &codes[i * dim..(i + 1) * dim];
                let n: i64 = q.iter().map(|&x| x as i64 * x as i64).sum();
                scale[i] * scale[i] * n as f32
            })
            .collect();
        Some(Sq8 {
            dim,
            metric,
            center,
            codes,
            scale,
            qnorm,
        })
    }

    pub fn rows(&self) -> usize {
        self.scale.len()
    }

    pub fn center(&self) -> &[f32] {
        &self.center
    }

    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    pub fn scales(&self) -> &[f32] {
        &self.scale
    }

    pub fn bytes(&self) -> usize {
        self.codes.len() + 8 * self.scale.len() + 4 * self.center.len()
    }

    /// Quantizes a prepared query.
    pub fn query(&self, q: &[f32]) -> Sq8Query {
        let mut out = vec![0i8; self.dim];
        let (s, n) = quantize(q, &self.center, &mut out);
        Sq8Query { q: out, s, n }
    }

    /// The reconstruction of row `id` (for tests and error measurement).
    pub fn decode(&self, id: u32) -> Vec<f32> {
        let i = id as usize;
        (0..self.dim)
            .map(|d| {
                self.codes[i * self.dim + d] as f32 * self.scale[i]
                    + self.center.get(d).copied().unwrap_or(0.0)
            })
            .collect()
    }

    pub fn space(&self) -> Sq8Space<'_> {
        Sq8Space {
            sq: self,
            k: kernels(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Sq8Space<'a> {
    sq: &'a Sq8,
    k: &'static Kernels,
}

impl Space for Sq8Space<'_> {
    type Query = Sq8Query;

    #[inline]
    fn dist(&self, q: &Sq8Query, id: u32) -> f32 {
        let i = id as usize;
        let d = self.sq.dim;
        let code = &self.sq.codes[i * d..(i + 1) * d];
        let dot = (self.k.dot_i8)(&q.q, code) as f32 * q.s * self.sq.scale[i];
        match self.sq.metric {
            Metric::L2 => (q.n + self.sq.qnorm[i] - 2.0 * dot).max(0.0),
            Metric::Ip => -dot,
            Metric::Cosine => 1.0 - dot,
        }
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        if let Some(p) = self.sq.codes.get(id as usize * self.sq.dim) {
            crate::index::prefetch_read(p as *const i8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn distances_track_full_precision() {
        let mut rng = Rng::new(8);
        let dim = 64;
        for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
            let raw: Vec<f32> = (0..500 * dim).map(|_| rng.normal() + 3.0).collect();
            let data: Vec<f32> = raw.chunks(dim).flat_map(|v| metric.prepare(v)).collect();
            let sq = Sq8::build(&data, dim, metric);
            let sp = sq.space();
            let k = kernels();
            let mut worst = 0.0f32;
            for _ in 0..200 {
                let q = metric.prepare(&(0..dim).map(|_| rng.normal() + 3.0).collect::<Vec<_>>());
                let qq = sq.query(&q);
                let id = rng.below(500) as u32;
                let exact = metric.prepared_distance(k, &q, &data[id as usize * dim..][..dim]);
                let approx = sp.dist(&qq, id);
                let scale = match metric {
                    Metric::L2 => exact.abs().max(1.0),
                    _ => 1.0f32.max(exact.abs()),
                };
                worst = worst.max((approx - exact).abs() / scale);
            }
            assert!(worst < 0.02, "{metric:?}: worst relative error {worst}");
            // Reconstructions are within half a quantization step.
            let v = &data[..dim];
            let dec = sq.decode(0);
            let step = sq.scales()[0];
            assert!(
                v.iter()
                    .zip(&dec)
                    .all(|(a, b)| (a - b).abs() <= 0.5 * step + 1e-5)
            );
        }
    }

    #[test]
    fn zero_vectors_and_round_trip() {
        let data = vec![0.0f32; 8 * 3];
        let sq = Sq8::build(&data, 8, Metric::Ip);
        let q = sq.query(&[0.0; 8]);
        assert_eq!(sq.space().dist(&q, 1), 0.0);
        let parts = Sq8::from_parts(
            8,
            Metric::Ip,
            sq.center().to_vec(),
            sq.codes().to_vec(),
            sq.scales().to_vec(),
        )
        .unwrap();
        assert_eq!(parts.rows(), 3);
        assert!(Sq8::from_parts(8, Metric::Ip, vec![], vec![0; 5], vec![1.0]).is_none());
    }
}
