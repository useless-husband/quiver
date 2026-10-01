//! Product quantization (Jégou, Douze & Schmid, TPAMI 2011).
//!
//! The vector is split into `m` sub-vectors of `dim / m` dimensions, and each
//! sub-vector is replaced by the index of its nearest centroid in a
//! 256-entry codebook trained with k-means, so a vector costs `m` bytes.
//! Search uses asymmetric distance computation (ADC): for each query, an
//! `m x 256` table of sub-distances is computed once, after which the distance
//! to any row is `m` table lookups.
//!
//! For cosine the vectors are normalised first and the tables hold negative
//! sub-dot-products, so the ADC sum plus one is `1 - dot`.

use super::kmeans::kmeans;
use crate::distance::{Metric, kernels};
use crate::index::Space;

pub const KS: usize = 256;

pub struct Pq {
    dim: usize,
    m: usize,
    dsub: usize,
    metric: Metric,
    /// `m * KS * dsub` floats: centroid `c` of sub-space `j` starts at
    /// `(j * KS + c) * dsub`.
    centroids: Vec<f32>,
    codes: Vec<u8>,
}

/// The per-query ADC table.
pub struct PqQuery {
    table: Vec<f32>,
}

impl Pq {
    /// Trains codebooks on (a sample of) `data` and encodes every row.
    pub fn build(data: &[f32], dim: usize, m: usize, metric: Metric, seed: u64, threads: usize) -> Result<Pq, String> {
        if m == 0 || dim % m != 0 {
            return Err(format!("pq: m={m} must divide the dimension {dim}"));
        }
        let rows = data.len() / dim;
        let dsub = dim / m;
        // 256 centroids need far fewer than all rows; FAISS warns below 39 per
        // centroid. Cap the training set to bound build time.
        let sample_rows = rows.min(KS * 100);
        let step = (rows / sample_rows.max(1)).max(1);
        let mut centroids = vec![0.0f32; m * KS * dsub];
        let mut sub = Vec::with_capacity(sample_rows * dsub);
        for j in 0..m {
            sub.clear();
            for i in (0..rows).step_by(step).take(sample_rows) {
                sub.extend_from_slice(&data[i * dim + j * dsub..i * dim + (j + 1) * dsub]);
            }
            if sub.is_empty() {
                break;
            }
            let c = kmeans(&sub, dsub, KS, 20, seed ^ j as u64, threads);
            centroids[j * KS * dsub..(j + 1) * KS * dsub].copy_from_slice(&c);
        }
        let mut pq = Pq { dim, m, dsub, metric, centroids, codes: Vec::new() };
        let mut codes = vec![0u8; rows * m];
        let chunk = rows.div_ceil(threads.max(1)).max(1);
        let pq_ref = &pq;
        std::thread::scope(|s| {
            for (ci, slot) in codes.chunks_mut(chunk * m).enumerate() {
                s.spawn(move || {
                    for (r, code) in slot.chunks_mut(m).enumerate() {
                        let i = ci * chunk + r;
                        pq_ref.encode_into(&data[i * dim..(i + 1) * dim], code);
                    }
                });
            }
        });
        pq.codes = codes;
        Ok(pq)
    }

    pub fn from_parts(dim: usize, m: usize, metric: Metric, centroids: Vec<f32>, codes: Vec<u8>) -> Option<Pq> {
        if m == 0 || dim % m != 0 || centroids.len() != m * KS * (dim / m) || codes.len() % m != 0 {
            return None;
        }
        Some(Pq { dim, m, dsub: dim / m, metric, centroids, codes })
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn rows(&self) -> usize {
        self.codes.len() / self.m
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn bytes(&self) -> usize {
        self.codes.len() + 4 * self.centroids.len()
    }

    fn centroid(&self, j: usize, c: usize) -> &[f32] {
        let s = (j * KS + c) * self.dsub;
        &self.centroids[s..s + self.dsub]
    }

    /// Nearest centroid per sub-space (always by L2: that is what the
    /// codebooks were trained to minimise).
    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        let k = kernels();
        for j in 0..self.m {
            let x = &v[j * self.dsub..(j + 1) * self.dsub];
            let mut best = (f32::INFINITY, 0u8);
            for c in 0..KS {
                let d = (k.l2sq_f32)(x, self.centroid(j, c));
                if d < best.0 {
                    best = (d, c as u8);
                }
            }
            out[j] = best.1;
        }
    }

    pub fn decode(&self, id: u32) -> Vec<f32> {
        let code = &self.codes[id as usize * self.m..(id as usize + 1) * self.m];
        code.iter().enumerate().flat_map(|(j, &c)| self.centroid(j, c as usize).to_vec()).collect()
    }

    /// Builds the ADC table for a prepared query.
    pub fn query(&self, q: &[f32]) -> PqQuery {
        let k = kernels();
        let mut table = vec![0.0f32; self.m * KS];
        for j in 0..self.m {
            let x = &q[j * self.dsub..(j + 1) * self.dsub];
            for c in 0..KS {
                table[j * KS + c] = match self.metric {
                    Metric::L2 => (k.l2sq_f32)(x, self.centroid(j, c)),
                    Metric::Ip | Metric::Cosine => -(k.dot_f32)(x, self.centroid(j, c)),
                };
            }
        }
        PqQuery { table }
    }

    pub fn space(&self) -> PqSpace<'_> {
        PqSpace { pq: self }
    }
}

#[derive(Clone, Copy)]
pub struct PqSpace<'a> {
    pq: &'a Pq,
}

impl Space for PqSpace<'_> {
    type Query = PqQuery;

    #[inline]
    fn dist(&self, q: &PqQuery, id: u32) -> f32 {
        let m = self.pq.m;
        let code = &self.pq.codes[id as usize * m..(id as usize + 1) * m];
        let t = &q.table[..m * KS];
        // Four independent accumulators hide the load latency of the lookups.
        let (mut a, mut b, mut c, mut d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut j = 0;
        while j + 4 <= m {
            a += t[j * KS + code[j] as usize];
            b += t[(j + 1) * KS + code[j + 1] as usize];
            c += t[(j + 2) * KS + code[j + 2] as usize];
            d += t[(j + 3) * KS + code[j + 3] as usize];
            j += 4;
        }
        while j < m {
            a += t[j * KS + code[j] as usize];
            j += 1;
        }
        let s = (a + b) + (c + d);
        if self.pq.metric == Metric::Cosine { 1.0 + s } else { s }
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        if let Some(p) = self.pq.codes.get(id as usize * self.pq.m) {
            crate::index::prefetch_read(p as *const u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::testkit::clustered;

    #[test]
    fn adc_matches_distance_to_reconstruction() {
        let mut rng = Rng::new(2);
        let dim = 16;
        for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
            let raw = clustered(&mut rng, 3000, dim, 20);
            let data: Vec<f32> = raw.chunks(dim).flat_map(|v| metric.prepare(v)).collect();
            let pq = Pq::build(&data, dim, 4, metric, 1, 2).unwrap();
            assert_eq!(pq.rows(), 3000);
            let sp = pq.space();
            let k = kernels();
            for _ in 0..100 {
                let q = metric.prepare(&(0..dim).map(|_| rng.normal()).collect::<Vec<_>>());
                let id = rng.below(3000) as u32;
                let rec = pq.decode(id);
                let want = metric.prepared_distance(k, &q, &rec);
                let got = sp.dist(&pq.query(&q), id);
                assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "{metric:?}: {got} vs {want}");
            }
            // Quantization error is far below the data spread.
            let mse: f32 =
                (0..3000u32).map(|i| (k.l2sq_f32)(&data[i as usize * dim..][..dim], &pq.decode(i))).sum::<f32>()
                    / 3000.0;
            let var: f32 = data.iter().map(|x| x * x).sum::<f32>() / 3000.0;
            assert!(mse < 0.25 * var, "{metric:?}: mse {mse} vs energy {var}");
        }
        assert!(Pq::build(&[0.0; 30], 10, 3, Metric::L2, 1, 1).is_err());
    }
}
