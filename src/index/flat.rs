//! Exact brute-force search: the baseline every approximate result is
//! measured against, and the strategy filtered search falls back to when a
//! filter matches few rows.

use super::{Neighbor, Space, TopK};
use crate::distance::Metric;
use crate::index::hnsw::F32Space;

/// The `k` closest rows among `ids`, exactly.
pub fn exact_knn<S: Space>(
    s: &S,
    q: &S::Query,
    ids: impl Iterator<Item = u32>,
    k: usize,
) -> Vec<Neighbor> {
    let mut top = TopK::new(k);
    for id in ids {
        let d = s.dist(q, id);
        if d < top.bound() {
            top.push(Neighbor::new(id, d));
        }
    }
    top.into_sorted()
}

/// A flat index: a contiguous array of prepared vectors.
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: Metric) -> FlatIndex {
        assert!(dim > 0);
        FlatIndex {
            dim,
            metric,
            data: Vec::new(),
        }
    }

    /// Builds from `rows * dim` raw vectors.
    pub fn from_vectors(dim: usize, metric: Metric, data: &[f32]) -> FlatIndex {
        let mut f = FlatIndex::new(dim, metric);
        for v in data.chunks_exact(dim) {
            f.add(v);
        }
        f
    }

    pub fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.len() as u32;
        self.data.extend_from_slice(&self.metric.prepare(v));
        id
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn space(&self) -> F32Space<'_> {
        F32Space::over(&self.data, self.dim, self.metric)
    }

    pub fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        self.search_with(q, k, |_| true)
    }

    pub fn search_with(&self, q: &[f32], k: usize, accept: impl Fn(u32) -> bool) -> Vec<Neighbor> {
        if self.is_empty() {
            return Vec::new();
        }
        let q = self.metric.prepare(q);
        exact_knn(
            &self.space(),
            &q[..],
            (0..self.len() as u32).filter(|&i| accept(i)),
            k,
        )
    }
}

/// Exact top-`k` ids for every query, computed with `threads` threads.
pub fn ground_truth(
    metric: Metric,
    dim: usize,
    base: &[f32],
    queries: &[f32],
    k: usize,
    threads: usize,
) -> Vec<Vec<u32>> {
    let flat = FlatIndex::from_vectors(dim, metric, base);
    let nq = queries.len() / dim;
    let mut out = vec![Vec::new(); nq];
    let threads = threads.max(1);
    let chunk = nq.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, slot) in out.chunks_mut(chunk).enumerate() {
            let flat = &flat;
            s.spawn(move || {
                for (j, o) in slot.iter_mut().enumerate() {
                    let qi = ci * chunk + j;
                    let q = &queries[qi * dim..(qi + 1) * dim];
                    *o = flat.search(q, k).into_iter().map(|n| n.id).collect();
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn flat_search_is_exact_and_sorted() {
        let mut rng = Rng::new(1);
        let dim = 7;
        let data: Vec<f32> = (0..500 * dim).map(|_| rng.normal()).collect();
        for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
            let f = FlatIndex::from_vectors(dim, metric, &data);
            let q: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
            let got = f.search(&q, 10);
            let mut all: Vec<Neighbor> = (0..500)
                .map(|i| Neighbor::new(i, metric.distance(&q, &data[i as usize * dim..][..dim])))
                .collect();
            all.sort();
            let want: Vec<u32> = all[..10].iter().map(|n| n.id).collect();
            let got_ids: Vec<u32> = got.iter().map(|n| n.id).collect();
            assert_eq!(got_ids, want, "{metric:?}");
            assert!(got.windows(2).all(|w| w[0].dist <= w[1].dist));
            let odd = f.search_with(&q, 5, |i| i % 2 == 1);
            assert!(odd.iter().all(|n| n.id % 2 == 1) && odd.len() == 5);
        }
        assert!(
            FlatIndex::new(3, Metric::L2)
                .search(&[0.0; 3], 4)
                .is_empty()
        );
    }
}
