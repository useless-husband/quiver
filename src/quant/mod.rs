//! Vector compression for graph traversal, with exact re-ranking.
//!
//! * [`sq8`]: int8 scalar quantization (4x smaller), distances by `SDOT`.
//! * [`pq`]: product quantization (`dim * 4 / m` times smaller), distances by
//!   table lookup.
//!
//! A quantized search walks the graph with compressed distances, keeps
//! `k * rerank` candidates, and re-ranks them with the full-precision vectors
//! (memory-mapped in a sealed segment), returning the best `k`.

pub mod kmeans;
pub mod pq;
pub mod sq8;

use serde::{Deserialize, Serialize};

/// How a sealed segment stores the vectors it traverses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Quantization {
    /// Full-precision f32 only.
    #[default]
    None,
    /// int8 codes with per-vector scale.
    Int8,
    /// Product quantization with `m` one-byte sub-codes.
    Pq { m: usize },
}

impl Quantization {
    pub fn parse(s: &str) -> Option<Quantization> {
        match s {
            "none" | "f32" => Some(Quantization::None),
            "int8" | "sq8" => Some(Quantization::Int8),
            _ => s
                .strip_prefix("pq")
                .and_then(|m| m.trim_start_matches(':').parse().ok())
                .map(|m| Quantization::Pq { m }),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Quantization::None => "f32".into(),
            Quantization::Int8 => "int8".into(),
            Quantization::Pq { m } => format!("pq{m}"),
        }
    }
}

use crate::distance::Metric;
use crate::index::graph::{Scratch, Topology, knn};
use crate::index::{Neighbor, Space};

/// Compressed copies of a set of rows, aligned with the rows' ids.
pub enum Codes {
    Int8(sq8::Sq8),
    Pq(pq::Pq),
}

impl Codes {
    /// Compresses `rows * dim` prepared vectors.
    pub fn build(
        q: Quantization,
        data: &[f32],
        dim: usize,
        metric: Metric,
        seed: u64,
        threads: usize,
    ) -> Result<Option<Codes>, String> {
        Ok(match q {
            Quantization::None => None,
            Quantization::Int8 => Some(Codes::Int8(sq8::Sq8::build(data, dim, metric))),
            Quantization::Pq { m } => Some(Codes::Pq(pq::Pq::build(data, dim, m, metric, seed, threads)?)),
        })
    }

    pub fn kind(&self) -> Quantization {
        match self {
            Codes::Int8(_) => Quantization::Int8,
            Codes::Pq(p) => Quantization::Pq { m: p.m() },
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Codes::Int8(s) => s.rows(),
            Codes::Pq(p) => p.rows(),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Codes::Int8(s) => s.bytes(),
            Codes::Pq(p) => p.bytes(),
        }
    }

    /// Traverses `topo` with compressed distances, keeping `max(k * rerank,
    /// k)` candidates, then re-ranks them with `exact` (full precision) and
    /// returns the best `k`. With `rerank == 0` the compressed distances are
    /// returned as they are.
    #[allow(clippy::too_many_arguments)]
    pub fn search<T, E, F>(
        &self,
        topo: &T,
        exact: &E,
        q: &[f32],
        k: usize,
        ef: usize,
        rerank: usize,
        accept: &F,
        sc: &mut Scratch,
    ) -> Vec<Neighbor>
    where
        T: Topology,
        E: Space<Query = [f32]>,
        F: Fn(u32) -> bool,
    {
        let keep = (k * rerank).max(k);
        let cands = match self {
            Codes::Int8(s) => knn(topo, &s.space(), &s.query(q), keep, ef.max(keep), accept, sc),
            Codes::Pq(p) => knn(topo, &p.space(), &p.query(q), keep, ef.max(keep), accept, sc),
        };
        if rerank == 0 {
            return cands;
        }
        let mut r: Vec<Neighbor> = cands.iter().map(|c| Neighbor::new(c.id, exact.dist(q, c.id))).collect();
        r.sort_unstable();
        r.truncate(k);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_labels() {
        assert_eq!(Quantization::parse("int8"), Some(Quantization::Int8));
        assert_eq!(Quantization::parse("pq32"), Some(Quantization::Pq { m: 32 }));
        assert_eq!(Quantization::parse("pq:8"), Some(Quantization::Pq { m: 8 }));
        assert_eq!(Quantization::parse("pqx"), None);
        assert_eq!(Quantization::Pq { m: 16 }.label(), "pq16");
        let j = serde_json::to_string(&Quantization::Pq { m: 4 }).unwrap();
        assert_eq!(j, r#"{"type":"pq","m":4}"#);
    }
}
