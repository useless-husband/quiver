//! Nearest-neighbour indexes.
//!
//! * [`flat`]: exact brute-force search, the baseline and the ground truth.
//! * [`graph`]: the HNSW graph itself (topology only, generic over how
//!   distances are computed), in a concurrent mutable form and a frozen form
//!   that reads directly from a memory-mapped segment file.
//! * [`hnsw`]: [`HnswIndex`], an in-memory index that owns its vectors.

pub mod bitset;
pub mod flat;
pub mod graph;
pub mod hnsw;
mod lock;
pub mod visited;

pub use flat::FlatIndex;
pub use graph::{Graph, GraphParams};
pub use hnsw::HnswIndex;

use std::cmp::Ordering;

/// A search result: an internal row id and its distance (smaller is closer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

impl Neighbor {
    #[inline]
    pub fn new(id: u32, dist: f32) -> Neighbor {
        Neighbor { id, dist }
    }
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    /// Orders by distance (NaN sorts last via `total_cmp`), then by id so that
    /// ties are deterministic.
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// How a search computes the distance from a prepared query to a stored row.
///
/// Implementations exist for full-precision vectors (in RAM or memory-mapped),
/// int8 codes and product-quantization codes, which is what lets the same graph
/// be traversed with compressed vectors and re-ranked with exact ones.
pub trait Space: Sync {
    type Query: ?Sized;
    /// Distance from the query to row `id`. Only called with ids that are
    /// reachable in the graph or explicitly enumerated as live rows.
    fn dist(&self, q: &Self::Query, id: u32) -> f32;
    /// Hint that row `id` will be read soon.
    #[inline]
    fn prefetch(&self, _id: u32) {}
}

/// A space that can also measure the distance between two stored rows; graph
/// construction needs this for the neighbour-selection heuristic.
pub trait BuildSpace: Space {
    fn between(&self, a: u32, b: u32) -> f32;
}

/// Keeps the `k` smallest neighbours seen so far.
pub struct TopK {
    k: usize,
    heap: std::collections::BinaryHeap<Neighbor>,
}

impl TopK {
    pub fn new(k: usize) -> TopK {
        TopK { k, heap: std::collections::BinaryHeap::with_capacity(k + 1) }
    }

    /// The distance a candidate must beat to enter, or +inf while not full.
    #[inline]
    pub fn bound(&self) -> f32 {
        if self.heap.len() < self.k { f32::INFINITY } else { self.heap.peek().map_or(f32::INFINITY, |n| n.dist) }
    }

    #[inline]
    pub fn push(&mut self, n: Neighbor) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(n);
        } else if let Some(top) = self.heap.peek()
            && n < *top
        {
            self.heap.pop();
            self.heap.push(n);
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// The kept neighbours, closest first.
    pub fn into_sorted(self) -> Vec<Neighbor> {
        self.heap.into_sorted_vec()
    }
}

/// Fraction of the true top-k ids that appear in `got` (recall@k).
pub fn recall(got: &[u32], truth: &[u32], k: usize) -> f64 {
    let truth = &truth[..k.min(truth.len())];
    if truth.is_empty() {
        return 1.0;
    }
    let hits = got.iter().take(k).filter(|id| truth.contains(id)).count();
    hits as f64 / truth.len() as f64
}

/// Issues a software prefetch for the cache line holding `p`.
#[inline(always)]
pub(crate) fn prefetch_read<T>(p: *const T) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM is a hint; it never faults, even on an invalid address.
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above, PREFETCHT0 never faults.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}
