//! [`HnswIndex`]: an in-memory HNSW index that owns its vectors.
//!
//! Rows are allocated from a fixed-capacity arena. A row's slot is written
//! exactly once, by the thread that allocated it, before the row's state is
//! set to `WRITTEN` (`Release`) and before the graph links it; every reader
//! reaches a row either through a graph link or after loading its state with
//! `Acquire`, so slot reads never race with the slot's write. That is the
//! whole safety argument for the raw-pointer arena below.
//!
//! Deletes are tombstones: the row stays in the graph as a routing node and is
//! filtered out of results. [`HnswIndex::rebuilt`] produces a fresh index
//! without the tombstoned rows; the collection layer calls it (through
//! compaction) once tombstones exceed a threshold. Repairing the graph in
//! place, as hnswlib's `markDelete` + `replaceDeleted` does, was rejected
//! because a reused slot's old in-links point at a vector that changed, which
//! hurts recall unless every in-neighbour is revisited.

use super::bitset::AtomicBitset;
use super::graph::{self, Graph, GraphParams, GraphStats, Scratch};
use super::{BuildSpace, Neighbor, Space, TopK};
use crate::distance::{Kernels, Metric, kernels};
use crate::error::{Error, Result, invalid};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

const EMPTY: u8 = 0;
const WRITTEN: u8 = 1;
const LINKED: u8 = 2;

/// A zero-initialised f32 buffer whose disjoint slots are written through a
/// shared reference (see the module documentation for the protocol).
struct Arena {
    ptr: NonNull<f32>,
    len: usize,
}

// SAFETY: the arena is a plain buffer of f32; cross-thread access follows the
// write-once / publish-with-Release protocol described above.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    fn zeroed(len: usize) -> Arena {
        let b: Box<[f32]> = vec![0.0f32; len].into_boxed_slice();
        let ptr = NonNull::new(Box::into_raw(b) as *mut f32).expect("non-null box");
        Arena { ptr, len }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` came from `Box::into_raw` of a `[f32]` of `len`.
        unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len))) }
    }
}

/// Full-precision vectors addressed by row id.
#[derive(Clone, Copy)]
pub struct F32Space<'a> {
    base: *const f32,
    dim: usize,
    rows: usize,
    metric: Metric,
    k: &'static Kernels,
    _life: PhantomData<&'a [f32]>,
}

// SAFETY: a read-only view; see the arena protocol.
unsafe impl Sync for F32Space<'_> {}
unsafe impl Send for F32Space<'_> {}

impl<'a> F32Space<'a> {
    /// A space over an immutable slice of `rows * dim` prepared vectors.
    pub fn over(data: &'a [f32], dim: usize, metric: Metric) -> F32Space<'a> {
        assert!(dim > 0 && data.len().is_multiple_of(dim));
        F32Space { base: data.as_ptr(), dim, rows: data.len() / dim, metric, k: kernels(), _life: PhantomData }
    }

    #[inline]
    pub fn row(&self, id: u32) -> &'a [f32] {
        let id = id as usize;
        assert!(id < self.rows, "row {id} out of range {}", self.rows);
        // SAFETY: in bounds (checked above); either the slice is immutable
        // (`over`) or the row was published under the arena protocol.
        unsafe { std::slice::from_raw_parts(self.base.add(id * self.dim), self.dim) }
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }
}

impl Space for F32Space<'_> {
    type Query = [f32];

    #[inline]
    fn dist(&self, q: &[f32], id: u32) -> f32 {
        self.metric.prepared_distance(self.k, q, self.row(id))
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        if (id as usize) < self.rows {
            // Prefetch the whole vector (128-byte lines): one line only covers
            // 32 dimensions, and a high-dimensional search is bound by memory
            // latency, not arithmetic.
            // SAFETY: every address is inside row `id`; prefetch never faults.
            let p = unsafe { self.base.add(id as usize * self.dim) } as *const u8;
            let mut off = 0;
            while off < self.dim * 4 {
                super::prefetch_read(unsafe { p.add(off) });
                off += 128;
            }
        }
    }
}

impl BuildSpace for F32Space<'_> {
    #[inline]
    fn between(&self, a: u32, b: u32) -> f32 {
        self.dist(self.row(a), b)
    }
}

/// An in-memory HNSW index over f32 vectors with tombstone deletes.
///
/// `insert`, `delete` and every search take `&self` and may run concurrently
/// from many threads.
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    cap: usize,
    arena: Arena,
    state: Box<[AtomicU8]>,
    next: AtomicUsize,
    graph: Graph,
    deleted: AtomicBitset,
    has_deletes: AtomicBool,
    pool: Mutex<Vec<Scratch>>,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, params: GraphParams, capacity: usize) -> HnswIndex {
        assert!(dim > 0, "dimension must be positive");
        HnswIndex {
            dim,
            metric,
            cap: capacity,
            arena: Arena::zeroed(capacity * dim),
            state: (0..capacity).map(|_| AtomicU8::new(EMPTY)).collect(),
            next: AtomicUsize::new(0),
            graph: Graph::new(params, capacity),
            deleted: AtomicBitset::new(capacity),
            has_deletes: AtomicBool::new(false),
            pool: Mutex::new(Vec::new()),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn params(&self) -> &GraphParams {
        self.graph.params()
    }

    /// Rows allocated so far (including rows still being inserted).
    pub fn len(&self) -> usize {
        self.next.load(Ordering::Acquire).min(self.cap)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted.count()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The arena as a [`Space`]. Crate-private: it can address rows that are
    /// still being written, which only the graph code knows to avoid.
    pub(crate) fn space(&self) -> F32Space<'_> {
        F32Space {
            base: self.arena.ptr.as_ptr(),
            dim: self.dim,
            rows: self.cap,
            metric: self.metric,
            k: kernels(),
            _life: PhantomData,
        }
    }

    fn scratch(&self) -> Scratch {
        let s = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
        s.unwrap_or_else(|| Scratch::new(self.cap))
    }

    fn recycle(&self, s: Scratch) {
        let mut p = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if p.len() < 64 {
            p.push(s);
        }
    }

    fn check_dim(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.dim {
            return invalid(format!("vector has {} dimensions, index has {}", v.len(), self.dim));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return invalid("vector contains NaN or infinity");
        }
        Ok(())
    }

    /// Writes a prepared copy of `v` into the unpublished slot `id`.
    fn write_slot(&self, id: usize, v: &[f32]) {
        debug_assert_eq!(self.state[id].load(Ordering::Relaxed), EMPTY);
        // SAFETY: `id` was handed out exactly once by `next.fetch_add`, so no
        // other thread writes this slot, and no thread reads it before the
        // `Release` store of WRITTEN below or a graph link to it.
        let dst = unsafe { std::slice::from_raw_parts_mut(self.arena.ptr.as_ptr().add(id * self.dim), self.dim) };
        dst.copy_from_slice(v);
        if self.metric.normalizes() {
            crate::distance::normalize(dst);
        }
        self.state[id].store(WRITTEN, Ordering::Release);
    }

    /// Adds a vector and returns its row id.
    pub fn insert(&self, v: &[f32]) -> Result<u32> {
        self.check_dim(v)?;
        let id = self.next.fetch_add(1, Ordering::AcqRel);
        if id >= self.cap {
            return Err(Error::Full);
        }
        self.write_slot(id, v);
        self.link(id as u32);
        Ok(id as u32)
    }

    fn link(&self, id: u32) {
        let mut sc = self.scratch();
        let space = self.space();
        self.graph.insert(&space, id, space.row(id), &mut sc);
        self.state[id as usize].store(LINKED, Ordering::Release);
        self.recycle(sc);
    }

    /// Adds `data.len() / dim` vectors using `threads` threads. Row ids are
    /// consecutive and in input order; returns the first one.
    pub fn insert_batch(&self, data: &[f32], threads: usize) -> Result<u32> {
        if !data.len().is_multiple_of(self.dim) {
            return invalid("batch length is not a multiple of the dimension");
        }
        let n = data.len() / self.dim;
        for v in data.chunks_exact(self.dim) {
            self.check_dim(v)?;
        }
        let start = self.next.fetch_add(n, Ordering::AcqRel);
        if start + n > self.cap {
            return Err(Error::Full);
        }
        for (i, v) in data.chunks_exact(self.dim).enumerate() {
            self.write_slot(start + i, v);
        }
        let cursor = AtomicUsize::new(0);
        let threads = threads.max(1).min(n.max(1));
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        let i = cursor.fetch_add(32, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        for j in i..(i + 32).min(n) {
                            self.link((start + j) as u32);
                        }
                    }
                });
            }
        });
        Ok(start as u32)
    }

    /// The stored (normalised, for cosine) vector of a row.
    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        let st = self.state.get(id as usize)?.load(Ordering::Acquire);
        if st == EMPTY { None } else { Some(self.space().row(id)) }
    }

    /// Marks a row deleted. Returns false if it was already deleted or does
    /// not exist.
    pub fn delete(&self, id: u32) -> bool {
        if self.vector(id).is_none() {
            return false;
        }
        let fresh = self.deleted.set(id as usize);
        if fresh {
            self.has_deletes.store(true, Ordering::Release);
        }
        fresh
    }

    pub fn is_deleted(&self, id: u32) -> bool {
        self.deleted.get(id as usize)
    }

    pub fn tombstones(&self) -> &AtomicBitset {
        &self.deleted
    }

    fn prepare_query<'q>(&self, q: &'q [f32], buf: &'q mut Vec<f32>) -> Result<&'q [f32]> {
        self.check_dim(q)?;
        if self.metric.normalizes() {
            *buf = self.metric.prepare(q);
            Ok(buf)
        } else {
            Ok(q)
        }
    }

    /// Approximate k nearest neighbours with beam width `ef`.
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Result<Vec<Neighbor>> {
        self.search_with(q, k, ef, &|_| true)
    }

    /// Approximate search restricted to rows for which `accept` is true.
    pub fn search_with<F: Fn(u32) -> bool>(&self, q: &[f32], k: usize, ef: usize, accept: &F) -> Result<Vec<Neighbor>> {
        let mut buf = Vec::new();
        let q = self.prepare_query(q, &mut buf)?;
        let mut sc = self.scratch();
        let space = self.space();
        let r = if self.has_deletes.load(Ordering::Acquire) {
            graph::knn(&self.graph, &space, q, k, ef, &|id| !self.deleted.get(id as usize) && accept(id), &mut sc)
        } else {
            graph::knn(&self.graph, &space, q, k, ef, accept, &mut sc)
        };
        self.recycle(sc);
        Ok(r)
    }

    /// Like [`search`](Self::search) but also returns the number of distance
    /// computations, for benchmarks.
    pub fn search_counted(&self, q: &[f32], k: usize, ef: usize) -> Result<(Vec<Neighbor>, u64)> {
        let mut buf = Vec::new();
        let q = self.prepare_query(q, &mut buf)?;
        let mut sc = self.scratch();
        sc.dist_count = 0;
        let space = self.space();
        let r = graph::knn(&self.graph, &space, q, k, ef, &|id| !self.deleted.get(id as usize), &mut sc);
        let c = sc.dist_count;
        self.recycle(sc);
        Ok((r, c))
    }

    /// Exact search over every live row accepted by `accept`.
    pub fn brute_force<F: Fn(u32) -> bool>(&self, q: &[f32], k: usize, accept: &F) -> Result<Vec<Neighbor>> {
        let mut buf = Vec::new();
        let q = self.prepare_query(q, &mut buf)?;
        let space = self.space();
        let mut top = TopK::new(k);
        for id in 0..self.len() as u32 {
            if self.state[id as usize].load(Ordering::Acquire) != EMPTY && !self.deleted.get(id as usize) && accept(id)
            {
                top.push(Neighbor::new(id, space.dist(q, id)));
            }
        }
        Ok(top.into_sorted())
    }

    /// Graph search with an already prepared query (the collection layer
    /// prepares once for all segments). Tombstones are excluded.
    pub(crate) fn knn_prepared<F: Fn(u32) -> bool>(&self, q: &[f32], k: usize, ef: usize, accept: &F) -> Vec<Neighbor> {
        let mut sc = self.scratch();
        let space = self.space();
        let r = graph::knn(&self.graph, &space, q, k, ef, &|id| !self.deleted.get(id as usize) && accept(id), &mut sc);
        self.recycle(sc);
        r
    }

    /// Exact search over explicitly listed rows (filtered brute force). Rows
    /// that are not yet written or are tombstoned are skipped.
    pub(crate) fn exact_prepared(&self, q: &[f32], k: usize, ids: impl Iterator<Item = u32>) -> Vec<Neighbor> {
        let space = self.space();
        let mut top = TopK::new(k);
        for id in ids {
            let ok = self.state.get(id as usize).is_some_and(|s| s.load(Ordering::Acquire) != EMPTY);
            if ok && !self.deleted.get(id as usize) {
                let d = space.dist(q, id);
                if d < top.bound() {
                    top.push(Neighbor::new(id, d));
                }
            }
        }
        top.into_sorted()
    }

    /// Compresses every row (in id order) for [`search_quantized`](Self::search_quantized).
    pub fn build_codes(&self, q: crate::quant::Quantization, threads: usize) -> Result<Option<crate::quant::Codes>> {
        let n = self.len();
        let mut data = Vec::with_capacity(n * self.dim);
        for id in 0..n as u32 {
            match self.vector(id) {
                Some(v) => data.extend_from_slice(v),
                None => data.extend(std::iter::repeat_n(0.0, self.dim)),
            }
        }
        crate::quant::Codes::build(q, &data, self.dim, self.metric, self.params().seed, threads).map_err(Error::Invalid)
    }

    /// Search that walks the graph with compressed distances and re-ranks
    /// `k * rerank` candidates with the stored f32 vectors.
    pub fn search_quantized(
        &self,
        codes: &crate::quant::Codes,
        q: &[f32],
        k: usize,
        ef: usize,
        rerank: usize,
    ) -> Result<Vec<Neighbor>> {
        let mut buf = Vec::new();
        let q = self.prepare_query(q, &mut buf)?;
        if codes.rows() < self.len() {
            return invalid("codes cover fewer rows than the index; rebuild them");
        }
        let mut sc = self.scratch();
        let accept = |id: u32| !self.deleted.get(id as usize);
        let r = codes.search(&self.graph, &self.space(), q, k, ef, rerank, &accept, &mut sc);
        self.recycle(sc);
        Ok(r)
    }

    /// Checks the graph invariants over every linked row.
    pub fn check(&self) -> std::result::Result<GraphStats, String> {
        let ids = (0..self.len() as u32).filter(|&i| self.state[i as usize].load(Ordering::Acquire) == LINKED);
        graph::check_invariants(&self.graph, ids)
    }

    /// Approximate resident bytes: vectors plus graph.
    pub fn memory_bytes(&self) -> usize {
        self.cap * self.dim * 4 + self.graph.memory_bytes() + self.cap / 8
    }

    /// Builds a new index containing only the live rows (the periodic rebuild
    /// that clears tombstones). Returns it with the old-to-new id mapping.
    pub fn rebuilt(&self, capacity: usize, threads: usize) -> Result<(HnswIndex, Vec<Option<u32>>)> {
        let n = self.len();
        let mut map = vec![None; n];
        let mut data = Vec::new();
        let mut next = 0u32;
        for id in 0..n as u32 {
            if let Some(v) = self.vector(id)
                && !self.is_deleted(id)
            {
                data.extend_from_slice(v);
                map[id as usize] = Some(next);
                next += 1;
            }
        }
        let fresh = HnswIndex::new(self.dim, self.metric, *self.params(), capacity.max(next as usize));
        if next > 0 {
            fresh.insert_batch(&data, threads)?;
        }
        Ok((fresh, map))
    }
}

#[cfg(test)]
mod tests;
