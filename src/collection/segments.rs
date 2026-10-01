//! The two kinds of segment a collection searches.
//!
//! * [`Growing`]: the one segment that accepts writes. An in-memory
//!   [`HnswIndex`] with fixed capacity plus per-row external ids, LSNs and
//!   payloads. Rows become visible only when the collection commits them.
//! * [`Sealed`]: an immutable, memory-mapped segment file plus an in-memory
//!   tombstone bitmap and payload index.
//!
//! Both answer the same per-segment query: top-k among live rows matching an
//! optional filter, using the strategy chosen from the filter's exact match
//! count (see [`choose_strategy`]).

use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::index::bitset::{AtomicBitset, Bitset};
use crate::index::flat::exact_knn;
use crate::index::graph::{GraphParams, Scratch, knn};
use crate::index::{HnswIndex, Neighbor};
use crate::payload::{Filter, PayloadIndex};
use crate::quant::{Codes, Quantization};
use crate::storage::SyncMode;
use crate::storage::segment::{SegmentFile, SegmentWrite, read_tombstones, write_segment, write_tombstones};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

/// How one segment answered a filtered query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Pick per segment from the filter's match count.
    Auto,
    /// HNSW traversal that skips non-matching rows in the result set.
    Graph,
    /// Exact scan over the matching rows only.
    Brute,
}

/// Brute force over the matching rows costs about `matched` distance
/// computations. A filtered graph search must expand roughly `ef / s` nodes
/// (with `s = matched / live`) of `M0` neighbours each to collect `ef`
/// matches, about `ef * M0 * live / matched` distances. The two are equal at
/// `matched = sqrt(ef * M0 * live)`; `factor` scales that crossover and was
/// calibrated with `quiver bench filter` (see docs/DESIGN.md).
pub fn choose_strategy(matched: usize, live: usize, ef: usize, m0: usize, factor: f64) -> Strategy {
    if live <= 2048 {
        return Strategy::Brute;
    }
    let crossover = factor * ((ef.max(10) * m0) as f64 * live as f64).sqrt();
    if (matched as f64) <= crossover { Strategy::Brute } else { Strategy::Graph }
}

/// A per-segment hit before payloads are attached.
#[derive(Clone, Copy, Debug)]
pub struct SegHit {
    pub ext: u64,
    pub dist: f32,
    pub seg: u64,
    pub row: u32,
}

/// The parameters of one per-segment search.
pub struct SegQuery<'a> {
    /// The prepared (normalised, for cosine) query.
    pub q: &'a [f32],
    pub k: usize,
    pub ef: usize,
    pub filter: Option<&'a Filter>,
    pub strategy: Strategy,
    pub rerank: usize,
    pub brute_factor: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SegPlan {
    pub segment: u64,
    pub live: usize,
    /// Rows matching the filter (absent without a filter).
    pub matched: Option<usize>,
    pub strategy: Option<Strategy>,
}

pub struct Growing {
    pub id: u64,
    pub index: HnswIndex,
    reserved: AtomicUsize,
    ext: Box<[AtomicU64]>,
    lsn: Box<[AtomicU64]>,
    payload: Box<[OnceLock<Value>]>,
    pub visible: AtomicBitset,
    pindex: RwLock<PayloadIndex>,
}

impl Growing {
    pub fn new(id: u64, dim: usize, metric: Metric, params: GraphParams, capacity: usize) -> Growing {
        Growing {
            id,
            index: HnswIndex::new(dim, metric, params, capacity),
            reserved: AtomicUsize::new(0),
            ext: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            lsn: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            payload: (0..capacity).map(|_| OnceLock::new()).collect(),
            visible: AtomicBitset::new(capacity),
            pindex: RwLock::new(PayloadIndex::new()),
        }
    }

    pub fn capacity(&self) -> usize {
        self.index.capacity()
    }

    /// Reserves room for `n` rows; false if the segment is too full.
    pub fn try_reserve(&self, n: usize) -> bool {
        let before = self.reserved.fetch_add(n, Ordering::AcqRel);
        if before + n > self.capacity() {
            self.reserved.fetch_sub(n, Ordering::AcqRel);
            return false;
        }
        true
    }

    pub fn rows(&self) -> usize {
        self.index.len()
    }

    /// Inserts a row (invisible until the collection commits it).
    pub fn insert(&self, ext: u64, lsn: u64, v: &[f32], payload: Option<Value>) -> Result<u32> {
        let row = self.index.insert(v)?;
        self.ext[row as usize].store(ext, Ordering::Relaxed);
        self.lsn[row as usize].store(lsn, Ordering::Relaxed);
        if let Some(p) = payload {
            self.pindex.write().unwrap_or_else(|e| e.into_inner()).add(row, &p);
            let _ = self.payload[row as usize].set(p);
        }
        Ok(row)
    }

    /// Makes a committed row visible to searches.
    pub fn publish(&self, row: u32) {
        self.visible.set(row as usize);
    }

    pub fn tombstone(&self, row: u32) {
        self.index.delete(row);
    }

    pub fn is_live(&self, row: u32) -> bool {
        self.visible.get(row as usize) && !self.index.is_deleted(row)
    }

    pub fn ext_id(&self, row: u32) -> u64 {
        self.ext[row as usize].load(Ordering::Relaxed)
    }

    pub fn row_lsn(&self, row: u32) -> u64 {
        self.lsn[row as usize].load(Ordering::Relaxed)
    }

    pub fn payload(&self, row: u32) -> Option<Value> {
        self.payload[row as usize].get().cloned()
    }

    pub fn vector(&self, row: u32) -> Option<Vec<f32>> {
        self.index.vector(row).map(|v| v.to_vec())
    }

    pub fn live_count(&self) -> usize {
        (0..self.rows() as u32).filter(|&r| self.is_live(r)).count()
    }

    fn live_bits(&self, n: usize) -> Bitset {
        let mut b = Bitset::from_words(self.visible.snapshot(), n);
        b.and_not_atomic(self.index.tombstones());
        b
    }

    pub fn search(&self, sq: &SegQuery<'_>, plan: &mut SegPlan) -> Vec<SegHit> {
        let n = self.rows();
        plan.segment = self.id;
        let to_hits = |ns: Vec<Neighbor>| -> Vec<SegHit> {
            ns.into_iter()
                .map(|nb| SegHit { ext: self.ext_id(nb.id), dist: nb.dist, seg: self.id, row: nb.id })
                .collect()
        };
        let Some(f) = sq.filter else {
            plan.live = self.index.len() - self.index.deleted_count();
            let r = self.index.knn_prepared(sq.q, sq.k, sq.ef, &|r| self.visible.get(r as usize));
            return to_hits(r);
        };
        let mut bits = self.pindex.read().unwrap_or_else(|e| e.into_inner()).eval(f, n);
        let live = self.live_bits(n);
        bits.and_with(&live);
        let matched = bits.count();
        plan.live = live.count();
        plan.matched = Some(matched);
        if matched == 0 {
            plan.strategy = Some(Strategy::Brute);
            return Vec::new();
        }
        let strat = match sq.strategy {
            Strategy::Auto => choose_strategy(matched, plan.live, sq.ef, 2 * self.index.params().m, sq.brute_factor),
            s => s,
        };
        plan.strategy = Some(strat);
        let r = match strat {
            Strategy::Brute => self.index.exact_prepared(sq.q, sq.k, bits.iter().map(|r| r as u32)),
            _ => self.index.knn_prepared(sq.q, sq.k, sq.ef, &|r| bits.get(r as usize)),
        };
        to_hits(r)
    }

    /// Writes this segment to `path` as a sealed segment file (the caller
    /// guarantees no writes are in flight) and returns its tombstones.
    pub fn write(&self, path: &Path, quant: Quantization, threads: usize, mode: SyncMode) -> Result<Vec<u64>> {
        let n = self.rows();
        let dim = self.index.dim();
        let mut ids = Vec::with_capacity(n);
        let mut lsns = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n * dim);
        let mut payloads = Vec::with_capacity(n);
        let mut dead = Bitset::new(n);
        for r in 0..n as u32 {
            ids.push(self.ext_id(r));
            lsns.push(self.row_lsn(r));
            let v = self.index.vector(r).ok_or_else(|| Error::Corrupt(format!("growing row {r} has no vector")))?;
            vectors.extend_from_slice(v);
            payloads.push(self.payload[r as usize].get().map(|p| serde_json::to_vec(p).unwrap()).unwrap_or_default());
            if !self.is_live(r) {
                dead.set(r as usize);
            }
        }
        let graph = self.index.graph().freeze(n);
        let codes = Codes::build(quant, &vectors, dim, self.index.metric(), self.index.params().seed, threads)
            .map_err(Error::Invalid)?;
        write_segment(
            path,
            &SegmentWrite {
                dim,
                metric: self.index.metric(),
                params: *self.index.params(),
                ids: &ids,
                lsns: &lsns,
                vectors: &vectors,
                graph: &graph,
                payloads: &payloads,
                codes: codes.as_ref(),
            },
            mode,
        )?;
        Ok(dead.words().to_vec())
    }

    /// Hands the payload index to the sealed segment that replaces this one.
    pub fn take_payload_index(&self) -> PayloadIndex {
        std::mem::take(&mut *self.pindex.write().unwrap_or_else(|e| e.into_inner()))
    }
}

pub struct Sealed {
    pub id: u64,
    pub file: SegmentFile,
    pub deleted: AtomicBitset,
    /// Tombstones changed since they were last persisted.
    pub dirty: AtomicBool,
    pindex: PayloadIndex,
    pool: Mutex<Vec<Scratch>>,
    path: PathBuf,
}

pub fn segment_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("seg-{id:08}.qseg"))
}

pub fn tombstone_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("seg-{id:08}.del"))
}

impl Sealed {
    /// Opens segment `id` in `dir` with its tombstones; builds the payload
    /// index from the stored payloads unless one is supplied.
    pub fn open(dir: &Path, id: u64, pindex: Option<PayloadIndex>) -> Result<Sealed> {
        let path = segment_path(dir, id);
        let file = SegmentFile::open(&path)?;
        let rows = file.rows();
        let dp = tombstone_path(dir, id);
        let deleted = if dp.exists() {
            AtomicBitset::from_words(&read_tombstones(&dp, rows)?, rows)
        } else {
            AtomicBitset::new(rows)
        };
        let pindex = match pindex {
            Some(mut p) => {
                p.seal();
                p
            }
            None => {
                let mut p = PayloadIndex::new();
                for r in 0..rows as u32 {
                    let b = file.payload(r);
                    if !b.is_empty() {
                        let v: Value = serde_json::from_slice(b)
                            .map_err(|e| Error::Corrupt(format!("{}: payload of row {r}: {e}", path.display())))?;
                        p.add(r, &v);
                    }
                }
                p.seal();
                p
            }
        };
        Ok(Sealed { id, file, deleted, dirty: AtomicBool::new(false), pindex, pool: Mutex::new(Vec::new()), path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows(&self) -> usize {
        self.file.rows()
    }

    pub fn live_count(&self) -> usize {
        self.rows() - self.deleted.count()
    }

    pub fn tombstone(&self, row: u32) {
        if self.deleted.set(row as usize) {
            self.dirty.store(true, Ordering::Release);
        }
    }

    pub fn is_live(&self, row: u32) -> bool {
        (row as usize) < self.rows() && !self.deleted.get(row as usize)
    }

    pub fn persist_tombstones(&self, dir: &Path, mode: SyncMode) -> Result<()> {
        write_tombstones(&tombstone_path(dir, self.id), self.rows(), &self.deleted.snapshot(), mode)
    }

    pub fn payload(&self, row: u32) -> Option<Value> {
        let b = self.file.payload(row);
        if b.is_empty() { None } else { serde_json::from_slice(b).ok() }
    }

    pub fn vector(&self, row: u32) -> Vec<f32> {
        self.file.vector(row).to_vec()
    }

    fn scratch(&self) -> Scratch {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_else(|| Scratch::new(self.rows()))
    }

    fn recycle(&self, s: Scratch) {
        let mut p = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if p.len() < 64 {
            p.push(s);
        }
    }

    pub fn search(&self, sq: &SegQuery<'_>, plan: &mut SegPlan) -> Vec<SegHit> {
        plan.segment = self.id;
        let rows = self.rows();
        let graph = self.file.graph();
        let space = self.file.space();
        let mut sc = self.scratch();
        let traverse = |accept: &dyn Fn(u32) -> bool, sc: &mut Scratch| -> Vec<Neighbor> {
            match self.file.codes() {
                Some(c) => c.search(&graph, &space, sq.q, sq.k, sq.ef, sq.rerank, &accept, sc),
                None => knn(&graph, &space, sq.q, sq.k, sq.ef, &accept, sc),
            }
        };
        let r = match sq.filter {
            None => {
                plan.live = self.live_count();
                traverse(&|r| !self.deleted.get(r as usize), &mut sc)
            }
            Some(f) => {
                let mut bits = self.pindex.eval(f, rows);
                bits.and_not_atomic(&self.deleted);
                let matched = bits.count();
                plan.live = self.live_count();
                plan.matched = Some(matched);
                let strat = match sq.strategy {
                    Strategy::Auto => {
                        choose_strategy(matched, plan.live, sq.ef, 2 * self.file.params().m, sq.brute_factor)
                    }
                    s => s,
                };
                plan.strategy = Some(strat);
                if matched == 0 {
                    Vec::new()
                } else if strat == Strategy::Brute {
                    exact_knn(&space, sq.q, bits.iter().map(|r| r as u32), sq.k)
                } else {
                    traverse(&|r| bits.get(r as usize), &mut sc)
                }
            }
        };
        self.recycle(sc);
        let ids = self.file.ids();
        r.into_iter().map(|nb| SegHit { ext: ids[nb.id as usize], dist: nb.dist, seg: self.id, row: nb.id }).collect()
    }
}
