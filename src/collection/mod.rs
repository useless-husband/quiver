//! Collections: durable, searchable sets of `(id, vector, payload)` points.
//!
//! A collection is a directory:
//!
//! ```text
//! MANIFEST.json          config, sealed segment ids, WAL generation, LSN mark
//! wal-<gen>.log          operations since the last checkpoint
//! seg-<id>.qseg          sealed, memory-mapped segments
//! seg-<id>.del           their tombstones (as of the last checkpoint)
//! ```
//!
//! **Writes.** An upsert or delete is appended to the WAL and synced (shared
//! with concurrent writers by group commit) *before* it touches memory; the
//! client is answered after it has been applied. Upserts go into the one
//! growing segment's HNSW graph concurrently; the cheap commit step (point the
//! id at its new row, tombstone the old one) runs under the id-map lock and
//! resolves races on the same id by LSN, so the in-memory result is the one a
//! sequential replay of the WAL produces.
//!
//! **Checkpoint.** When the growing segment is full (or on request) writers
//! are paused, the segment is written as a sealed file (with quantized codes
//! if configured), dirty tombstones are persisted, a fresh WAL is created and
//! the manifest is replaced atomically; then the old WAL is deleted.
//!
//! **Compaction** rebuilds sealed segments without their tombstoned rows (a
//! fresh HNSW graph) in the background, then swaps them in under a short
//! writer pause, carrying over deletes that happened meanwhile.
//!
//! **Recovery** opens the manifest's segments, rebuilds the id map from their
//! rows (highest LSN wins), and replays the WAL: inserts run in parallel,
//! commits in LSN order.
//!
//! Lock order: `maint` → `gate` → `ids` → `segs` → `wal`.

pub mod segments;

pub use segments::{SegPlan, Strategy, choose_strategy};

use crate::distance::Metric;
use crate::error::{Error, Result, invalid};
use crate::index::HnswIndex;
use crate::index::bitset::Bitset;
use crate::index::graph::GraphParams;
use crate::payload::Filter;
use crate::quant::{Codes, Quantization};
use crate::storage::SyncMode;
use crate::storage::fs::{sync_dir, write_atomic};
use crate::storage::segment::{SegmentWrite, write_segment, write_tombstones};
use crate::storage::wal::{Wal, WalOp};
use segments::{Growing, Sealed, SegHit, SegQuery, segment_path, tombstone_path};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Largest number of points accepted in one upsert call.
pub const MAX_BATCH: usize = 10_000;
/// Largest accepted dimension.
pub const MAX_DIM: usize = 16_384;
/// Largest serialized payload per point.
pub const MAX_PAYLOAD: usize = 1 << 20;

fn d_m() -> usize {
    16
}
fn d_efc() -> usize {
    200
}
fn d_ef() -> usize {
    64
}
fn d_rerank() -> usize {
    4
}
fn d_seg() -> usize {
    100_000
}
fn d_bf() -> f64 {
    1.0
}

/// Per-collection settings, fixed at creation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionConfig {
    pub dim: usize,
    pub metric: Metric,
    #[serde(default = "d_m")]
    pub m: usize,
    #[serde(default = "d_efc")]
    pub ef_construction: usize,
    /// Default beam width for searches.
    #[serde(default = "d_ef")]
    pub ef_search: usize,
    /// Compression used by sealed segments for graph traversal.
    #[serde(default)]
    pub quantization: Quantization,
    /// Candidates re-ranked at full precision per result (quantized only).
    #[serde(default = "d_rerank")]
    pub rerank: usize,
    /// Rows per segment: the growing segment is sealed when it is full.
    #[serde(default = "d_seg")]
    pub segment_rows: usize,
    /// Scales the brute-force / graph crossover for filtered search.
    #[serde(default = "d_bf")]
    pub brute_force_factor: f64,
    #[serde(default)]
    pub seed: u64,
}

impl CollectionConfig {
    pub fn new(dim: usize, metric: Metric) -> CollectionConfig {
        CollectionConfig {
            dim,
            metric,
            m: d_m(),
            ef_construction: d_efc(),
            ef_search: d_ef(),
            quantization: Quantization::None,
            rerank: d_rerank(),
            segment_rows: d_seg(),
            brute_force_factor: d_bf(),
            seed: 0,
        }
    }

    pub fn graph_params(&self) -> GraphParams {
        GraphParams { m: self.m, ef_construction: self.ef_construction, seed: self.seed }
    }

    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.dim > MAX_DIM {
            return invalid(format!("dim must be between 1 and {MAX_DIM}"));
        }
        self.graph_params().validate().map_err(Error::Invalid)?;
        if self.ef_search == 0 || self.ef_search > 10_000 {
            return invalid("ef_search must be between 1 and 10000");
        }
        if self.segment_rows < 16 || self.segment_rows > 1 << 26 {
            return invalid("segment_rows must be between 16 and 67108864");
        }
        if self.rerank > 100 {
            return invalid("rerank must be at most 100");
        }
        if !self.brute_force_factor.is_finite() || self.brute_force_factor < 0.0 {
            return invalid("brute_force_factor must be a non-negative number");
        }
        if let Quantization::Pq { m } = self.quantization
            && (m == 0 || !self.dim.is_multiple_of(m))
        {
            return invalid(format!("pq m={m} must divide dim={}", self.dim));
        }
        Ok(())
    }
}

/// Process-level options for opening collections.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub sync: SyncMode,
    /// Threads for segment builds (compaction, PQ training, WAL replay).
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { sync: SyncMode::Full, threads: 4 }
    }
}

/// A point as written by clients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub id: u64,
    pub vector: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

/// A search request.
#[derive(Clone, Debug)]
pub struct Query<'a> {
    pub vector: &'a [f32],
    pub k: usize,
    pub ef: Option<usize>,
    pub filter: Option<&'a Filter>,
    pub strategy: Strategy,
    pub rerank: Option<usize>,
    pub with_payload: bool,
    pub with_vector: bool,
}

impl<'a> Query<'a> {
    pub fn new(vector: &'a [f32], k: usize) -> Query<'a> {
        Query {
            vector,
            k,
            ef: None,
            filter: None,
            strategy: Strategy::Auto,
            rerank: None,
            with_payload: false,
            with_vector: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchResult {
    pub hits: Vec<Hit>,
    /// What each segment did (strategy, filter match count).
    pub plan: Vec<SegPlan>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Record {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SegmentStats {
    pub id: u64,
    pub kind: &'static str,
    pub rows: usize,
    pub live: usize,
    pub bytes: usize,
    pub quantization: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CollectionStats {
    pub name: String,
    pub config: CollectionConfig,
    pub points: usize,
    pub segments: Vec<SegmentStats>,
    pub wal_bytes: u64,
    pub wal_syncs: u64,
    pub next_lsn: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompactionReport {
    pub inputs: Vec<u64>,
    pub output: Option<u64>,
    pub rows_in: usize,
    pub rows_out: usize,
    pub seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Slot {
    Live { seg: u64, row: u32, lsn: u64 },
    Dead { lsn: u64 },
}

impl Slot {
    fn lsn(&self) -> u64 {
        match *self {
            Slot::Live { lsn, .. } | Slot::Dead { lsn } => lsn,
        }
    }
}

#[derive(Clone)]
struct Segments {
    sealed: Vec<Arc<Sealed>>,
    growing: Arc<Growing>,
}

impl Segments {
    fn tombstone(&self, seg: u64, row: u32) {
        if self.growing.id == seg {
            self.growing.tombstone(row);
        } else if let Some(s) = self.sealed.iter().find(|s| s.id == seg) {
            s.tombstone(row);
        }
    }

    fn record(&self, id: u64, seg: u64, row: u32, with_vector: bool) -> Option<Record> {
        if self.growing.id == seg {
            let g = &self.growing;
            return Some(Record { id, vector: with_vector.then(|| g.vector(row)).flatten(), payload: g.payload(row) });
        }
        let s = self.sealed.iter().find(|s| s.id == seg)?;
        Some(Record { id, vector: with_vector.then(|| s.vector(row)), payload: s.payload(row) })
    }
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    config: CollectionConfig,
    segments: Vec<u64>,
    wal: u64,
    next_segment: u64,
    last_lsn: u64,
}

const MANIFEST: &str = "MANIFEST.json";

fn wal_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("wal-{generation:08}.log"))
}

pub struct Collection {
    name: String,
    dir: PathBuf,
    config: CollectionConfig,
    opts: Options,
    wal: RwLock<Arc<Wal>>,
    segs: RwLock<Segments>,
    ids: Mutex<HashMap<u64, Slot>>,
    gate: RwLock<()>,
    maint: Mutex<()>,
    compacting: Mutex<()>,
    next_seg: AtomicU64,
    /// Highest LSN applied in memory.
    applied_lsn: AtomicU64,
    /// The LSN mark of the manifest on disk (all operations up to it are in
    /// sealed segments).
    manifest_lsn: AtomicU64,
    failed: AtomicBool,
    closed: AtomicBool,
    _lock: crate::storage::fs::DirLock,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn read<T>(m: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(|e| e.into_inner())
}

fn write<T>(m: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(|e| e.into_inner())
}

impl Collection {
    /// Creates a new collection in the (empty or absent) directory `dir`.
    pub fn create(dir: &Path, name: &str, config: CollectionConfig, opts: Options) -> Result<Collection> {
        config.validate()?;
        std::fs::create_dir_all(dir)?;
        let m = Manifest { version: 1, config, segments: vec![], wal: 1, next_segment: 1, last_lsn: 0 };
        Wal::create(&wal_path(dir, 1), 1, 1, opts.sync)?;
        write_atomic(&dir.join(MANIFEST), &serde_json::to_vec_pretty(&m).unwrap(), opts.sync)?;
        Self::open(dir, name, opts)
    }

    /// Opens a collection, recovering from the WAL.
    pub fn open(dir: &Path, name: &str, opts: Options) -> Result<Collection> {
        let dir_lock = crate::storage::fs::DirLock::acquire(dir)?;
        let mpath = dir.join(MANIFEST);
        let m: Manifest = serde_json::from_slice(&std::fs::read(&mpath)?)
            .map_err(|e| Error::Corrupt(format!("{}: {e}", mpath.display())))?;
        m.config.validate()?;
        // Remove leftovers of interrupted checkpoints and compactions.
        let keep: HashSet<String> = m
            .segments
            .iter()
            .flat_map(|&id| {
                [segment_path(dir, id), tombstone_path(dir, id)]
                    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            })
            .chain([wal_path(dir, m.wal).file_name().unwrap().to_string_lossy().into_owned(), MANIFEST.to_string()])
            .collect();
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().into_owned();
            let ours = n.ends_with(".tmp") || n.starts_with("seg-") || n.starts_with("wal-");
            if ours && !keep.contains(&n) {
                std::fs::remove_file(e.path())?;
            }
        }
        let mut sealed = Vec::new();
        for &id in &m.segments {
            sealed.push(Arc::new(Sealed::open(dir, id, None)?));
        }
        // Id map from sealed rows; a duplicate id keeps its highest LSN.
        let mut ids: HashMap<u64, Slot> = HashMap::new();
        for s in &sealed {
            let (ext, lsns) = (s.file.ids(), s.file.lsns());
            for r in 0..s.rows() as u32 {
                if !s.is_live(r) {
                    continue;
                }
                let (id, lsn) = (ext[r as usize], lsns[r as usize]);
                match ids.get(&id).copied() {
                    Some(Slot::Live { seg, row, lsn: l }) if l > lsn => {
                        let _ = (seg, row);
                        s.tombstone(r);
                    }
                    Some(Slot::Live { seg, row, .. }) => {
                        if let Some(o) = sealed.iter().find(|x| x.id == seg) {
                            o.tombstone(row);
                        }
                        ids.insert(id, Slot::Live { seg: s.id, row: r, lsn });
                    }
                    _ => {
                        ids.insert(id, Slot::Live { seg: s.id, row: r, lsn });
                    }
                }
            }
        }
        let wp = wal_path(dir, m.wal);
        let (wal, records) = if wp.exists() {
            let (w, replay) = Wal::open(&wp, m.last_lsn + 1, opts.sync)?;
            if replay.discarded > 0 {
                eprintln!("quiver: {name}: discarded {} bytes of torn WAL tail", replay.discarded);
            }
            (w, replay.records)
        } else {
            (Wal::create(&wp, m.wal, m.last_lsn + 1, opts.sync)?, Vec::new())
        };
        let upserts = records.iter().filter(|r| matches!(r.op, WalOp::Upsert { .. })).count();
        let cap = m.config.segment_rows.max(upserts);
        let growing =
            Arc::new(Growing::new(m.next_segment, m.config.dim, m.config.metric, m.config.graph_params(), cap));
        let c = Collection {
            name: name.to_string(),
            dir: dir.to_path_buf(),
            config: m.config,
            opts,
            wal: RwLock::new(Arc::new(wal)),
            segs: RwLock::new(Segments { sealed, growing: growing.clone() }),
            ids: Mutex::new(ids),
            gate: RwLock::new(()),
            maint: Mutex::new(()),
            compacting: Mutex::new(()),
            next_seg: AtomicU64::new(m.next_segment + 1),
            applied_lsn: AtomicU64::new(records.last().map_or(m.last_lsn, |r| r.lsn.max(m.last_lsn))),
            manifest_lsn: AtomicU64::new(m.last_lsn),
            failed: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            _lock: dir_lock,
        };
        c.replay(&growing, records, upserts)?;
        Ok(c)
    }

    /// Re-applies WAL records: graph inserts in parallel, commits in LSN
    /// order (the same commit logic as the live path).
    fn replay(&self, g: &Growing, records: Vec<crate::storage::wal::WalRecord>, upserts: usize) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        assert!(g.try_reserve(upserts));
        let ups: Vec<usize> =
            records.iter().enumerate().filter(|(_, r)| matches!(r.op, WalOp::Upsert { .. })).map(|(i, _)| i).collect();
        let rows: Vec<AtomicU32> = (0..records.len()).map(|_| AtomicU32::new(u32::MAX)).collect();
        let err = Mutex::new(None);
        let cursor = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..self.opts.threads.max(1) {
                s.spawn(|| {
                    loop {
                        let j = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = ups.get(j) else { break };
                        let r = &records[i];
                        let WalOp::Upsert { id, vector, payload } = &r.op else { unreachable!() };
                        let p = if payload.is_empty() { None } else { serde_json::from_slice(payload).ok() };
                        match g.insert(*id, r.lsn, vector, p) {
                            Ok(row) => rows[i].store(row, Ordering::Relaxed),
                            Err(e) => *lock(&err) = Some(e),
                        }
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(Error::Corrupt(format!("WAL replay: {e}")));
        }
        let mut map = lock(&self.ids);
        let segs = read(&self.segs);
        for (i, r) in records.iter().enumerate() {
            match &r.op {
                WalOp::Upsert { id, .. } => {
                    Self::commit_upsert(&mut map, &segs, g, *id, r.lsn, rows[i].load(Ordering::Relaxed))
                }
                WalOp::Delete { id } => {
                    Self::commit_delete(&mut map, &segs, *id, r.lsn);
                }
            }
        }
        map.retain(|_, s| matches!(s, Slot::Live { .. }));
        Ok(())
    }

    fn commit_upsert(map: &mut HashMap<u64, Slot>, segs: &Segments, g: &Growing, id: u64, lsn: u64, row: u32) {
        match map.get(&id).copied() {
            Some(s) if s.lsn() > lsn => g.tombstone(row),
            prev => {
                if let Some(Slot::Live { seg, row: old, .. }) = prev {
                    segs.tombstone(seg, old);
                }
                map.insert(id, Slot::Live { seg: g.id, row, lsn });
                g.publish(row);
            }
        }
    }

    fn commit_delete(map: &mut HashMap<u64, Slot>, segs: &Segments, id: u64, lsn: u64) -> bool {
        match map.get(&id).copied() {
            Some(s) if s.lsn() > lsn => false,
            Some(Slot::Live { seg, row, .. }) => {
                segs.tombstone(seg, row);
                map.insert(id, Slot::Dead { lsn });
                true
            }
            _ => {
                map.insert(id, Slot::Dead { lsn });
                false
            }
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Stops the collection: waits for a running checkpoint or compaction,
    /// then makes every later call fail. Used before the directory is removed.
    pub fn close(&self) {
        let _c = lock(&self.compacting);
        let _m = lock(&self.maint);
        let _g = write(&self.gate);
        self.closed.store(true, Ordering::Release);
    }

    fn check_ok(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::NotFound(format!("collection {:?} was dropped", self.name)));
        }
        if self.failed.load(Ordering::Acquire) {
            return Err(Error::Io(std::io::Error::other(
                "collection stopped after a failed checkpoint; restart the server to recover from disk",
            )));
        }
        Ok(())
    }

    fn validate_vector(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.config.dim {
            return invalid(format!(
                "vector has {} dimensions, collection {:?} has {}",
                v.len(),
                self.name,
                self.config.dim
            ));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return invalid("vector contains NaN or infinity");
        }
        Ok(())
    }

    /// Inserts or replaces points. Returns the LSN of the last one once all
    /// are durable and visible.
    pub fn upsert(&self, points: &[Point]) -> Result<u64> {
        self.check_ok()?;
        if points.is_empty() {
            return Ok(self.last_applied());
        }
        if points.len() > MAX_BATCH.min(self.config.segment_rows) {
            return invalid(format!("at most {} points per upsert", MAX_BATCH.min(self.config.segment_rows)));
        }
        let mut ops = Vec::with_capacity(points.len());
        for p in points {
            self.validate_vector(&p.vector)?;
            let payload = match &p.payload {
                None | Some(Value::Null) => Vec::new(),
                Some(v @ Value::Object(_)) => serde_json::to_vec(v).unwrap(),
                Some(_) => return invalid(format!("payload of point {} must be a JSON object", p.id)),
            };
            if payload.len() > MAX_PAYLOAD {
                return invalid(format!("payload of point {} exceeds {MAX_PAYLOAD} bytes", p.id));
            }
            ops.push(WalOp::Upsert { id: p.id, vector: p.vector.clone(), payload });
        }
        let n = points.len();
        let (_gate, growing) = loop {
            let gate = read(&self.gate);
            let growing = read(&self.segs).growing.clone();
            if growing.try_reserve(n) {
                break (gate, growing);
            }
            drop(gate);
            self.seal(Some(growing.id))?;
            self.check_ok()?;
        };
        let wal = read(&self.wal).clone();
        let first = wal.commit(&ops)?;
        let mut rows = Vec::with_capacity(n);
        for (i, p) in points.iter().enumerate() {
            let payload = p.payload.clone().filter(|v| !v.is_null());
            rows.push(growing.insert(p.id, first + i as u64, &p.vector, payload)?);
        }
        let mut map = lock(&self.ids);
        let segs = read(&self.segs);
        for (i, p) in points.iter().enumerate() {
            Self::commit_upsert(&mut map, &segs, &growing, p.id, first + i as u64, rows[i]);
        }
        let last = first + n as u64 - 1;
        self.applied_lsn.fetch_max(last, Ordering::AcqRel);
        Ok(last)
    }

    /// Deletes points; returns how many existed and the LSN of the last
    /// delete record.
    pub fn delete(&self, ids: &[u64]) -> Result<(usize, u64)> {
        self.check_ok()?;
        if ids.is_empty() {
            return Ok((0, self.last_applied()));
        }
        if ids.len() > MAX_BATCH {
            return invalid(format!("at most {MAX_BATCH} ids per delete"));
        }
        let _gate = read(&self.gate);
        let ops: Vec<WalOp> = ids.iter().map(|&id| WalOp::Delete { id }).collect();
        let first = read(&self.wal).clone().commit(&ops)?;
        let mut map = lock(&self.ids);
        let segs = read(&self.segs);
        let mut n = 0;
        for (i, &id) in ids.iter().enumerate() {
            if Self::commit_delete(&mut map, &segs, id, first + i as u64) {
                n += 1;
            }
        }
        let last = first + ids.len() as u64 - 1;
        self.applied_lsn.fetch_max(last, Ordering::AcqRel);
        Ok((n, last))
    }

    fn last_applied(&self) -> u64 {
        self.applied_lsn.load(Ordering::Acquire)
    }

    /// Looks points up by id (absent ids give `None`).
    pub fn get(&self, ids: &[u64], with_vector: bool) -> Vec<Option<Record>> {
        let map = lock(&self.ids);
        let segs = read(&self.segs);
        ids.iter()
            .map(|id| match map.get(id) {
                Some(Slot::Live { seg, row, .. }) => segs.record(*id, *seg, *row, with_vector),
                _ => None,
            })
            .collect()
    }

    /// Number of live points.
    pub fn len(&self) -> usize {
        lock(&self.ids).values().filter(|s| matches!(s, Slot::Live { .. })).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn search(&self, q: &Query<'_>) -> Result<SearchResult> {
        self.check_ok()?;
        self.validate_vector(q.vector)?;
        if q.k == 0 || q.k > 10_000 {
            return invalid("k must be between 1 and 10000");
        }
        let ef = q.ef.unwrap_or(self.config.ef_search).max(q.k);
        if ef > 100_000 {
            return invalid("ef must be at most 100000");
        }
        let prepared = self.config.metric.prepare(q.vector);
        let segs = read(&self.segs).clone();
        let sq = SegQuery {
            q: &prepared,
            k: q.k,
            ef,
            filter: q.filter,
            strategy: q.strategy,
            rerank: q.rerank.unwrap_or(self.config.rerank),
            brute_factor: self.config.brute_force_factor,
        };
        let mut hits: Vec<SegHit> = Vec::new();
        let mut plan = Vec::new();
        for s in &segs.sealed {
            let mut p = SegPlan::default();
            hits.extend(s.search(&sq, &mut p));
            plan.push(p);
        }
        let mut p = SegPlan::default();
        hits.extend(segs.growing.search(&sq, &mut p));
        plan.push(p);
        hits.sort_by(|a, b| a.dist.total_cmp(&b.dist).then(a.ext.cmp(&b.ext)));
        // A point being moved by an upsert can briefly be visible in two
        // segments; keep its best entry.
        let mut seen = HashSet::new();
        hits.retain(|h| seen.insert(h.ext));
        hits.truncate(q.k);
        let hits = hits
            .into_iter()
            .map(|h| {
                let rec = (q.with_payload || q.with_vector)
                    .then(|| segs.record(h.ext, h.seg, h.row, q.with_vector))
                    .flatten();
                let (payload, vector) = rec.map_or((None, None), |r| (r.payload, r.vector));
                Hit { id: h.ext, distance: h.dist, payload: if q.with_payload { payload } else { None }, vector }
            })
            .collect();
        Ok(SearchResult { hits, plan })
    }

    pub fn stats(&self) -> CollectionStats {
        let segs = read(&self.segs).clone();
        let mut out = Vec::new();
        for s in &segs.sealed {
            out.push(SegmentStats {
                id: s.id,
                kind: "sealed",
                rows: s.rows(),
                live: s.live_count(),
                bytes: s.file.file_bytes(),
                quantization: s.file.quantization().label(),
            });
        }
        let g = &segs.growing;
        out.push(SegmentStats {
            id: g.id,
            kind: "growing",
            rows: g.rows(),
            live: g.live_count(),
            bytes: g.index.memory_bytes(),
            quantization: "f32".into(),
        });
        let wal = read(&self.wal).clone();
        CollectionStats {
            name: self.name.clone(),
            config: self.config.clone(),
            points: self.len(),
            segments: out,
            wal_bytes: wal.size_bytes(),
            wal_syncs: wal.sync_count(),
            next_lsn: wal.next_lsn(),
        }
    }

    fn write_manifest(&self, segments: Vec<u64>, wal: u64, last_lsn: u64) -> Result<()> {
        let m = Manifest {
            version: 1,
            config: self.config.clone(),
            segments,
            wal,
            next_segment: self.next_seg.load(Ordering::Acquire),
            last_lsn,
        };
        write_atomic(&self.dir.join(MANIFEST), &serde_json::to_vec_pretty(&m).unwrap(), self.opts.sync)?;
        Ok(())
    }

    /// Seals the growing segment and starts a new WAL. Returns false when
    /// there was nothing to do.
    pub fn checkpoint(&self) -> Result<bool> {
        self.seal(None)
    }

    fn seal(&self, only: Option<u64>) -> Result<bool> {
        self.check_ok()?;
        let _m = lock(&self.maint);
        let _g = write(&self.gate);
        self.check_ok()?;
        let segs = read(&self.segs).clone();
        let g = segs.growing.clone();
        if only.is_some_and(|id| id != g.id) {
            return Ok(false);
        }
        let wal = read(&self.wal).clone();
        let rows = g.rows();
        let dirty: Vec<Arc<Sealed>> = segs.sealed.iter().filter(|s| s.dirty.load(Ordering::Acquire)).cloned().collect();
        let last_lsn = wal.next_lsn() - 1;
        if rows == 0 && dirty.is_empty() && last_lsn == self.manifest_lsn() {
            return Ok(false);
        }
        // Everything up to the manifest write can fail without harm: the old
        // manifest still describes a consistent state and the leftovers are
        // removed on the next open.
        let mut ids: Vec<u64> = segs.sealed.iter().map(|s| s.id).collect();
        if rows > 0 {
            let dead =
                g.write(&segment_path(&self.dir, g.id), self.config.quantization, self.opts.threads, self.opts.sync)?;
            if dead.iter().any(|&w| w != 0) {
                write_tombstones(&tombstone_path(&self.dir, g.id), rows, &dead, self.opts.sync)?;
            }
            ids.push(g.id);
        }
        for s in &dirty {
            s.persist_tombstones(&self.dir, self.opts.sync)?;
        }
        let next_gen = wal.generation() + 1;
        let new_wal = Wal::create(&wal_path(&self.dir, next_gen), next_gen, last_lsn + 1, self.opts.sync)?;
        let fresh_id = if rows > 0 { self.next_seg.fetch_add(1, Ordering::AcqRel) } else { g.id };
        self.write_manifest(ids, next_gen, last_lsn)?;
        self.set_manifest_lsn(last_lsn);
        // Past this point the manifest refers to the new WAL; failing to switch
        // to it in memory would lose writes, so the collection stops instead.
        let finish = || -> Result<()> {
            for s in &dirty {
                s.dirty.store(false, Ordering::Release);
            }
            let sealed = if rows > 0 {
                Some(Arc::new(Sealed::open(&self.dir, g.id, Some(g.take_payload_index()))?))
            } else {
                None
            };
            let fresh = if rows > 0 {
                Arc::new(Growing::new(
                    fresh_id,
                    self.config.dim,
                    self.config.metric,
                    self.config.graph_params(),
                    self.config.segment_rows,
                ))
            } else {
                g.clone()
            };
            let mut map = lock(&self.ids);
            map.retain(|_, s| matches!(s, Slot::Live { .. }));
            let mut st = write(&self.segs);
            if let Some(s) = sealed {
                st.sealed.push(s);
            }
            st.growing = fresh;
            drop(st);
            drop(map);
            *write(&self.wal) = Arc::new(new_wal);
            Ok(())
        };
        if let Err(e) = finish() {
            self.failed.store(true, Ordering::Release);
            return Err(e);
        }
        let _ = std::fs::remove_file(wal.path());
        let _ = sync_dir(&self.dir, self.opts.sync);
        Ok(true)
    }

    fn manifest_lsn(&self) -> u64 {
        self.manifest_lsn.load(Ordering::Acquire)
    }

    fn set_manifest_lsn(&self, v: u64) {
        self.manifest_lsn.store(v, Ordering::Release);
    }

    /// Fraction of sealed rows that are tombstoned, and the number of sealed
    /// segments: the inputs of the auto-compaction policy.
    pub fn garbage(&self) -> (f64, usize) {
        let segs = read(&self.segs);
        let rows: usize = segs.sealed.iter().map(|s| s.rows()).sum();
        let dead: usize = segs.sealed.iter().map(|s| s.rows() - s.live_count()).sum();
        (if rows == 0 { 0.0 } else { dead as f64 / rows as f64 }, segs.sealed.len())
    }

    /// Merges every sealed segment into one, dropping tombstoned rows and
    /// building a fresh graph. Writers and searches keep running during the
    /// build; writers pause only for the final swap.
    pub fn compact(&self) -> Result<CompactionReport> {
        self.check_ok()?;
        let _c = lock(&self.compacting);
        let t0 = std::time::Instant::now();
        // Only compaction removes sealed segments, so the inputs stay valid
        // while this lock is held, even if checkpoints add new segments.
        let inputs: Vec<Arc<Sealed>> = read(&self.segs).sealed.clone();
        if inputs.is_empty() || (inputs.len() == 1 && inputs[0].live_count() == inputs[0].rows()) {
            return Ok(CompactionReport::default());
        }
        let mut src = Vec::new();
        for (i, s) in inputs.iter().enumerate() {
            for r in 0..s.rows() as u32 {
                if s.is_live(r) {
                    src.push((i, r));
                }
            }
        }
        let n = src.len();
        let dim = self.config.dim;
        let out_id = self.next_seg.fetch_add(1, Ordering::AcqRel);
        let mut ids = Vec::with_capacity(n);
        let mut out = None;
        if n > 0 {
            let mut vectors = Vec::with_capacity(n * dim);
            let mut lsns = Vec::with_capacity(n);
            let mut payloads = Vec::with_capacity(n);
            for &(i, r) in &src {
                let f = &inputs[i].file;
                vectors.extend_from_slice(f.vector(r));
                ids.push(f.ids()[r as usize]);
                lsns.push(f.lsns()[r as usize]);
                payloads.push(f.payload(r).to_vec());
            }
            let idx = HnswIndex::new(dim, self.config.metric, self.config.graph_params(), n);
            idx.insert_batch(&vectors, self.opts.threads)?;
            let graph = idx.graph().freeze(n);
            drop(idx);
            let codes = Codes::build(
                self.config.quantization,
                &vectors,
                dim,
                self.config.metric,
                self.config.seed,
                self.opts.threads,
            )
            .map_err(Error::Invalid)?;
            write_segment(
                &segment_path(&self.dir, out_id),
                &SegmentWrite {
                    dim,
                    metric: self.config.metric,
                    params: self.config.graph_params(),
                    ids: &ids,
                    lsns: &lsns,
                    vectors: &vectors,
                    graph: &graph,
                    payloads: &payloads,
                    codes: codes.as_ref(),
                },
                self.opts.sync,
            )?;
            out = Some(Arc::new(Sealed::open(&self.dir, out_id, None)?));
        }
        // Swap: pause writers, carry over deletes made during the build.
        let _m = lock(&self.maint);
        let _g = write(&self.gate);
        self.check_ok()?;
        let mut dead = Bitset::new(n);
        for (j, &(i, r)) in src.iter().enumerate() {
            if !inputs[i].is_live(r) {
                dead.set(j);
            }
        }
        if n > 0 && dead.count() > 0 {
            write_tombstones(&tombstone_path(&self.dir, out_id), n, dead.words(), self.opts.sync)?;
        }
        let input_ids: HashSet<u64> = inputs.iter().map(|s| s.id).collect();
        let current = read(&self.segs).sealed.clone();
        let mut list: Vec<Arc<Sealed>> = Vec::new();
        let mut placed = false;
        for s in current {
            if input_ids.contains(&s.id) {
                if !placed {
                    if let Some(o) = &out {
                        list.push(o.clone());
                    }
                    placed = true;
                }
            } else {
                list.push(s);
            }
        }
        let wal_gen = read(&self.wal).generation();
        self.write_manifest(list.iter().map(|s| s.id).collect(), wal_gen, self.manifest_lsn())?;
        let mut map = lock(&self.ids);
        if let Some(o) = &out {
            for (j, &(i, r)) in src.iter().enumerate() {
                if dead.get(j) {
                    o.tombstone(j as u32);
                    continue;
                }
                if let Some(Slot::Live { seg, row, lsn }) = map.get(&ids[j]).copied()
                    && seg == inputs[i].id
                    && row == r
                {
                    map.insert(ids[j], Slot::Live { seg: o.id, row: j as u32, lsn });
                }
            }
            o.dirty.store(false, Ordering::Release);
        }
        write(&self.segs).sealed = list;
        drop(map);
        for s in &inputs {
            let _ = std::fs::remove_file(segment_path(&self.dir, s.id));
            let _ = std::fs::remove_file(tombstone_path(&self.dir, s.id));
        }
        let _ = sync_dir(&self.dir, self.opts.sync);
        Ok(CompactionReport {
            inputs: inputs.iter().map(|s| s.id).collect(),
            output: out.map(|o| o.id),
            rows_in: inputs.iter().map(|s| s.rows()).sum(),
            rows_out: n - dead.count(),
            seconds: t0.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests;
