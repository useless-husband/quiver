//! The HNSW graph (Malkov & Yashunin, "Efficient and robust approximate
//! nearest neighbor search using Hierarchical Navigable Small World graphs",
//! TPAMI 2018).
//!
//! The graph only stores topology. Distances come from a [`Space`], so the
//! same code searches full-precision vectors, int8 codes or PQ codes, and the
//! same search routine runs over the concurrent in-memory [`Graph`] and over a
//! [`FrozenGraph`] whose arrays live in a memory-mapped segment file.
//!
//! # Concurrency
//!
//! Inserts and searches run concurrently on a shared `&Graph`:
//!
//! * Each link list is `[count, id_1, .., id_max]` of `AtomicU32`. Writers hold
//!   the node's [`SpinLock`]; readers take no lock. A writer stores ids with
//!   `Release` and then the count with `Release`; a reader loads the count with
//!   `Acquire`, the ids with `Relaxed`, then issues an `Acquire` fence, so it
//!   synchronises with every writer whose ids it saw. A reader can observe a
//!   list half-way through a rewrite; every slot still holds the id of a node
//!   that was fully initialised before it was first linked, so the worst case
//!   is a slightly stale neighbourhood, which approximate search tolerates.
//! * A node's level and upper-layer lists are written before the node is
//!   linked anywhere, so anyone who reaches it through a link sees them.
//! * The entry point is one `AtomicU64` (level << 32 | id). An insert whose
//!   level exceeds the current top holds `top_lock` for its whole insertion, so
//!   two inserts never race to raise the top level (the hnswlib scheme).
//!
//! Callers must write a row's vector before calling [`Graph::insert`] for it;
//! the `Release` stores above then publish the vector as well.

use super::lock::SpinLock;
use super::visited::VisitedSet;
use super::{BuildSpace, Neighbor, Space};
use crate::rng::splitmix64;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Mutex, OnceLock};

/// Highest layer a node can be assigned (the expected top for a billion nodes
/// at M = 16 is about 7).
pub const MAX_LEVEL: usize = 15;
const NOT_INSERTED: u8 = u8::MAX;
const EMPTY: u64 = u64::MAX;

/// Graph construction parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphParams {
    /// Out-degree bound on the upper layers; layer 0 allows `2 * m`.
    pub m: usize,
    /// Beam width while inserting.
    pub ef_construction: usize,
    /// Seed for level assignment.
    pub seed: u64,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams {
            m: 16,
            ef_construction: 200,
            seed: 0x5157_4952,
        }
    }
}

impl GraphParams {
    pub fn m0(&self) -> usize {
        2 * self.m
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(2..=128).contains(&self.m) {
            return Err(format!("m must be between 2 and 128, got {}", self.m));
        }
        if self.ef_construction == 0 || self.ef_construction > 10_000 {
            return Err(format!(
                "ef_construction must be between 1 and 10000, got {}",
                self.ef_construction
            ));
        }
        Ok(())
    }
}

/// Read access to a graph's topology, shared by the mutable and frozen forms.
pub trait Topology: Sync {
    /// Entry node and its level, or `None` for an empty graph.
    fn entry(&self) -> Option<(u32, usize)>;
    /// Copies the neighbours of `id` on `level` into `out`.
    fn neighbors(&self, id: u32, level: usize, out: &mut Vec<u32>);
    /// The level of `id`, or `None` if it is not in the graph.
    fn level(&self, id: u32) -> Option<usize>;
    /// Every id is below this bound.
    fn capacity(&self) -> usize;
    /// Out-degree bound on `level`.
    fn max_degree(&self, level: usize) -> usize;
}

/// Reusable per-search state, so that searches do not allocate.
pub struct Scratch {
    visited: VisitedSet,
    cand: BinaryHeap<Reverse<Neighbor>>,
    res: BinaryHeap<Neighbor>,
    nbrs: Vec<u32>,
    sel: Vec<Neighbor>,
    tmp: Vec<Neighbor>,
    /// Distance computations since the last reset, for benchmarks.
    pub dist_count: u64,
}

impl Scratch {
    pub fn new(capacity: usize) -> Scratch {
        Scratch {
            visited: VisitedSet::new(capacity),
            cand: BinaryHeap::new(),
            res: BinaryHeap::new(),
            nbrs: Vec::new(),
            sel: Vec::new(),
            tmp: Vec::new(),
            dist_count: 0,
        }
    }
}

/// Greedy walk on one layer: move to the closest neighbour until none is closer.
fn greedy<T: Topology, S: Space>(
    t: &T,
    s: &S,
    q: &S::Query,
    mut cur: Neighbor,
    level: usize,
    sc: &mut Scratch,
) -> Neighbor {
    loop {
        let mut changed = false;
        t.neighbors(cur.id, level, &mut sc.nbrs);
        sc.dist_count += sc.nbrs.len() as u64;
        for &n in &sc.nbrs {
            let d = s.dist(q, n);
            if d < cur.dist {
                cur = Neighbor::new(n, d);
                changed = true;
            }
        }
        if !changed {
            return cur;
        }
    }
}

/// Beam search on one layer (Algorithm 2 of the paper), returning up to `ef`
/// accepted nodes, closest first.
///
/// Nodes rejected by `accept` (tombstones, rows outside a filter) are still
/// expanded, because they carry the graph's connectivity, but never enter the
/// result set. While fewer than `ef` accepted nodes are known the search keeps
/// expanding, which is what makes selective filters expensive (see
/// `collection.rs` for the brute-force fallback).
#[allow(clippy::too_many_arguments)]
pub fn search_layer<T, S, F>(
    t: &T,
    s: &S,
    q: &S::Query,
    eps: &[Neighbor],
    ef: usize,
    level: usize,
    accept: &F,
    sc: &mut Scratch,
) -> Vec<Neighbor>
where
    T: Topology,
    S: Space,
    F: Fn(u32) -> bool,
{
    let ef = ef.max(1);
    sc.visited.reset(t.capacity());
    sc.cand.clear();
    sc.res.clear();
    for &e in eps {
        if sc.visited.insert(e.id) {
            sc.cand.push(Reverse(e));
            if accept(e.id) {
                sc.res.push(e);
            }
        }
    }
    while sc.res.len() > ef {
        sc.res.pop();
    }
    let mut bound = if sc.res.len() >= ef {
        sc.res.peek().map_or(f32::INFINITY, |n| n.dist)
    } else {
        f32::INFINITY
    };
    while let Some(Reverse(c)) = sc.cand.pop() {
        if c.dist > bound {
            break;
        }
        t.neighbors(c.id, level, &mut sc.nbrs);
        for &n in &sc.nbrs {
            sc.visited.prefetch(n);
        }
        let len = sc.nbrs.len();
        for j in 0..len {
            let n = sc.nbrs[j];
            if j + 1 < len {
                s.prefetch(sc.nbrs[j + 1]);
            }
            if !sc.visited.insert(n) {
                continue;
            }
            let d = s.dist(q, n);
            sc.dist_count += 1;
            if d < bound {
                sc.cand.push(Reverse(Neighbor::new(n, d)));
                if accept(n) {
                    sc.res.push(Neighbor::new(n, d));
                    if sc.res.len() > ef {
                        sc.res.pop();
                    }
                    if sc.res.len() >= ef {
                        bound = sc.res.peek().map_or(f32::INFINITY, |n| n.dist);
                    }
                }
            }
        }
    }
    let mut out: Vec<Neighbor> = sc.res.drain().collect();
    out.sort_unstable();
    out
}

/// k-NN search: greedy descent through the upper layers, then a beam of width
/// `max(ef, k)` on layer 0.
pub fn knn<T, S, F>(
    t: &T,
    s: &S,
    q: &S::Query,
    k: usize,
    ef: usize,
    accept: &F,
    sc: &mut Scratch,
) -> Vec<Neighbor>
where
    T: Topology,
    S: Space,
    F: Fn(u32) -> bool,
{
    let Some((ep, top)) = t.entry() else {
        return Vec::new();
    };
    if k == 0 {
        return Vec::new();
    }
    let mut cur = Neighbor::new(ep, s.dist(q, ep));
    sc.dist_count += 1;
    for level in (1..=top).rev() {
        cur = greedy(t, s, q, cur, level, sc);
    }
    let mut r = search_layer(t, s, q, &[cur], ef.max(k), 0, accept, sc);
    r.truncate(k);
    r
}

/// The neighbour-selection heuristic (Algorithm 4 without the optional
/// extensions, as in hnswlib): walk candidates from closest to farthest and
/// keep one only if it is closer to the base than to every neighbour already
/// kept. This favours neighbours in different directions, which keeps
/// clustered data navigable where "keep the M closest" would not.
pub fn select_heuristic<S: BuildSpace>(
    s: &S,
    cands: &[Neighbor],
    m: usize,
    out: &mut Vec<Neighbor>,
) {
    out.clear();
    if cands.len() <= m {
        out.extend_from_slice(cands);
        return;
    }
    for &c in cands {
        if out.len() >= m {
            break;
        }
        if out.iter().all(|kept| s.between(c.id, kept.id) >= c.dist) {
            out.push(c);
        }
    }
}

/// The concurrent, growable-by-insert HNSW graph over row ids `0..capacity`.
pub struct Graph {
    params: GraphParams,
    m: usize,
    m0: usize,
    ml: f64,
    cap: usize,
    levels: Box<[AtomicU8]>,
    links0: Box<[AtomicU32]>,
    upper: Box<[OnceLock<Box<[AtomicU32]>>]>,
    locks: Box<[SpinLock]>,
    entry: AtomicU64,
    top_lock: Mutex<()>,
    rng: AtomicU64,
    len: AtomicUsize,
}

#[inline]
fn pack(id: u32, level: usize) -> u64 {
    (level as u64) << 32 | id as u64
}

#[inline]
fn unpack(e: u64) -> (u32, usize) {
    (e as u32, (e >> 32) as usize)
}

impl Graph {
    pub fn new(params: GraphParams, capacity: usize) -> Graph {
        params.validate().expect("invalid graph parameters");
        assert!(capacity < u32::MAX as usize, "capacity must fit in u32 ids");
        let m = params.m;
        let m0 = params.m0();
        Graph {
            params,
            m,
            m0,
            ml: 1.0 / (m as f64).ln(),
            cap: capacity,
            levels: (0..capacity).map(|_| AtomicU8::new(NOT_INSERTED)).collect(),
            links0: (0..capacity * (1 + m0))
                .map(|_| AtomicU32::new(0))
                .collect(),
            upper: (0..capacity).map(|_| OnceLock::new()).collect(),
            locks: (0..capacity).map(|_| SpinLock::new()).collect(),
            entry: AtomicU64::new(EMPTY),
            top_lock: Mutex::new(()),
            rng: AtomicU64::new(0),
            len: AtomicUsize::new(0),
        }
    }

    pub fn params(&self) -> &GraphParams {
        &self.params
    }

    /// Number of inserted nodes.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes used by the graph structure (excluding vectors).
    pub fn memory_bytes(&self) -> usize {
        let upper: usize = self
            .upper
            .iter()
            .filter_map(|u| u.get())
            .map(|u| u.len() * 4)
            .sum();
        self.cap * (1 + 1 + 24 + 4 * (1 + self.m0)) + upper
    }

    /// Draws a level from the geometric distribution with parameter
    /// `1 / ln(M)`. Deterministic for a given seed and insertion order.
    fn random_level(&self) -> usize {
        let i = self.rng.fetch_add(1, Ordering::Relaxed);
        let x = splitmix64(self.params.seed ^ splitmix64(i));
        let u = ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        ((-u.ln() * self.ml) as usize).min(MAX_LEVEL)
    }

    #[inline]
    fn list(&self, id: u32, level: usize) -> Option<&[AtomicU32]> {
        if level == 0 {
            let w = 1 + self.m0;
            let s = id as usize * w;
            self.links0.get(s..s + w)
        } else {
            let w = 1 + self.m;
            let u = self.upper.get(id as usize)?.get()?;
            u.get((level - 1) * w..level * w)
        }
    }

    /// Writes `ns` as the neighbours of `id` on `level`. The caller holds the
    /// node's lock.
    fn store_list(l: &[AtomicU32], ns: &[Neighbor]) {
        debug_assert!(ns.len() < l.len());
        for (j, n) in ns.iter().enumerate() {
            l[1 + j].store(n.id, Ordering::Release);
        }
        l[0].store(ns.len() as u32, Ordering::Release);
    }

    /// Inserts row `id`, whose vector the caller has already written and which
    /// `q` describes. Each id may be inserted once.
    pub fn insert<S: BuildSpace>(&self, s: &S, id: u32, q: &S::Query, sc: &mut Scratch) {
        assert!(
            (id as usize) < self.cap,
            "id {id} beyond capacity {}",
            self.cap
        );
        let level = self.random_level();
        self.levels[id as usize]
            .compare_exchange(
                NOT_INSERTED,
                level as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .unwrap_or_else(|_| panic!("row {id} inserted twice"));
        if level > 0 {
            let lists = (0..level * (1 + self.m))
                .map(|_| AtomicU32::new(0))
                .collect();
            if self.upper[id as usize].set(lists).is_err() {
                panic!("row {id} already has upper layers");
            }
        }

        let mut ep = self.entry.load(Ordering::Acquire);
        let mut _top_guard = None;
        if ep == EMPTY || level > unpack(ep).1 {
            _top_guard = Some(self.top_lock.lock().unwrap_or_else(|e| e.into_inner()));
            ep = self.entry.load(Ordering::Acquire);
            if ep == EMPTY {
                self.entry.store(pack(id, level), Ordering::Release);
                self.len.fetch_add(1, Ordering::AcqRel);
                return;
            }
            if level <= unpack(ep).1 {
                _top_guard = None;
            }
        }
        let (ep_id, top) = unpack(ep);
        let mut cur = Neighbor::new(ep_id, s.dist(q, ep_id));
        for lc in (level + 1..=top).rev() {
            cur = greedy(self, s, q, cur, lc, sc);
        }
        let mut eps = vec![cur];
        // The new node can already be reachable on this layer: once it is
        // wired on a higher layer, a concurrent insert may find it there and
        // link to it on this layer. Excluding it from the results keeps it
        // from selecting itself as a neighbour (a self loop).
        let not_self = |x: u32| x != id;
        for lc in (0..=level.min(top)).rev() {
            let w = search_layer(
                self,
                s,
                q,
                &eps,
                self.params.ef_construction,
                lc,
                &not_self,
                sc,
            );
            let mut sel = std::mem::take(&mut sc.sel);
            select_heuristic(s, &w, self.m, &mut sel);
            self.set_own_links(s, id, lc, &sel, sc);
            for nb in &sel {
                self.connect(s, nb.id, id, nb.dist, lc, sc);
            }
            sc.sel = sel;
            eps = w;
        }
        if _top_guard.is_some() && level > top {
            self.entry.store(pack(id, level), Ordering::Release);
        }
        self.len.fetch_add(1, Ordering::AcqRel);
    }

    /// Sets the new node's own list. Another insert may already have linked
    /// back to it (it can be found through a higher layer before this layer is
    /// wired), so existing entries are merged rather than overwritten.
    fn set_own_links<S: BuildSpace>(
        &self,
        s: &S,
        id: u32,
        level: usize,
        sel: &[Neighbor],
        sc: &mut Scratch,
    ) {
        let Some(l) = self.list(id, level) else {
            return;
        };
        let max = self.max_degree(level);
        let _g = self.locks[id as usize].lock();
        let n = (l[0].load(Ordering::Relaxed) as usize).min(max);
        if n == 0 {
            Self::store_list(l, &sel[..sel.len().min(max)]);
            return;
        }
        let mut all: Vec<Neighbor> = sel.to_vec();
        for j in 0..n {
            let x = l[1 + j].load(Ordering::Relaxed);
            if !all.iter().any(|a| a.id == x) {
                all.push(Neighbor::new(x, s.between(id, x)));
            }
        }
        all.sort_unstable();
        let mut out = std::mem::take(&mut sc.tmp);
        select_heuristic(s, &all, max, &mut out);
        Self::store_list(l, &out);
        sc.tmp = out;
    }

    /// Adds `q` to the neighbours of `e` on `level`, shrinking the list with
    /// the heuristic when it is full.
    fn connect<S: BuildSpace>(
        &self,
        s: &S,
        e: u32,
        q: u32,
        d_eq: f32,
        level: usize,
        sc: &mut Scratch,
    ) {
        let Some(l) = self.list(e, level) else { return };
        let max = self.max_degree(level);
        let _g = self.locks[e as usize].lock();
        let n = (l[0].load(Ordering::Relaxed) as usize).min(max);
        for j in 0..n {
            if l[1 + j].load(Ordering::Relaxed) == q {
                return;
            }
        }
        if n < max {
            l[1 + n].store(q, Ordering::Release);
            l[0].store(n as u32 + 1, Ordering::Release);
            return;
        }
        let mut cands = std::mem::take(&mut sc.tmp);
        cands.clear();
        for j in 0..n {
            let x = l[1 + j].load(Ordering::Relaxed);
            cands.push(Neighbor::new(x, s.between(e, x)));
        }
        cands.push(Neighbor::new(q, d_eq));
        cands.sort_unstable();
        let mut out = Vec::with_capacity(max);
        select_heuristic(s, &cands, max, &mut out);
        Self::store_list(l, &out);
        sc.tmp = cands;
    }

    /// Copies the topology into flat arrays for a segment file.
    pub fn freeze(&self, count: usize) -> FrozenParts {
        assert!(count <= self.cap);
        let mut levels = vec![0u8; count];
        let mut links0 = vec![0u32; count * (1 + self.m0)];
        let mut upper_off = vec![0u32; count];
        let mut upper = Vec::new();
        let mut buf = Vec::new();
        for id in 0..count as u32 {
            let lv = self.levels[id as usize].load(Ordering::Acquire);
            assert!(lv != NOT_INSERTED, "freeze: row {id} was never inserted");
            levels[id as usize] = lv;
            self.neighbors(id, 0, &mut buf);
            let s = id as usize * (1 + self.m0);
            links0[s] = buf.len() as u32;
            links0[s + 1..s + 1 + buf.len()].copy_from_slice(&buf);
            if lv > 0 {
                upper_off[id as usize] = upper.len() as u32;
                for level in 1..=lv as usize {
                    self.neighbors(id, level, &mut buf);
                    let start = upper.len();
                    upper.resize(start + 1 + self.m, 0);
                    upper[start] = buf.len() as u32;
                    upper[start + 1..start + 1 + buf.len()].copy_from_slice(&buf);
                }
            }
        }
        let entry = self.entry.load(Ordering::Acquire);
        let entry = if entry == EMPTY {
            None
        } else {
            Some(unpack(entry))
        };
        FrozenParts {
            m: self.m,
            entry,
            levels,
            links0,
            upper_off,
            upper,
        }
    }

    /// Rebuilds a mutable graph from frozen parts (used when a sealed segment
    /// must accept inserts again, and by tests).
    pub fn thaw(params: GraphParams, parts: &FrozenGraph<'_>, capacity: usize) -> Graph {
        let g = Graph::new(params, capacity.max(parts.count()));
        assert_eq!(params.m, parts.m, "thaw: m differs");
        let mut buf = Vec::new();
        for id in 0..parts.count() as u32 {
            let lv = parts.level(id).unwrap();
            g.levels[id as usize].store(lv as u8, Ordering::Relaxed);
            if lv > 0 {
                let _ = g.upper[id as usize]
                    .set((0..lv * (1 + g.m)).map(|_| AtomicU32::new(0)).collect());
            }
            for level in 0..=lv {
                parts.neighbors(id, level, &mut buf);
                let ns: Vec<Neighbor> = buf.iter().map(|&x| Neighbor::new(x, 0.0)).collect();
                Self::store_list(g.list(id, level).unwrap(), &ns);
            }
        }
        if let Some((e, l)) = parts.entry() {
            g.entry.store(pack(e, l), Ordering::Release);
        }
        g.rng.store(parts.count() as u64, Ordering::Relaxed);
        g.len.store(parts.count(), Ordering::Release);
        g
    }
}

impl Topology for Graph {
    #[inline]
    fn entry(&self) -> Option<(u32, usize)> {
        let e = self.entry.load(Ordering::Acquire);
        if e == EMPTY { None } else { Some(unpack(e)) }
    }

    #[inline]
    fn neighbors(&self, id: u32, level: usize, out: &mut Vec<u32>) {
        out.clear();
        let Some(l) = self.list(id, level) else {
            return;
        };
        let n = (l[0].load(Ordering::Acquire) as usize).min(l.len() - 1);
        out.extend(l[1..1 + n].iter().map(|x| x.load(Ordering::Relaxed)));
        fence(Ordering::Acquire);
    }

    fn level(&self, id: u32) -> Option<usize> {
        let l = self.levels.get(id as usize)?.load(Ordering::Acquire);
        if l == NOT_INSERTED {
            None
        } else {
            Some(l as usize)
        }
    }

    fn capacity(&self) -> usize {
        self.cap
    }

    fn max_degree(&self, level: usize) -> usize {
        if level == 0 { self.m0 } else { self.m }
    }
}

/// An owned frozen graph, as written to a segment file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenParts {
    pub m: usize,
    pub entry: Option<(u32, usize)>,
    pub levels: Vec<u8>,
    pub links0: Vec<u32>,
    pub upper_off: Vec<u32>,
    pub upper: Vec<u32>,
}

impl FrozenParts {
    pub fn view(&self) -> FrozenGraph<'_> {
        FrozenGraph {
            m: self.m,
            m0: 2 * self.m,
            entry: self.entry,
            levels: &self.levels,
            links0: &self.links0,
            upper_off: &self.upper_off,
            upper: &self.upper,
        }
    }
}

/// A read-only graph over borrowed arrays (for example a mapped file).
#[derive(Clone, Copy, Debug)]
pub struct FrozenGraph<'a> {
    pub m: usize,
    pub m0: usize,
    pub entry: Option<(u32, usize)>,
    pub levels: &'a [u8],
    pub links0: &'a [u32],
    pub upper_off: &'a [u32],
    pub upper: &'a [u32],
}

impl FrozenGraph<'_> {
    pub fn count(&self) -> usize {
        self.levels.len()
    }

    /// Checks that the arrays are mutually consistent, so that searching a
    /// corrupted file cannot index out of bounds.
    pub fn validate_shape(&self) -> Result<(), String> {
        let n = self.count();
        if self.links0.len() != n * (1 + self.m0) || self.upper_off.len() != n {
            return Err("graph arrays have inconsistent lengths".into());
        }
        if let Some((e, l)) = self.entry {
            if e as usize >= n || self.levels[e as usize] as usize != l {
                return Err("graph entry point is invalid".into());
            }
        } else if n > 0 {
            return Err("non-empty graph without an entry point".into());
        }
        let check_list = |id: usize, level: usize, l: &[u32], max: usize| -> Result<(), String> {
            let c = l[0] as usize;
            if c > max {
                return Err(format!("row {id} has {c} links on layer {level}"));
            }
            if let Some(&x) = l[1..1 + c].iter().find(|&&x| x as usize >= n) {
                return Err(format!(
                    "row {id} layer {level} links to row {x} beyond {n}"
                ));
            }
            Ok(())
        };
        for id in 0..n {
            let lv = self.levels[id] as usize;
            if lv > MAX_LEVEL {
                return Err(format!("row {id} has level {lv}"));
            }
            let s = id * (1 + self.m0);
            check_list(id, 0, &self.links0[s..s + 1 + self.m0], self.m0)?;
            if lv > 0 {
                let off = self.upper_off[id] as usize;
                let end = off
                    .checked_add(lv * (1 + self.m))
                    .ok_or("upper offset overflow")?;
                if end > self.upper.len() {
                    return Err(format!("row {id} upper links out of range"));
                }
                for level in 1..=lv {
                    let s = off + (level - 1) * (1 + self.m);
                    check_list(id, level, &self.upper[s..s + 1 + self.m], self.m)?;
                }
            }
        }
        Ok(())
    }
}

impl Topology for FrozenGraph<'_> {
    #[inline]
    fn entry(&self) -> Option<(u32, usize)> {
        self.entry
    }

    #[inline]
    fn neighbors(&self, id: u32, level: usize, out: &mut Vec<u32>) {
        out.clear();
        let id = id as usize;
        let l = if level == 0 {
            let s = id * (1 + self.m0);
            &self.links0[s..s + 1 + self.m0]
        } else {
            if level > self.levels[id] as usize {
                return;
            }
            let s = self.upper_off[id] as usize + (level - 1) * (1 + self.m);
            &self.upper[s..s + 1 + self.m]
        };
        let n = (l[0] as usize).min(l.len() - 1);
        out.extend_from_slice(&l[1..1 + n]);
    }

    fn level(&self, id: u32) -> Option<usize> {
        self.levels.get(id as usize).map(|&l| l as usize)
    }

    fn capacity(&self) -> usize {
        self.levels.len()
    }

    fn max_degree(&self, level: usize) -> usize {
        if level == 0 { self.m0 } else { self.m }
    }
}

/// Structural invariants of an HNSW graph, checked by tests and by
/// `quiver check`: no self loops, no duplicate or dangling links, degrees
/// within bounds, every link points to a node that exists on that layer, and
/// the entry point sits on the top layer.
pub fn check_invariants<T: Topology>(
    t: &T,
    ids: impl Iterator<Item = u32>,
) -> Result<GraphStats, String> {
    let mut stats = GraphStats::default();
    let mut buf = Vec::new();
    let mut max_level = 0;
    for id in ids {
        let Some(lv) = t.level(id) else {
            return Err(format!("row {id} is not in the graph"));
        };
        stats.nodes += 1;
        max_level = max_level.max(lv);
        for level in 0..=lv {
            t.neighbors(id, level, &mut buf);
            if buf.len() > t.max_degree(level) {
                return Err(format!(
                    "row {id} layer {level}: degree {} > {}",
                    buf.len(),
                    t.max_degree(level)
                ));
            }
            let mut sorted = buf.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != buf.len() {
                return Err(format!("row {id} layer {level}: duplicate links {buf:?}"));
            }
            for &x in &buf {
                if x == id {
                    return Err(format!("row {id} layer {level}: self loop"));
                }
                match t.level(x) {
                    None => return Err(format!("row {id} layer {level}: dangling link to {x}")),
                    Some(xl) if xl < level => {
                        return Err(format!(
                            "row {id} layer {level}: link to {x} which only reaches layer {xl}"
                        ));
                    }
                    _ => {}
                }
            }
            if level == 0 {
                stats.edges0 += buf.len();
            }
        }
    }
    match t.entry() {
        None if stats.nodes > 0 => return Err("graph has nodes but no entry point".into()),
        Some((e, l)) => {
            if t.level(e) != Some(l) {
                return Err(format!(
                    "entry {e} claims level {l} but has {:?}",
                    t.level(e)
                ));
            }
            if l != max_level {
                return Err(format!("entry level {l} is not the top level {max_level}"));
            }
        }
        None => {}
    }
    stats.top_level = max_level;
    Ok(stats)
}

/// Fraction of `ids` reachable from the entry point on layer 0.
pub fn reachable_fraction<T: Topology>(t: &T, n: usize) -> f64 {
    let Some((e, _)) = t.entry() else { return 1.0 };
    let mut seen = vec![false; t.capacity()];
    let mut stack = vec![e];
    seen[e as usize] = true;
    let mut count = 0usize;
    let mut buf = Vec::new();
    while let Some(x) = stack.pop() {
        count += 1;
        t.neighbors(x, 0, &mut buf);
        for &y in &buf {
            if !seen[y as usize] {
                seen[y as usize] = true;
                stack.push(y);
            }
        }
    }
    count as f64 / n.max(1) as f64
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges0: usize,
    pub top_level: usize,
}
