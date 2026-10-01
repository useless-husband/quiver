use super::*;
use crate::index::flat::{FlatIndex, ground_truth};
use crate::index::graph::{FrozenGraph, check_invariants, reachable_fraction};
use crate::index::recall;
use crate::rng::Rng;
use crate::testkit::{clustered, property};
use std::sync::Arc;

fn params(m: usize, efc: usize) -> GraphParams {
    GraphParams {
        m,
        ef_construction: efc,
        seed: 42,
    }
}

fn mean_recall(idx: &HnswIndex, queries: &[f32], truth: &[Vec<u32>], k: usize, ef: usize) -> f64 {
    let dim = idx.dim();
    let mut total = 0.0;
    for (qi, t) in truth.iter().enumerate() {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let got: Vec<u32> = idx.search(q, k, ef).unwrap().iter().map(|n| n.id).collect();
        total += recall(&got, t, k);
    }
    total / truth.len() as f64
}

#[test]
fn recall_on_clustered_data_for_every_metric() {
    let mut rng = Rng::new(11);
    let (n, dim, nq) = (12_000, 24, 200);
    let mut base = clustered(&mut rng, n + nq, dim, 40);
    let queries = base.split_off(n * dim);
    for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
        let idx = HnswIndex::new(dim, metric, params(16, 200), n);
        idx.insert_batch(&base, 4).unwrap();
        let stats = idx.check().unwrap();
        assert_eq!(stats.nodes, n);
        let truth = ground_truth(metric, dim, &base, &queries, 10, 4);
        let r64 = mean_recall(&idx, &queries, &truth, 10, 64);
        let r256 = mean_recall(&idx, &queries, &truth, 10, 256);
        eprintln!("{metric:?}: recall@10 ef=64 {r64:.4} ef=256 {r256:.4}");
        // Inner-product search is not a metric space; HNSW is known to do
        // worse on it, hence the lower bar.
        let bar = if metric == Metric::Ip { 0.85 } else { 0.95 };
        assert!(r64 >= bar, "{metric:?} recall@10 at ef=64 is {r64}");
        assert!(
            r256 >= bar + 0.03,
            "{metric:?} recall@10 at ef=256 is {r256}"
        );
        // Under inner product a low-norm point can be nobody's neighbour, so
        // only the metric spaces must stay (almost) fully connected.
        let reach = reachable_fraction(idx.graph(), n);
        if metric != Metric::Ip {
            assert!(reach > 0.999, "{metric:?}: reachable {reach}");
        }
    }
}

#[test]
fn results_are_sorted_unique_and_bounded() {
    property("search output shape", 60, |rng| {
        let n = rng.range(1, 400);
        let dim = rng.range(1, 9);
        let m = rng.range(2, 12);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.normal()).collect();
        let idx = HnswIndex::new(dim, Metric::L2, params(m, rng.range(4, 64)), n);
        idx.insert_batch(&data, 1).map_err(|e| e.to_string())?;
        idx.check()?;
        let q: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
        let k = rng.range(1, 20);
        let r = idx.search(&q, k, rng.range(1, 100)).unwrap();
        if r.len() > k.min(n) {
            return Err(format!("{} results for k={k}", r.len()));
        }
        if !r.windows(2).all(|w| w[0] < w[1]) {
            return Err(format!("not strictly sorted: {r:?}"));
        }
        // With ef >= n every reachable node is examined, so the answer is exact
        // whenever the graph is connected.
        if reachable_fraction(idx.graph(), n) == 1.0 {
            let exact = idx.brute_force(&q, k, &|_| true).unwrap();
            let r = idx.search(&q, k, n).unwrap();
            if r.iter().map(|x| x.id).collect::<Vec<_>>()
                != exact.iter().map(|x| x.id).collect::<Vec<_>>()
            {
                return Err(format!("ef=n search {r:?} differs from exact {exact:?}"));
            }
        }
        Ok(())
    });
}

#[test]
fn concurrent_inserts_and_searches() {
    let mut rng = Rng::new(5);
    let (n, dim) = (16_000, 16);
    let mut all = clustered(&mut rng, n + 200, dim, 30);
    let queries = all.split_off(n * dim);
    let base = Arc::new(all);
    let idx = Arc::new(HnswIndex::new(dim, Metric::L2, params(12, 100), n));
    let done = Arc::new(AtomicBool::new(false));
    let searchers: Vec<_> = (0..2)
        .map(|t| {
            let (idx, base, done) = (idx.clone(), base.clone(), done.clone());
            std::thread::spawn(move || {
                let mut rng = Rng::new(100 + t);
                let mut searches = 0;
                while !done.load(Ordering::Acquire) {
                    let qi = rng.below(n);
                    let r = idx.search(&base[qi * dim..(qi + 1) * dim], 10, 32).unwrap();
                    assert!(r.windows(2).all(|w| w[0] < w[1]));
                    for nb in &r {
                        assert!(
                            idx.vector(nb.id).is_some(),
                            "search returned unpublished row {}",
                            nb.id
                        );
                    }
                    searches += 1;
                }
                searches
            })
        })
        .collect();
    let writers: Vec<_> = (0..6)
        .map(|t| {
            let (idx, base) = (idx.clone(), base.clone());
            std::thread::spawn(move || {
                let mut ids = Vec::new();
                for i in (t..n).step_by(6) {
                    let id = idx.insert(&base[i * dim..(i + 1) * dim]).unwrap();
                    ids.push((id, i));
                }
                ids
            })
        })
        .collect();
    let mut mapping = vec![0u32; n];
    for w in writers {
        for (id, i) in w.join().unwrap() {
            mapping[id as usize] = i as u32;
        }
    }
    done.store(true, Ordering::Release);
    let searches: usize = searchers.into_iter().map(|s| s.join().unwrap()).sum();
    assert!(searches > 0);
    assert_eq!(idx.check().unwrap().nodes, n);
    assert_eq!(
        idx.insert(&base[..dim]).unwrap_err().to_string(),
        "index is full"
    );

    // Recall of the concurrently built graph, with ids mapped back to inputs.
    let truth = ground_truth(Metric::L2, dim, &base, &queries, 10, 4);
    let mut total = 0.0;
    for (qi, t) in truth.iter().enumerate() {
        let got: Vec<u32> = idx
            .search(&queries[qi * dim..(qi + 1) * dim], 10, 64)
            .unwrap()
            .iter()
            .map(|nb| mapping[nb.id as usize])
            .collect();
        total += recall(&got, t, 10);
    }
    let r = total / truth.len() as f64;
    eprintln!("concurrent build recall@10 ef=64: {r:.4}");
    assert!(r >= 0.95, "recall {r}");
}

#[test]
fn tombstones_are_never_returned_and_rebuild_drops_them() {
    let mut rng = Rng::new(9);
    let (n, dim) = (8_000, 16);
    let base = clustered(&mut rng, n, dim, 20);
    let idx = HnswIndex::new(dim, Metric::L2, params(16, 100), n);
    idx.insert_batch(&base, 4).unwrap();
    let mut deleted = vec![false; n];
    for _ in 0..n * 4 / 10 {
        let i = rng.below(n);
        if idx.delete(i as u32) {
            deleted[i] = true;
        }
    }
    assert!(!idx.delete(n as u32 + 5));
    let live: Vec<u32> = (0..n as u32).filter(|&i| !deleted[i as usize]).collect();
    let flat_live = FlatIndex::from_vectors(
        dim,
        Metric::L2,
        &live
            .iter()
            .flat_map(|&i| base[i as usize * dim..][..dim].to_vec())
            .collect::<Vec<_>>(),
    );
    let mut total = 0.0;
    for _ in 0..200 {
        let q = &base[rng.below(n) * dim..][..dim];
        let r = idx.search(q, 10, 64).unwrap();
        assert!(
            r.iter().all(|nb| !deleted[nb.id as usize]),
            "deleted row returned"
        );
        let truth: Vec<u32> = flat_live
            .search(q, 10)
            .iter()
            .map(|nb| live[nb.id as usize])
            .collect();
        total += recall(&r.iter().map(|nb| nb.id).collect::<Vec<_>>(), &truth, 10);
    }
    let r = total / 200.0;
    eprintln!("recall@10 with 33% tombstones: {r:.4}");
    assert!(r >= 0.93, "recall with tombstones {r}");

    let (fresh, map) = idx.rebuilt(0, 2).unwrap();
    assert_eq!(fresh.len(), live.len());
    assert_eq!(fresh.deleted_count(), 0);
    fresh.check().unwrap();
    for (old, new) in map.iter().enumerate() {
        match new {
            Some(new) => assert_eq!(fresh.vector(*new).unwrap(), idx.vector(old as u32).unwrap()),
            None => assert!(deleted[old]),
        }
    }
}

#[test]
fn freeze_view_searches_like_the_live_graph_and_thaws_back() {
    let mut rng = Rng::new(3);
    let (n, dim) = (3_000, 12);
    let base = clustered(&mut rng, n, dim, 10);
    let idx = HnswIndex::new(dim, Metric::Cosine, params(8, 64), n);
    idx.insert_batch(&base, 1).unwrap();
    let parts = idx.graph().freeze(n);
    let view = parts.view();
    view.validate_shape().unwrap();
    check_invariants(&view, 0..n as u32).unwrap();
    let space = idx.space();
    let mut sc = Scratch::new(n);
    for _ in 0..50 {
        let q = Metric::Cosine.prepare(&base[rng.below(n) * dim..][..dim]);
        let a = graph::knn(idx.graph(), &space, &q[..], 10, 40, &|_| true, &mut sc);
        let b = graph::knn(&view, &space, &q[..], 10, 40, &|_| true, &mut sc);
        assert_eq!(a, b);
    }
    let thawed = Graph::thaw(*idx.params(), &view, n);
    assert_eq!(thawed.freeze(n), parts);
}

#[test]
fn corrupted_frozen_graphs_are_rejected_or_harmless() {
    let mut rng = Rng::new(77);
    let (n, dim) = (300, 4);
    let base: Vec<f32> = (0..n * dim).map(|_| rng.normal()).collect();
    let idx = HnswIndex::new(dim, Metric::L2, params(4, 16), n);
    idx.insert_batch(&base, 1).unwrap();
    let parts = idx.graph().freeze(n);
    let flat = FlatIndex::from_vectors(dim, Metric::L2, &base);
    property("corrupt frozen graph", 400, |rng| {
        let mut p = parts.clone();
        for _ in 0..rng.range(1, 4) {
            match rng.below(4) {
                0 => {
                    let i = rng.below(p.links0.len());
                    p.links0[i] = rng.next_u64() as u32 % 1000;
                }
                1 if !p.upper.is_empty() => {
                    let i = rng.below(p.upper.len());
                    p.upper[i] = rng.next_u64() as u32 % 1000;
                }
                2 => {
                    let i = rng.below(p.levels.len());
                    p.levels[i] = rng.next_u64() as u8 % 20;
                }
                _ => {
                    let i = rng.below(p.upper_off.len());
                    p.upper_off[i] = rng.next_u64() as u32 % 2000;
                }
            }
        }
        let v: FrozenGraph<'_> = p.view();
        if v.validate_shape().is_ok() {
            // Must not panic or read out of bounds.
            let mut sc = Scratch::new(n);
            let q: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
            let _ = graph::knn(&v, &flat.space(), &q[..], 5, 20, &|_| true, &mut sc);
        }
        Ok(())
    });
}

#[test]
fn degenerate_inputs() {
    // Empty, single-row and all-identical indexes.
    let idx = HnswIndex::new(3, Metric::L2, params(4, 8), 10);
    assert!(idx.search(&[0.0; 3], 5, 10).unwrap().is_empty());
    assert!(idx.search(&[0.0; 2], 5, 10).is_err());
    assert!(idx.insert(&[f32::NAN, 0.0, 0.0]).is_err());
    idx.insert(&[1.0, 2.0, 3.0]).unwrap();
    let r = idx.search(&[0.0; 3], 5, 10).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].dist, 14.0);

    // All-identical vectors give the heuristic no information (every distance
    // is 0), so later copies are pruned out of full lists and become
    // unreachable; hnswlib behaves the same way. The index must stay
    // structurally valid and return only exact matches.
    let same = HnswIndex::new(4, Metric::L2, params(4, 16), 500);
    for _ in 0..500 {
        same.insert(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    }
    same.check().unwrap();
    let r = same.search(&[1.0; 4], 20, 50).unwrap();
    assert!(!r.is_empty());
    assert!(r.iter().all(|nb| nb.dist == 0.0));
}

#[test]
fn duplicates_inside_real_data_are_found() {
    // A more realistic case: 10% of the rows are exact copies of others.
    let mut rng = Rng::new(21);
    let (n, dim) = (4_000, 8);
    let mut base = clustered(&mut rng, n, dim, 8);
    for i in 0..n / 10 {
        let src = rng.below(n / 2) * dim;
        let dst = (n / 2 + i) * dim;
        let row: Vec<f32> = base[src..src + dim].to_vec();
        base[dst..dst + dim].copy_from_slice(&row);
    }
    let idx = HnswIndex::new(dim, Metric::L2, params(16, 100), n);
    idx.insert_batch(&base, 2).unwrap();
    idx.check().unwrap();
    let mut found = 0;
    for i in 0..n / 10 {
        let dst = (n / 2 + i) as u32;
        let r = idx
            .search(&base[dst as usize * dim..][..dim], 10, 64)
            .unwrap();
        if r.iter().any(|nb| nb.id == dst) {
            found += 1;
        }
    }
    assert!(
        found as f64 >= 0.97 * (n / 10) as f64,
        "found {found} of {}",
        n / 10
    );
}

#[test]
fn concurrent_stress_keeps_invariants() {
    // Small M and tight clusters maximise contention on the same link lists.
    for round in 0..12u64 {
        let mut rng = Rng::new(1000 + round);
        let (n, dim) = (3_000, 4);
        let base = Arc::new(clustered(&mut rng, n, dim, 3));
        let idx = Arc::new(HnswIndex::new(
            dim,
            Metric::L2,
            GraphParams {
                m: 4,
                ef_construction: 24,
                seed: round,
            },
            n,
        ));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let (idx, base) = (idx.clone(), base.clone());
                std::thread::spawn(move || {
                    for i in (t..n).step_by(8) {
                        idx.insert(&base[i * dim..(i + 1) * dim]).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        if let Err(e) = idx.check() {
            panic!("round {round}: {e}");
        }
    }
}
