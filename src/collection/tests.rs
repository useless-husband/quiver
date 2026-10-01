use super::*;
use crate::rng::Rng;
use crate::testkit::{clustered, property};
use serde_json::json;

fn tmpdir(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "quiver-coll-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn opts() -> Options {
    Options { sync: SyncMode::None, threads: 2 }
}

type Model = HashMap<u64, (Vec<f32>, Value)>;

fn exact_top(model: &Model, metric: Metric, q: &[f32], k: usize) -> Vec<u64> {
    let mut all: Vec<(f32, u64)> = model.iter().map(|(&id, (v, _))| (metric.distance(q, v), id)).collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|x| x.1).collect()
}

fn check_against_model(
    c: &Collection,
    model: &Model,
    metric: Metric,
    rng: &mut Rng,
    dim: usize,
) -> std::result::Result<(), String> {
    if c.len() != model.len() {
        return Err(format!("len {} but model has {}", c.len(), model.len()));
    }
    let ids: Vec<u64> = (0..60).collect();
    for (id, got) in ids.iter().zip(c.get(&ids, true)) {
        match (model.get(id), got) {
            (None, None) => {}
            (Some((v, p)), Some(r)) => {
                let want = metric.prepare(v);
                let gv = r.vector.unwrap();
                if gv.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-5) {
                    return Err(format!("id {id}: vector {gv:?} != {want:?}"));
                }
                if r.payload.as_ref() != Some(p) {
                    return Err(format!("id {id}: payload {:?} != {p}", r.payload));
                }
            }
            (m, g) => return Err(format!("id {id}: model {:?} vs collection {:?}", m.is_some(), g.is_some())),
        }
    }
    // A filter matching every point with brute force must be exact.
    let all = Filter::parse(&json!({"field": "g", "gte": 0})).unwrap();
    let q: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
    let mut query = Query::new(&q, 5);
    query.filter = Some(&all);
    query.strategy = Strategy::Brute;
    let got: Vec<u64> = c.search(&query).map_err(|e| e.to_string())?.hits.iter().map(|h| h.id).collect();
    let want = exact_top(model, metric, &q, 5);
    if got != want {
        return Err(format!("brute-force search {got:?} != exact {want:?}"));
    }
    // Unfiltered graph search: well-formed and drawn from live points.
    let r = c.search(&Query::new(&q, 5)).map_err(|e| e.to_string())?;
    if r.hits.len() != model.len().min(5) || r.hits.iter().any(|h| !model.contains_key(&h.id)) {
        return Err(format!("graph search returned {:?}", r.hits.iter().map(|h| h.id).collect::<Vec<_>>()));
    }
    if !r.hits.windows(2).all(|w| w[0].distance <= w[1].distance) {
        return Err("graph results not sorted".into());
    }
    Ok(())
}

/// Random operation sequences, including checkpoints, compactions and
/// reopening, always agree with a plain map.
#[test]
fn model_based_random_operations() {
    property("collection vs model", 40, |rng| {
        let dir = tmpdir("model");
        let dim = 4;
        let metric = [Metric::L2, Metric::Ip, Metric::Cosine][rng.below(3)];
        let mut cfg = CollectionConfig::new(dim, metric);
        cfg.m = 4;
        cfg.ef_construction = 16;
        cfg.segment_rows = rng.range(16, 64);
        cfg.quantization = [Quantization::None, Quantization::Int8, Quantization::Pq { m: 2 }][rng.below(3)];
        let mut c = Collection::create(&dir, "t", cfg.clone(), opts()).map_err(|e| e.to_string())?;
        let mut model: Model = HashMap::new();
        for step in 0..60 {
            match rng.below(20) {
                0..=8 => {
                    let n = rng.range(1, 10);
                    let pts: Vec<Point> = (0..n)
                        .map(|_| {
                            let id = rng.below(60) as u64;
                            let v: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
                            Point { id, vector: v, payload: Some(json!({"g": id % 3, "s": step})) }
                        })
                        .collect();
                    c.upsert(&pts).map_err(|e| format!("step {step}: {e}"))?;
                    for p in pts {
                        model.insert(p.id, (p.vector, p.payload.unwrap()));
                    }
                }
                9..=12 => {
                    let ids: Vec<u64> = (0..rng.range(1, 6)).map(|_| rng.below(60) as u64).collect();
                    let mut uniq = ids.clone();
                    uniq.sort();
                    uniq.dedup();
                    let existed = uniq.iter().filter(|id| model.contains_key(id)).count();
                    let (n, _) = c.delete(&ids).map_err(|e| e.to_string())?;
                    if n != existed {
                        return Err(format!("step {step}: delete reported {n}, {existed} existed"));
                    }
                    for id in ids {
                        model.remove(&id);
                    }
                }
                13 | 14 => {
                    c.checkpoint().map_err(|e| e.to_string())?;
                }
                15 => {
                    c.compact().map_err(|e| e.to_string())?;
                }
                16 | 17 => {
                    drop(c);
                    c = Collection::open(&dir, "t", opts()).map_err(|e| format!("step {step} reopen: {e}"))?;
                }
                _ => {}
            }
            check_against_model(&c, &model, metric, rng, dim).map_err(|e| format!("step {step} ({cfg:?}): {e}"))?;
        }
        drop(c);
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    });
}

/// Many threads write the same ids; the result is "highest LSN wins", both
/// in memory and after replaying the WAL and segments from disk.
#[test]
fn concurrent_writes_to_the_same_ids_resolve_by_lsn() {
    let dir = tmpdir("race");
    let mut cfg = CollectionConfig::new(4, Metric::L2);
    cfg.segment_rows = 100;
    cfg.m = 6;
    cfg.ef_construction = 20;
    // fsync makes group commit release many writers at once, which widens
    // the window in which commits of the same id race.
    let c = Arc::new(Collection::create(&dir, "race", cfg, Options { sync: SyncMode::Fsync, threads: 2 }).unwrap());
    let threads: Vec<_> = (0..8)
        .map(|t| {
            let c = c.clone();
            std::thread::spawn(move || {
                let mut rng = Rng::new(t);
                let mut log = Vec::new();
                for _ in 0..300 {
                    let id = rng.below(12) as u64;
                    if rng.bool(0.15) {
                        let (_, lsn) = c.delete(&[id]).unwrap();
                        log.push((id, None, lsn));
                    } else {
                        let v = vec![t as f32, rng.normal(), rng.normal(), id as f32];
                        let lsn = c.upsert(&[Point { id, vector: v.clone(), payload: None }]).unwrap();
                        log.push((id, Some(v), lsn));
                    }
                }
                log
            })
        })
        .collect();
    // The operation with the highest LSN decides each id's final state.
    let mut last: HashMap<u64, (u64, Option<Vec<f32>>)> = HashMap::new();
    for t in threads {
        for (id, v, lsn) in t.join().unwrap() {
            let e = last.entry(id).or_insert((0, None));
            if lsn > e.0 {
                *e = (lsn, v);
            }
        }
    }
    let before: Vec<Option<Record>> = c.get(&(0..50).collect::<Vec<_>>(), true);
    for (id, r) in before.iter().enumerate() {
        let want = last.get(&(id as u64)).and_then(|x| x.1.clone());
        assert_eq!(r.as_ref().and_then(|r| r.vector.clone()), want, "id {id}");
    }
    let stats = c.stats();
    assert!(stats.segments.len() > 3, "expected several sealed segments: {stats:?}");
    drop(c);
    let c = Collection::open(&dir, "race", opts()).unwrap();
    let after = c.get(&(0..50).collect::<Vec<_>>(), true);
    assert_eq!(before, after, "state after recovery differs");
    drop(c);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Commits that arrive out of LSN order (an older upsert finishing its graph
/// insert after a newer one) must not overwrite the newer version, and a
/// delete must block an older in-flight upsert.
#[test]
fn out_of_order_commits_keep_the_highest_lsn() {
    let dir = tmpdir("order");
    let c = Collection::create(&dir, "o", CollectionConfig::new(2, Metric::L2), opts()).unwrap();
    let segs = read(&c.segs).clone();
    let g = segs.growing.clone();
    let old = g.insert(7, 10, &[1.0, 0.0], None).unwrap();
    let new = g.insert(7, 11, &[2.0, 0.0], None).unwrap();
    {
        let mut map = lock(&c.ids);
        Collection::commit_upsert(&mut map, &segs, &g, 7, 11, new);
        Collection::commit_upsert(&mut map, &segs, &g, 7, 10, old);
    }
    assert_eq!(c.get(&[7], true)[0].as_ref().unwrap().vector, Some(vec![2.0, 0.0]));
    assert!(!g.is_live(old) && g.is_live(new));
    let late = g.insert(8, 20, &[3.0, 0.0], None).unwrap();
    {
        let mut map = lock(&c.ids);
        assert!(!Collection::commit_delete(&mut map, &segs, 8, 21));
        Collection::commit_upsert(&mut map, &segs, &g, 8, 20, late);
    }
    assert!(c.get(&[8], false)[0].is_none(), "a newer delete must win over an older upsert");
    drop(c);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wal_only_recovery_and_torn_tail() {
    let dir = tmpdir("wal");
    let cfg = CollectionConfig::new(8, Metric::Cosine);
    let c = Collection::create(&dir, "w", cfg, opts()).unwrap();
    let mut rng = Rng::new(4);
    let data = clustered(&mut rng, 500, 8, 4);
    for (i, v) in data.chunks(8).enumerate() {
        c.upsert(&[Point { id: i as u64, vector: v.to_vec(), payload: Some(json!({"i": i})) }]).unwrap();
    }
    c.delete(&[3, 4, 5]).unwrap();
    assert_eq!(c.len(), 497);
    drop(c);
    let c = Collection::open(&dir, "w", opts()).unwrap();
    assert_eq!(c.len(), 497);
    assert!(c.get(&[3], false)[0].is_none());
    assert_eq!(c.get(&[7], false)[0].as_ref().unwrap().payload, Some(json!({"i": 7})));
    let q = &data[16..24];
    let r = c.search(&Query::new(q, 1)).unwrap();
    assert_eq!(r.hits[0].id, 2);
    drop(c);
    // Tear the last record: the last delete (id 5... the batch) is lost, the
    // rest survives.
    let wal = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("wal-"))
        .unwrap();
    let len = std::fs::metadata(&wal).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&wal).unwrap();
    f.set_len(len - 3).unwrap();
    drop(f);
    let c = Collection::open(&dir, "w", opts()).unwrap();
    assert_eq!(c.len(), 497 + 1, "the torn delete record of id 5 is dropped");
    assert!(c.get(&[5], false)[0].is_some());
    // The log is writable again after truncation.
    c.upsert(&[Point { id: 1000, vector: vec![1.0; 8], payload: None }]).unwrap();
    drop(c);
    let c = Collection::open(&dir, "w", opts()).unwrap();
    assert!(c.get(&[1000], false)[0].is_some());
    drop(c);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn invalid_requests_change_nothing() {
    let dir = tmpdir("inv");
    let c = Collection::create(&dir, "v", CollectionConfig::new(3, Metric::L2), opts()).unwrap();
    let bad = [
        vec![Point { id: 1, vector: vec![1.0, 2.0], payload: None }],
        vec![Point { id: 1, vector: vec![1.0, f32::NAN, 0.0], payload: None }],
        vec![Point { id: 1, vector: vec![1.0; 3], payload: Some(json!([1, 2])) }],
    ];
    for b in &bad {
        assert!(matches!(c.upsert(b), Err(Error::Invalid(_))));
    }
    let many: Vec<Point> =
        (0..MAX_BATCH as u64 + 1).map(|id| Point { id, vector: vec![0.0; 3], payload: None }).collect();
    assert!(c.upsert(&many).is_err());
    assert_eq!(c.stats().next_lsn, 1, "rejected requests must not reach the WAL");
    assert!(c.search(&Query::new(&[0.0; 2], 3)).is_err());
    assert!(c.search(&Query::new(&[0.0; 3], 0)).is_err());
    assert!(c.search(&Query::new(&[0.0; 3], 3)).unwrap().hits.is_empty());
    let mut cfg = CollectionConfig::new(10, Metric::L2);
    cfg.quantization = Quantization::Pq { m: 3 };
    assert!(cfg.validate().is_err());
    let j = serde_json::from_value::<CollectionConfig>(json!({"dim": 4, "metric": "cosine", "bogus": 1}));
    assert!(j.is_err(), "unknown fields are rejected");
    drop(c);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn database_create_list_drop() {
    let root = tmpdir("db");
    let db = crate::db::Database::open(&root, opts()).unwrap();
    db.create("a", CollectionConfig::new(2, Metric::L2)).unwrap();
    assert!(matches!(db.create("a", CollectionConfig::new(2, Metric::L2)), Err(Error::Exists(_))));
    assert!(db.create("bad name", CollectionConfig::new(2, Metric::L2)).is_err());
    let a = db.get("a").unwrap();
    a.upsert(&[Point { id: 1, vector: vec![1.0, 1.0], payload: None }]).unwrap();
    db.create("b", CollectionConfig::new(3, Metric::Ip)).unwrap();
    // A second instance on the same directory is refused while one is open.
    assert!(crate::db::Database::open(&root, opts()).is_err());
    assert!(Collection::open(a.dir(), "a", opts()).is_err());
    drop(a);
    drop(db);
    let db = crate::db::Database::open(&root, opts()).unwrap();
    let a = db.get("a").unwrap();
    assert_eq!(db.list().iter().map(|c| c.name().to_string()).collect::<Vec<_>>(), vec!["a", "b"]);
    assert_eq!(db.get("a").unwrap().len(), 1);
    db.drop_collection("a").unwrap();
    assert!(a.upsert(&[Point { id: 2, vector: vec![1.0, 1.0], payload: None }]).is_err(), "dropped handle must stop");
    assert!(db.get("a").is_err());
    db.create("a", CollectionConfig::new(2, Metric::L2)).unwrap();
    assert_eq!(db.get("a").unwrap().len(), 0);
    drop(db);
    std::fs::remove_dir_all(&root).unwrap();
}
