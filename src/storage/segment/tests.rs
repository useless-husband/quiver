use super::*;
use crate::index::HnswIndex;
use crate::index::graph::{self, Scratch};
use crate::rng::Rng;
use crate::testkit::{clustered, property};

struct Fixture {
    dir: std::path::PathBuf,
    idx: HnswIndex,
    ids: Vec<u64>,
    lsns: Vec<u64>,
    vectors: Vec<f32>,
    payloads: Vec<Vec<u8>>,
}

fn fixture(tag: &str, metric: Metric, n: usize, dim: usize) -> Fixture {
    let dir = std::env::temp_dir().join(format!("quiver-seg-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng::new(n as u64);
    let raw = clustered(&mut rng, n, dim, 5);
    let idx = HnswIndex::new(dim, metric, GraphParams { m: 8, ef_construction: 40, seed: 3 }, n);
    idx.insert_batch(&raw, 2).unwrap();
    let vectors: Vec<f32> = (0..n as u32).flat_map(|i| idx.vector(i).unwrap().to_vec()).collect();
    let ids: Vec<u64> = (0..n as u64).map(|i| i * 7 + 1).collect();
    let lsns: Vec<u64> = (0..n as u64).map(|i| 1000 + i).collect();
    let payloads: Vec<Vec<u8>> =
        (0..n).map(|i| if i % 3 == 0 { Vec::new() } else { format!("{{\"i\":{i}}}").into_bytes() }).collect();
    Fixture { dir, idx, ids, lsns, vectors, payloads }
}

fn write(fx: &Fixture, codes: Option<&Codes>, name: &str) -> std::path::PathBuf {
    let parts = fx.idx.graph().freeze(fx.ids.len());
    let p = fx.dir.join(name);
    write_segment(
        &p,
        &SegmentWrite {
            dim: fx.idx.dim(),
            metric: fx.idx.metric(),
            params: *fx.idx.params(),
            ids: &fx.ids,
            lsns: &fx.lsns,
            vectors: &fx.vectors,
            graph: &parts,
            payloads: &fx.payloads,
            codes,
        },
        SyncMode::None,
    )
    .unwrap();
    p
}

#[test]
fn round_trip_for_every_metric_and_quantization() {
    for metric in [Metric::L2, Metric::Ip, Metric::Cosine] {
        let fx = fixture(metric.name(), metric, 700, 16);
        for q in [Quantization::None, Quantization::Int8, Quantization::Pq { m: 4 }] {
            let codes = Codes::build(q, &fx.vectors, 16, metric, 1, 2).unwrap();
            let p = write(&fx, codes.as_ref(), &format!("{}.qseg", q.label()));
            let s = SegmentFile::open(&p).unwrap();
            assert_eq!((s.rows(), s.dim(), s.metric(), s.quantization()), (700, 16, metric, q));
            assert_eq!(s.ids(), &fx.ids[..]);
            assert_eq!(s.lsns(), &fx.lsns[..]);
            assert_eq!(s.vectors(), &fx.vectors[..]);
            for r in 0..700u32 {
                assert_eq!(s.payload(r), &fx.payloads[r as usize][..]);
            }
            assert_eq!(s.graph().levels, &fx.idx.graph().freeze(700).levels[..]);
            // Searching the mapped graph gives the same answer as the live one.
            let mut sc = Scratch::new(700);
            let mut rng = Rng::new(5);
            for _ in 0..20 {
                let qv = metric.prepare(&(0..16).map(|_| rng.normal()).collect::<Vec<_>>());
                let a = graph::knn(fx.idx.graph(), &fx.idx.space(), &qv[..], 10, 32, &|_| true, &mut sc);
                let b = graph::knn(&s.graph(), &s.space(), &qv[..], 10, 32, &|_| true, &mut sc);
                assert_eq!(a, b);
                if let Some(c) = s.codes() {
                    let r = c.search(&s.graph(), &s.space(), &qv, 10, 32, 4, &|_| true, &mut sc);
                    assert_eq!(r.len(), 10);
                }
            }
        }
        std::fs::remove_dir_all(&fx.dir).unwrap();
    }
}

/// Flipping any byte either makes `open` fail or (for alignment padding,
/// which is not checksummed) leaves every value intact.
#[test]
fn byte_flips_are_detected() {
    let fx = fixture("flip", Metric::L2, 200, 8);
    let codes = Codes::build(Quantization::Int8, &fx.vectors, 8, Metric::L2, 1, 1).unwrap();
    let p = write(&fx, codes.as_ref(), "a.qseg");
    let orig = std::fs::read(&p).unwrap();
    let q = fx.dir.join("flipped.qseg");
    property("segment byte flip", 300, |rng| {
        let mut b = orig.clone();
        let i = rng.below(b.len());
        b[i] ^= 1 << rng.below(8);
        std::fs::write(&q, &b).unwrap();
        match SegmentFile::open(&q) {
            Err(_) => Ok(()),
            Ok(s) => {
                if s.ids() == &fx.ids[..] && s.vectors() == &fx.vectors[..] && s.lsns() == &fx.lsns[..] {
                    Ok(())
                } else {
                    Err(format!("flip at byte {i} went unnoticed"))
                }
            }
        }
    });
    for cut in [0, 10, 511, 512, orig.len() / 2, orig.len() - 1] {
        std::fs::write(&q, &orig[..cut]).unwrap();
        assert!(SegmentFile::open(&q).is_err(), "truncation to {cut} accepted");
    }
    std::fs::remove_dir_all(&fx.dir).unwrap();
}

#[test]
fn tombstone_files_round_trip_and_reject_damage() {
    let dir = std::env::temp_dir().join(format!("quiver-del-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("1.del");
    let words = vec![0xF0F0u64, 0, 1 << 63];
    write_tombstones(&p, 190, &words, SyncMode::None).unwrap();
    assert_eq!(read_tombstones(&p, 190).unwrap(), words);
    assert!(read_tombstones(&p, 191).is_err() || read_tombstones(&p, 191).unwrap().len() == 3);
    assert!(read_tombstones(&p, 300).is_err());
    let mut b = std::fs::read(&p).unwrap();
    b[30] ^= 4;
    std::fs::write(&p, &b).unwrap();
    assert!(read_tombstones(&p, 190).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
