//! `quiver bench ...`: the measurements behind the README.

use super::args::Args;
use quiver::bench::{parallel_for, peak_rss_bytes, percentile, read_fbin, read_ibin};
use quiver::index::recall;
use quiver::{GraphParams, HnswIndex, Metric};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

pub const ANN_HEADER: &str = "library,variant,dataset,metric,m,ef_construction,ef,k,query_threads,build_threads,recall,qps,p50_us,p99_us,build_s,index_mb,peak_rss_mb";

pub struct Workload {
    pub dim: usize,
    pub base: Vec<f32>,
    pub queries: Vec<f32>,
    pub truth: Vec<u32>,
    pub truth_k: usize,
}

impl Workload {
    pub fn load(a: &Args) -> Result<Workload, String> {
        let e = |x: quiver::Error| x.to_string();
        let (n, dim, base) = read_fbin(&PathBuf::from(a.req("base")?)).map_err(e)?;
        let (nq, qd, mut queries) = read_fbin(&PathBuf::from(a.req("queries")?)).map_err(e)?;
        let (tn, truth_k, mut truth) = read_ibin(&PathBuf::from(a.req("truth")?)).map_err(e)?;
        if qd != dim || tn != nq {
            return Err(format!("shape mismatch: base {n}x{dim}, queries {nq}x{qd}, truth {tn}x{truth_k}"));
        }
        if let Some(limit) = a.opt::<usize>("nq")? {
            let limit = limit.min(nq);
            queries.truncate(limit * dim);
            truth.truncate(limit * truth_k);
        }
        Ok(Workload { dim, base, queries, truth, truth_k })
    }

    pub fn nq(&self) -> usize {
        self.queries.len() / self.dim
    }

    pub fn query(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dim..(i + 1) * self.dim]
    }

    pub fn truth(&self, i: usize) -> &[u32] {
        &self.truth[i * self.truth_k..(i + 1) * self.truth_k]
    }
}

/// One point on a recall/QPS curve.
pub struct Point {
    pub recall: f64,
    pub qps: f64,
    pub p50_us: f64,
    pub p99_us: f64,
}

/// Runs every query once on `threads` threads with `search`, after a short
/// warm-up, and measures recall@k and throughput.
pub fn sweep_point<F>(w: &Workload, k: usize, threads: usize, search: F) -> Point
where
    F: Fn(&[f32], &mut Vec<u32>) + Sync,
{
    let nq = w.nq();
    let ids: Vec<AtomicU32> = (0..nq * k).map(|_| AtomicU32::new(u32::MAX)).collect();
    let lat: Vec<AtomicU64> = (0..nq).map(|_| AtomicU64::new(0)).collect();
    // Warm-up: touch the index from every thread.
    parallel_for(nq.min(500), threads, |i| {
        let mut out = Vec::new();
        search(w.query(i), &mut out);
    });
    let secs = parallel_for(nq, threads, |i| {
        let mut out = Vec::with_capacity(k);
        let t0 = Instant::now();
        search(w.query(i), &mut out);
        lat[i].store(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        for (j, &id) in out.iter().take(k).enumerate() {
            ids[i * k + j].store(id, Ordering::Relaxed);
        }
    });
    let mut total = 0.0;
    for i in 0..nq {
        let got: Vec<u32> = ids[i * k..(i + 1) * k].iter().map(|x| x.load(Ordering::Relaxed)).collect();
        total += recall(&got, w.truth(i), k);
    }
    let mut l: Vec<f64> = lat.iter().map(|x| x.load(Ordering::Relaxed) as f64 / 1000.0).collect();
    l.sort_by(f64::total_cmp);
    Point {
        recall: total / nq as f64,
        qps: nq as f64 / secs,
        p50_us: percentile(&l, 50.0),
        p99_us: percentile(&l, 99.0),
    }
}

/// `quiver bench ann`: build an HNSW index from an `.fbin` file and sweep `ef`.
pub fn ann(a: &Args) -> Result<(), String> {
    let w = Workload::load(a)?;
    let metric = Metric::parse(&a.req("metric")?).ok_or("--metric must be l2, ip or cosine")?;
    let dataset = a.get("dataset", "unnamed".to_string())?;
    let m = a.get("m", 16usize)?;
    let efc = a.get("efc", 200usize)?;
    let k = a.get("k", 10usize)?;
    let efs = a.list("ef", &[10usize, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])?;
    let threads = a.list("threads", &[1usize])?;
    let build_threads = a.get("build-threads", 4usize)?;
    let quants = a.list("quant", &["f32".to_string()])?;
    let rerank = a.get("rerank", 4usize)?;
    let out = a.str("out");
    let seed = a.get("seed", 1u64)?;
    a.finish()?;
    if k > w.truth_k {
        return Err(format!("k={k} exceeds the ground truth depth {}", w.truth_k));
    }

    let n = w.base.len() / w.dim;
    let params = GraphParams { m, ef_construction: efc, seed };
    eprintln!(
        "building HNSW over {n} x {} ({}) with M={m} efConstruction={efc} on {build_threads} threads",
        w.dim,
        metric.name()
    );
    let t0 = Instant::now();
    let idx = HnswIndex::new(w.dim, metric, params, n);
    idx.insert_batch(&w.base, build_threads).map_err(|e| e.to_string())?;
    let build_s = t0.elapsed().as_secs_f64();
    let index_mb = idx.memory_bytes() as f64 / 1e6;
    eprintln!("built in {build_s:.1} s, index {index_mb:.0} MB, kernels {}", quiver::distance::kernels().name);

    let mut sink: Box<dyn Write> = match &out {
        Some(p) => {
            let exists = std::path::Path::new(p).exists();
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).map_err(|e| e.to_string())?;
            if !exists {
                writeln!(f, "{ANN_HEADER}").map_err(|e| e.to_string())?;
            }
            Box::new(f)
        }
        None => {
            println!("{ANN_HEADER}");
            Box::new(std::io::stdout())
        }
    };
    for qname in &quants {
        let quant = quiver::quant::Quantization::parse(qname).ok_or(format!("unknown --quant {qname}"))?;
        let t0 = Instant::now();
        let codes = idx.build_codes(quant, build_threads).map_err(|e| e.to_string())?;
        // Runs forced onto other kernels (QUIVER_KERNELS) are labelled, so
        // they are never mistaken for the default build.
        let kname = quiver::distance::kernels().name;
        let best = quiver::distance::available().last().map_or("", |k| k.name);
        let suffix = if kname == best { String::new() } else { format!("-{kname}") };
        let (variant, mb) = match &codes {
            None => (format!("hnsw-f32{suffix}"), index_mb),
            Some(c) => {
                eprintln!(
                    "{} codes: {:.1} MB, built in {:.1} s",
                    quant.label(),
                    c.bytes() as f64 / 1e6,
                    t0.elapsed().as_secs_f64()
                );
                // Resident set for a quantized segment: graph + codes; the f32
                // vectors are only read for re-ranking (memory-mapped).
                let graph_mb = idx.graph().memory_bytes() as f64 / 1e6;
                (format!("hnsw-{}-rerank{rerank}{suffix}", quant.label()), graph_mb + c.bytes() as f64 / 1e6)
            }
        };
        for &t in &threads {
            for &ef in &efs {
                let p = sweep_point(&w, k, t, |q, out| {
                    let r = match &codes {
                        None => idx.search(q, k, ef),
                        Some(c) => idx.search_quantized(c, q, k, ef, rerank),
                    };
                    out.extend(r.unwrap().iter().map(|nb| nb.id));
                });
                let line = format!(
                    "quiver,{variant},{dataset},{},{m},{efc},{ef},{k},{t},{build_threads},{:.4},{:.1},{:.1},{:.1},{build_s:.2},{mb:.1},{:.1}",
                    metric.name(),
                    p.recall,
                    p.qps,
                    p.p50_us,
                    p.p99_us,
                    peak_rss_bytes() as f64 / 1e6
                );
                if out.is_some() {
                    eprintln!("{line}");
                }
                writeln!(sink, "{line}").map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

/// Times `f` over `pairs` input pairs; returns the median ns per call of
/// `runs` runs.
fn time_kernel<F: Fn(usize) -> f64>(pairs: usize, f: F) -> f64 {
    let calls_per_run = 2_000_000usize;
    let reps = (calls_per_run / pairs).max(1);
    let mut sink = 0.0;
    for i in 0..pairs * 200 {
        sink += f(i % pairs);
    }
    let mut samples = Vec::new();
    for _ in 0..7 {
        let t0 = Instant::now();
        for _ in 0..reps {
            for i in 0..pairs {
                sink += f(std::hint::black_box(i));
            }
        }
        samples.push(t0.elapsed().as_nanos() as f64 / (reps * pairs) as f64);
    }
    std::hint::black_box(sink);
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// `quiver bench kernels`: every available kernel table on L1-resident
/// inputs, at several dimensions.
pub fn kernels(a: &Args) -> Result<(), String> {
    let dims = a.list("dims", &[32usize, 100, 128, 384, 768, 784, 1536])?;
    let out = a.str("out");
    a.finish()?;
    let tables = quiver::distance::available();
    let mut rng = quiver::rng::Rng::new(1);
    let pairs = 32;
    let mut lines = vec!["kernel,impl,dim,ns_per_call,gflops".to_string()];
    println!("{:<10} {:<13} {:>5} {:>10} {:>8}", "kernel", "impl", "dim", "ns/call", "GFLOP/s");
    for &dim in &dims {
        let fa: Vec<f32> = (0..pairs * dim).map(|_| rng.normal()).collect();
        let fb: Vec<f32> = (0..pairs * dim).map(|_| rng.normal()).collect();
        let ia: Vec<i8> = (0..pairs * dim).map(|_| rng.next_u64() as i8).collect();
        let ib: Vec<i8> = (0..pairs * dim).map(|_| rng.next_u64() as i8).collect();
        let sl = |v: &'static [f32], i: usize| &v[i * dim..(i + 1) * dim];
        let fa: &'static [f32] = Box::leak(fa.into_boxed_slice());
        let fb: &'static [f32] = Box::leak(fb.into_boxed_slice());
        let ia: &'static [i8] = Box::leak(ia.into_boxed_slice());
        let ib: &'static [i8] = Box::leak(ib.into_boxed_slice());
        for k in &tables {
            let cases: [(&str, f64, Box<dyn Fn(usize) -> f64>); 5] = [
                ("l2sq_f32", 3.0, Box::new(move |i| (k.l2sq_f32)(sl(fa, i), sl(fb, i)) as f64)),
                ("dot_f32", 2.0, Box::new(move |i| (k.dot_f32)(sl(fa, i), sl(fb, i)) as f64)),
                ("cos_f32", 6.0, Box::new(move |i| (k.cosine_f32)(sl(fa, i), sl(fb, i)) as f64)),
                (
                    "dot_i8",
                    2.0,
                    Box::new(move |i| (k.dot_i8)(&ia[i * dim..(i + 1) * dim], &ib[i * dim..(i + 1) * dim]) as f64),
                ),
                (
                    "l2sq_i8",
                    3.0,
                    Box::new(move |i| (k.l2sq_i8)(&ia[i * dim..(i + 1) * dim], &ib[i * dim..(i + 1) * dim]) as f64),
                ),
            ];
            for (name, flops, f) in cases.iter() {
                let ns = time_kernel(pairs, f);
                let gf = flops * dim as f64 / ns;
                println!("{name:<10} {:<13} {dim:>5} {ns:>10.2} {gf:>8.1}", k.name);
                lines.push(format!("{name},{},{dim},{ns:.3},{gf:.2}", k.name));
            }
        }
    }
    if let Some(p) = out {
        std::fs::write(&p, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `quiver bench filter`: filtered search at several selectivities through a
/// real collection (one sealed, memory-mapped segment), against exact
/// filtered ground truth.
pub fn filter(a: &Args) -> Result<(), String> {
    use quiver::collection::{Collection, CollectionConfig, Options, Point, Query, Strategy};
    use quiver::payload::Filter;
    let e = |x: quiver::Error| x.to_string();
    let (n_all, dim, base) = read_fbin(&PathBuf::from(a.req("base")?)).map_err(e)?;
    let (nq_all, _, queries) = read_fbin(&PathBuf::from(a.req("queries")?)).map_err(e)?;
    let metric = Metric::parse(&a.req("metric")?).ok_or("--metric must be l2, ip or cosine")?;
    let n = a.get("n", n_all)?.min(n_all);
    let nq = a.get("nq", 1000usize)?.min(nq_all);
    let k = a.get("k", 10usize)?;
    let ef = a.get("ef", 64usize)?;
    let dataset = a.get("dataset", "unnamed".to_string())?;
    let threads = a.get("build-threads", 4usize)?;
    let pcts = a.list("selectivity", &[50.0f64, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1])?;
    let out = a.str("out");
    a.finish()?;
    let base = &base[..n * dim];
    let mut rng = quiver::rng::Rng::new(42);
    // bucket uniform in 0..10000, so `bucket < 100 * pct` selects pct %.
    let bucket: Vec<u32> = (0..n).map(|_| rng.below(10_000) as u32).collect();
    let dir = std::env::temp_dir().join(format!("quiver-bench-filter-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut cfg = CollectionConfig::new(dim, metric);
    cfg.segment_rows = n.max(16);
    let opts = Options { sync: quiver::storage::SyncMode::None, threads };
    let c = Collection::create(&dir, "bench", cfg, opts).map_err(e)?;
    let t0 = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let start = next.fetch_add(256, Ordering::Relaxed);
                    if start >= n {
                        break;
                    }
                    let pts: Vec<Point> = (start..(start + 256).min(n))
                        .map(|i| Point {
                            id: i as u64,
                            vector: base[i * dim..(i + 1) * dim].to_vec(),
                            payload: Some(serde_json::json!({"bucket": bucket[i]})),
                        })
                        .collect();
                    c.upsert(&pts).expect("upsert");
                }
            });
        }
    });
    c.checkpoint().map_err(e)?;
    eprintln!("loaded {n} points and sealed in {:.1}s", t0.elapsed().as_secs_f64());
    let header = "dataset,n,selectivity_pct,matched,method,ef,recall,qps,chosen";
    let mut lines = vec![header.to_string()];
    println!("{header}");
    let flat = quiver::FlatIndex::from_vectors(dim, metric, base);
    for &pct in &pcts {
        let limit = (pct * 100.0).round() as u32;
        let matched = bucket.iter().filter(|&&b| b < limit).count();
        let fjson = serde_json::json!({"field": "bucket", "lt": limit});
        let f = Filter::parse(&fjson).map_err(e)?;
        // Exact filtered ground truth.
        let truth: Vec<Vec<u64>> = {
            let mut t = vec![Vec::new(); nq];
            let chunk = nq.div_ceil(threads);
            std::thread::scope(|s| {
                for (ci, slot) in t.chunks_mut(chunk).enumerate() {
                    let (flat, bucket, queries) = (&flat, &bucket, &queries);
                    s.spawn(move || {
                        for (j, o) in slot.iter_mut().enumerate() {
                            let qi = ci * chunk + j;
                            let q = &queries[qi * dim..(qi + 1) * dim];
                            *o = flat
                                .search_with(q, k, |i| bucket[i as usize] < limit)
                                .iter()
                                .map(|x| x.id as u64)
                                .collect();
                        }
                    });
                }
            });
            t
        };
        let mut run = |method: &str, strategy: Option<Strategy>| -> Result<(), String> {
            let mut total = 0.0;
            let mut chosen = String::new();
            let t0 = Instant::now();
            for qi in 0..nq {
                let qv = &queries[qi * dim..(qi + 1) * dim];
                let got: Vec<u64> = match strategy {
                    Some(st) => {
                        let mut q = Query::new(qv, k);
                        q.ef = Some(ef);
                        q.filter = Some(&f);
                        q.strategy = st;
                        let r = c.search(&q).map_err(e)?;
                        if chosen.is_empty() {
                            chosen = r
                                .plan
                                .iter()
                                .filter_map(|p| p.strategy)
                                .map(|s| format!("{s:?}").to_lowercase())
                                .collect::<Vec<_>>()
                                .join("+");
                        }
                        r.hits.iter().map(|h| h.id).collect()
                    }
                    None => {
                        // Post-filtering: fetch k / selectivity candidates
                        // unfiltered, keep the matching ones.
                        let kk = ((k as f64 * 100.0 / pct).ceil() as usize).clamp(k, 4096);
                        let mut q = Query::new(qv, kk);
                        q.ef = Some(kk.max(ef));
                        let r = c.search(&q).map_err(e)?;
                        r.hits.iter().map(|h| h.id).filter(|&id| bucket[id as usize] < limit).take(k).collect()
                    }
                };
                let t = &truth[qi];
                let hits = got.iter().filter(|id| t.contains(id)).count();
                total += if t.is_empty() { 1.0 } else { hits as f64 / t.len() as f64 };
            }
            let qps = nq as f64 / t0.elapsed().as_secs_f64();
            let line =
                format!("{dataset},{n},{pct},{matched},{method},{ef},{:.4},{qps:.1},{chosen}", total / nq as f64);
            println!("{line}");
            lines.push(line);
            Ok(())
        };
        run("graph", Some(Strategy::Graph))?;
        run("brute", Some(Strategy::Brute))?;
        run("auto", Some(Strategy::Auto))?;
        run("postfilter", None)?;
    }
    drop(c);
    let _ = std::fs::remove_dir_all(&dir);
    if let Some(p) = out {
        std::fs::write(&p, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    }
    Ok(())
}
