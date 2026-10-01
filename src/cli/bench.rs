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
        let (variant, mb) = match &codes {
            None => ("hnsw-f32".to_string(), index_mb),
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
                (format!("hnsw-{}-rerank{rerank}", quant.label()), graph_mb + c.bytes() as f64 / 1e6)
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
