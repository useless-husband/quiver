//! `quiver serve` and the client subcommands.

use super::args::Args;
use quiver::client::{parse_addr, request};
use quiver::collection::{CollectionConfig, Options};
use quiver::db::Database;
use quiver::quant::Quantization;
use quiver::server::{ServerConfig, install_signal_handlers, serve as run_server};
use quiver::storage::SyncMode;
use quiver::{Metric, bench::read_fbin};
use serde_json::{Value, json};
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;

pub fn serve(a: &Args) -> Result<(), String> {
    let data = PathBuf::from(a.get("data", "quiver-data".to_string())?);
    let listen = a.get("listen", "127.0.0.1:6333".to_string())?;
    let threads = a.get("threads", 8usize)?;
    let build_threads = a.get("build-threads", 4usize)?;
    let sync = SyncMode::parse(&a.get("sync", "full".to_string())?).ok_or("--sync must be full, fsync or none")?;
    let port_file = a.str("port-file").map(PathBuf::from);
    let compact_interval = a.get("compact-interval", 10u64)?;
    a.finish()?;
    install_signal_handlers();
    let db = Database::open(&data, Options { sync, threads: build_threads }).map_err(|e| e.to_string())?;
    let cfg = ServerConfig { listen, threads, port_file, compact_interval, ..ServerConfig::default() };
    run_server(Arc::new(db), cfg).map_err(|e| e.to_string())
}

struct Client {
    addr: String,
}

impl Client {
    fn new(a: &Args) -> Result<Client, String> {
        let url = a.get("url", std::env::var("QUIVER_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".into()))?;
        Ok(Client { addr: parse_addr(&url) })
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
        let r = request(&self.addr, method, path, body.as_ref()).map_err(|e| format!("{}: {e}", self.addr))?;
        if r.status != 200 {
            let msg =
                r.body.get("error").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| r.body.to_string());
            return Err(format!("HTTP {}: {msg}", r.status));
        }
        Ok(r.body)
    }
}

fn print(v: &Value) {
    println!("{}", serde_json::to_string_pretty(v).unwrap());
}

fn name(a: &Args) -> Result<String, String> {
    a.positional.first().cloned().ok_or_else(|| "missing collection name".to_string())
}

pub fn client(cmd: &str, a: &Args) -> Result<(), String> {
    let c = Client::new(a)?;
    match cmd {
        "list" => {
            a.finish()?;
            let v = c.call("GET", "/collections", None)?;
            for col in v["collections"].as_array().into_iter().flatten() {
                println!(
                    "{:<24} {:>10} points  dim {:<5} {:<7} {}",
                    col["name"].as_str().unwrap_or(""),
                    col["points"],
                    col["config"]["dim"],
                    col["config"]["metric"].as_str().unwrap_or(""),
                    col["segments"].as_array().map_or(0, |s| s.len())
                );
            }
            Ok(())
        }
        "create" => {
            let n = name(a)?;
            let dim = a.req("dim")?.parse::<usize>().map_err(|_| "--dim must be a number")?;
            let metric = Metric::parse(&a.req("metric")?).ok_or("--metric must be l2, ip or cosine")?;
            let mut cfg = CollectionConfig::new(dim, metric);
            cfg.m = a.get("m", cfg.m)?;
            cfg.ef_construction = a.get("ef-construction", cfg.ef_construction)?;
            cfg.ef_search = a.get("ef-search", cfg.ef_search)?;
            cfg.segment_rows = a.get("segment-rows", cfg.segment_rows)?;
            cfg.rerank = a.get("rerank", cfg.rerank)?;
            if let Some(q) = a.str("quantization") {
                cfg.quantization = Quantization::parse(&q).ok_or("--quantization must be none, int8 or pqM")?;
            }
            a.finish()?;
            print(&c.call("PUT", &format!("/collections/{n}"), Some(serde_json::to_value(&cfg).unwrap()))?);
            Ok(())
        }
        "drop" => {
            let n = name(a)?;
            a.finish()?;
            print(&c.call("DELETE", &format!("/collections/{n}"), None)?);
            Ok(())
        }
        "stats" => {
            let n = name(a)?;
            a.finish()?;
            print(&c.call("GET", &format!("/collections/{n}"), None)?);
            Ok(())
        }
        "checkpoint" | "compact" => {
            let n = name(a)?;
            a.finish()?;
            print(&c.call("POST", &format!("/collections/{n}/{cmd}"), None)?);
            Ok(())
        }
        "get" => {
            let n = name(a)?;
            let id = a.positional.get(1).ok_or("missing point id")?;
            a.finish()?;
            print(&c.call("GET", &format!("/collections/{n}/points/{id}"), None)?);
            Ok(())
        }
        "delete" => {
            let n = name(a)?;
            let ids: Vec<u64> = a
                .positional
                .get(1)
                .ok_or("missing ids (comma-separated)")?
                .split(',')
                .map(|s| s.trim().parse().map_err(|_| format!("invalid id {s:?}")))
                .collect::<Result<_, _>>()?;
            a.finish()?;
            print(&c.call("POST", &format!("/collections/{n}/points/delete"), Some(json!({"ids": ids})))?);
            Ok(())
        }
        "search" => {
            let n = name(a)?;
            let vector: Vec<f32> = a
                .req("vector")?
                .split(',')
                .map(|s| s.trim().parse().map_err(|_| format!("invalid number {s:?} in --vector")))
                .collect::<Result<_, _>>()?;
            let mut body = json!({"vector": vector, "k": a.get("k", 10usize)?, "with_payload": true});
            if let Some(ef) = a.opt::<usize>("ef")? {
                body["ef"] = json!(ef);
            }
            if let Some(f) = a.str("filter") {
                body["filter"] = serde_json::from_str(&f).map_err(|e| format!("--filter is not JSON: {e}"))?;
            }
            if let Some(s) = a.str("strategy") {
                body["strategy"] = json!(s);
            }
            a.finish()?;
            print(&c.call("POST", &format!("/collections/{n}/search"), Some(body))?);
            Ok(())
        }
        "import" => {
            let n = name(a)?;
            let file = PathBuf::from(a.positional.get(1).ok_or("missing input file (.fbin or .jsonl)")?);
            let batch = a.get("batch", 1000usize)?;
            let start = a.get("start-id", 0u64)?;
            let limit = a.opt::<usize>("limit")?;
            a.finish()?;
            let t0 = std::time::Instant::now();
            let mut sent = 0usize;
            let mut flush = |pts: &mut Vec<Value>| -> Result<(), String> {
                if !pts.is_empty() {
                    c.call("PUT", &format!("/collections/{n}/points"), Some(json!({"points": pts})))?;
                    sent += pts.len();
                    pts.clear();
                }
                Ok(())
            };
            let mut pts = Vec::with_capacity(batch);
            if file.extension().is_some_and(|e| e == "fbin") {
                let (rows, dim, data) = read_fbin(&file).map_err(|e| e.to_string())?;
                for i in 0..limit.unwrap_or(rows).min(rows) {
                    pts.push(json!({"id": start + i as u64, "vector": &data[i * dim..(i + 1) * dim]}));
                    if pts.len() == batch {
                        flush(&mut pts)?;
                    }
                }
            } else {
                let f = std::fs::File::open(&file).map_err(|e| format!("{}: {e}", file.display()))?;
                for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
                    if limit.is_some_and(|l| i >= l) {
                        break;
                    }
                    let line = line.map_err(|e| e.to_string())?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    pts.push(serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", i + 1))?);
                    if pts.len() == batch {
                        flush(&mut pts)?;
                    }
                }
            }
            flush(&mut pts)?;
            eprintln!("imported {sent} points in {:.1}s", t0.elapsed().as_secs_f64());
            Ok(())
        }
        _ => Err("usage".into()),
    }
}
