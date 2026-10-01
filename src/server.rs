//! The HTTP/JSON API (`quiver serve`).
//!
//! | method | path | body | |
//! |---|---|---|---|
//! | GET | `/healthz` | | liveness |
//! | GET | `/collections` | | stats of every collection |
//! | PUT | `/collections/{name}` | config | create |
//! | GET | `/collections/{name}` | | stats |
//! | DELETE | `/collections/{name}` | | drop |
//! | PUT/POST | `/collections/{name}/points` | `{"points": [...]}` | upsert |
//! | POST | `/collections/{name}/points/delete` | `{"ids": [...]}` | delete |
//! | POST | `/collections/{name}/points/get` | `{"ids": [...], "with_vector": b}` | bulk get |
//! | GET | `/collections/{name}/points/{id}` | | get one |
//! | POST | `/collections/{name}/search` | search request | k-NN search |
//! | POST | `/collections/{name}/checkpoint` | | seal + rotate WAL |
//! | POST | `/collections/{name}/compact` | | rebuild sealed segments |
//!
//! Writes are answered only after they are durable (see `collection.rs`).
//! Errors are `{"error": "..."}` with 400 (bad request), 404, 405, 409
//! (exists), 413 (body too large) or 500.

use crate::collection::{CollectionConfig, Point, Query, Strategy};
use crate::db::Database;
use crate::error::Error;
use crate::payload::Filter;
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct ServerConfig {
    pub listen: String,
    pub threads: usize,
    /// Written with the bound address once listening (for `--listen :0`).
    pub port_file: Option<PathBuf>,
    pub max_body: usize,
    /// Seconds between auto-compaction checks; 0 disables them.
    pub compact_interval: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            listen: "127.0.0.1:6333".into(),
            threads: 8,
            port_file: None,
            max_body: 64 << 20,
            compact_interval: 10,
        }
    }
}

struct HttpError(u16, String);

impl From<Error> for HttpError {
    fn from(e: Error) -> Self {
        let code = match &e {
            Error::Invalid(_) => 400,
            Error::NotFound(_) => 404,
            Error::Exists(_) => 409,
            Error::Full => 503,
            Error::Corrupt(_) | Error::Io(_) => 500,
        };
        HttpError(code, e.to_string())
    }
}

fn bad(msg: impl Into<String>) -> HttpError {
    HttpError(400, msg.into())
}

fn parse<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T, HttpError> {
    serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON body: {e}")))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertBody {
    points: Vec<Point>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdsBody {
    ids: Vec<u64>,
    #[serde(default)]
    with_vector: bool,
}

fn ten() -> usize {
    10
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchBody {
    vector: Vec<f32>,
    #[serde(default = "ten")]
    k: usize,
    ef: Option<usize>,
    filter: Option<Value>,
    strategy: Option<Strategy>,
    rerank: Option<usize>,
    #[serde(default)]
    with_payload: bool,
    #[serde(default)]
    with_vector: bool,
}

fn route(db: &Database, method: &str, parts: &[&str], body: &[u8]) -> Result<Value, HttpError> {
    let not_allowed = || HttpError(405, format!("{method} is not allowed here"));
    match parts {
        ["healthz"] => Ok(json!({"status": "ok"})),
        ["collections"] => match method {
            "GET" => Ok(json!({"collections": db.list().iter().map(|c| c.stats()).collect::<Vec<_>>()})),
            _ => Err(not_allowed()),
        },
        ["collections", name] => match method {
            "PUT" | "POST" => {
                let cfg: CollectionConfig = parse(body)?;
                db.create(name, cfg)?;
                Ok(json!({"created": name}))
            }
            "GET" => Ok(serde_json::to_value(db.get(name)?.stats()).unwrap()),
            "DELETE" => {
                db.drop_collection(name)?;
                Ok(json!({"dropped": name}))
            }
            _ => Err(not_allowed()),
        },
        ["collections", name, "points"] if method == "PUT" || method == "POST" => {
            let b: UpsertBody = parse(body)?;
            let c = db.get(name)?;
            let lsn = c.upsert(&b.points)?;
            Ok(json!({"upserted": b.points.len(), "lsn": lsn}))
        }
        ["collections", name, "points", "delete"] if method == "POST" => {
            let b: IdsBody = parse(body)?;
            let (n, lsn) = db.get(name)?.delete(&b.ids)?;
            Ok(json!({"deleted": n, "lsn": lsn}))
        }
        ["collections", name, "points", "get"] if method == "POST" => {
            let b: IdsBody = parse(body)?;
            if b.ids.len() > 100_000 {
                return Err(bad("at most 100000 ids per request"));
            }
            Ok(json!({"points": db.get(name)?.get(&b.ids, b.with_vector)}))
        }
        ["collections", name, "points", id] if method == "GET" => {
            let id: u64 = id.parse().map_err(|_| bad(format!("invalid point id {id:?}")))?;
            match db.get(name)?.get(&[id], true).pop().flatten() {
                Some(r) => Ok(serde_json::to_value(r).unwrap()),
                None => Err(HttpError(404, format!("point {id} not found"))),
            }
        }
        ["collections", name, "search"] if method == "POST" => {
            let b: SearchBody = parse(body)?;
            let c = db.get(name)?;
            let filter = b.filter.as_ref().map(Filter::parse).transpose()?;
            let q = Query {
                vector: &b.vector,
                k: b.k,
                ef: b.ef,
                filter: filter.as_ref(),
                strategy: b.strategy.unwrap_or(Strategy::Auto),
                rerank: b.rerank,
                with_payload: b.with_payload,
                with_vector: b.with_vector,
            };
            let t0 = Instant::now();
            let r = c.search(&q)?;
            Ok(json!({"hits": r.hits, "plan": r.plan, "took_us": t0.elapsed().as_micros() as u64}))
        }
        ["collections", name, "checkpoint"] if method == "POST" => Ok(json!({"sealed": db.get(name)?.checkpoint()?})),
        ["collections", name, "compact"] if method == "POST" => {
            Ok(serde_json::to_value(db.get(name)?.compact()?).unwrap())
        }
        ["collections", _, ..] => Err(not_allowed()),
        _ => Err(HttpError(404, "no such endpoint".into())),
    }
}

fn handle(db: &Database, mut req: tiny_http::Request, max_body: usize) {
    let method = req.method().as_str().to_ascii_uppercase();
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("");
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let result = (|| {
        if req.body_length().is_some_and(|n| n > max_body) {
            return Err(HttpError(413, format!("body larger than {max_body} bytes")));
        }
        let mut body = Vec::new();
        req.as_reader()
            .take(max_body as u64 + 1)
            .read_to_end(&mut body)
            .map_err(|e| bad(format!("reading the body: {e}")))?;
        if body.len() > max_body {
            return Err(HttpError(413, format!("body larger than {max_body} bytes")));
        }
        route(db, &method, &parts, &body)
    })();
    let (code, value) = match result {
        Ok(v) => (200, v),
        Err(HttpError(c, m)) => (c, json!({"error": m})),
    };
    let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let resp =
        tiny_http::Response::from_data(serde_json::to_vec(&value).unwrap()).with_status_code(code).with_header(header);
    let _ = req.respond(resp);
}

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

/// Installs SIGINT/SIGTERM handlers that make [`serve`] return.
pub fn install_signal_handlers() {
    // SAFETY: the handler only stores to an atomic, which is
    // async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
}

/// Serves until SIGINT/SIGTERM, then checkpoints every collection so the next
/// start does not need to replay the WAL.
pub fn serve(db: Arc<Database>, cfg: ServerConfig) -> crate::Result<()> {
    let server = tiny_http::Server::http(&cfg.listen)
        .map_err(|e| Error::Io(std::io::Error::other(format!("cannot listen on {}: {e}", cfg.listen))))?;
    let addr = server.server_addr().to_ip().map(|a| a.to_string()).unwrap_or_else(|| cfg.listen.clone());
    eprintln!("quiver {} listening on http://{addr} (data: {})", env!("CARGO_PKG_VERSION"), db.root().display());
    if let Some(p) = &cfg.port_file {
        crate::storage::fs::write_atomic(p, addr.as_bytes(), crate::storage::SyncMode::None)?;
    }
    let server = Arc::new(server);
    let mut handles = Vec::new();
    for _ in 0..cfg.threads.max(1) {
        let (server, db) = (server.clone(), db.clone());
        let max_body = cfg.max_body;
        handles.push(std::thread::spawn(move || {
            while !STOP.load(Ordering::SeqCst) {
                match server.recv_timeout(Duration::from_millis(200)) {
                    Ok(Some(req)) => handle(&db, req, max_body),
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
        }));
    }
    if cfg.compact_interval > 0 {
        let db = db.clone();
        let every = Duration::from_secs(cfg.compact_interval);
        handles.push(std::thread::spawn(move || {
            let mut last = Instant::now();
            while !STOP.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() < every {
                    continue;
                }
                last = Instant::now();
                for c in db.list() {
                    let (garbage, segments) = c.garbage();
                    if garbage > 0.2 || segments > 8 {
                        match c.compact() {
                            Ok(r) => eprintln!(
                                "quiver: compacted {}: {:?} -> {:?} in {:.1}s",
                                c.name(),
                                r.inputs,
                                r.output,
                                r.seconds
                            ),
                            Err(e) => eprintln!("quiver: compaction of {} failed: {e}", c.name()),
                        }
                    }
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    eprintln!("quiver: shutting down, checkpointing collections");
    for c in db.list() {
        if let Err(e) = c.checkpoint() {
            eprintln!("quiver: checkpoint of {} failed: {e} (the WAL still has every write)", c.name());
        }
    }
    Ok(())
}
