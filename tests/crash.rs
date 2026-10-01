//! Durability under SIGKILL.
//!
//! Writers upsert and delete through the HTTP API while the server is killed
//! with SIGKILL (by PID) at a random moment. After a restart every
//! acknowledged upsert must be present with its exact vector and payload and
//! every acknowledged delete must be gone. Requests that were in flight when
//! the server died may or may not have been applied; those ids are excluded.
//! Small segments make checkpoints (and compactions) happen during the load,
//! so kills also land in the middle of them.

mod common;

use common::{Server, tmpdir};
use quiver::client::request;
use quiver::rng::Rng;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
enum Expect {
    Live(Vec<f32>, u64),
    Gone,
    Unknown,
}

const DIM: usize = 8;

fn writer(addr: String, t: u64, seed: u64, stop: Arc<AtomicBool>, state: Arc<Mutex<HashMap<u64, Expect>>>) -> usize {
    let mut rng = Rng::new(seed ^ t);
    let mut mine: Vec<u64> = Vec::new();
    let mut acked = 0;
    let mut next = 0u64;
    while !stop.load(Ordering::Relaxed) {
        if !mine.is_empty() && rng.bool(0.15) {
            let n = rng.range(1, 3.min(mine.len()));
            let ids: Vec<u64> = (0..n).map(|_| mine.swap_remove(rng.below(mine.len()))).collect();
            {
                let mut s = state.lock().unwrap();
                for id in &ids {
                    s.insert(*id, Expect::Unknown);
                }
            }
            match request(&addr, "POST", "/collections/c/points/delete", Some(&json!({"ids": ids}))) {
                Ok(r) if r.status == 200 => {
                    let mut s = state.lock().unwrap();
                    for id in ids {
                        s.insert(id, Expect::Gone);
                    }
                    acked += 1;
                }
                Ok(r) => panic!("delete failed: {} {}", r.status, r.body),
                Err(_) => return acked,
            }
            continue;
        }
        let n = rng.range(1, 20);
        let mut pts = Vec::new();
        let mut batch = Vec::new();
        for _ in 0..n {
            // Disjoint id spaces per thread; occasionally re-upsert an own id.
            let id = if !mine.is_empty() && rng.bool(0.1) {
                mine[rng.below(mine.len())]
            } else {
                next += 1;
                t << 40 | next
            };
            if batch.iter().any(|(b, _)| *b == id) {
                continue;
            }
            let v: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
            pts.push(json!({"id": id, "vector": v, "payload": {"t": t, "n": next}}));
            batch.push((id, v));
        }
        {
            let mut s = state.lock().unwrap();
            for (id, _) in &batch {
                s.insert(*id, Expect::Unknown);
            }
        }
        match request(&addr, "PUT", "/collections/c/points", Some(&json!({"points": pts}))) {
            Ok(r) if r.status == 200 => {
                let mut s = state.lock().unwrap();
                for (j, (id, v)) in batch.into_iter().enumerate() {
                    let n = pts[j]["payload"]["n"].as_u64().unwrap();
                    s.insert(id, Expect::Live(v, n));
                    if !mine.contains(&id) {
                        mine.push(id);
                    }
                }
                acked += 1;
            }
            Ok(r) => panic!("upsert failed: {} {}", r.status, r.body),
            Err(_) => return acked,
        }
    }
    acked
}

fn verify(s: &Server, state: &HashMap<u64, Expect>) -> (usize, usize) {
    let ids: Vec<u64> = state.iter().filter(|(_, e)| **e != Expect::Unknown).map(|(id, _)| *id).collect();
    let (mut live, mut gone) = (0, 0);
    for chunk in ids.chunks(5000) {
        let r = s.ok("POST", "/collections/c/points/get", Some(json!({"ids": chunk, "with_vector": true})));
        for (id, got) in chunk.iter().zip(r["points"].as_array().unwrap()) {
            match &state[id] {
                Expect::Live(v, n) => {
                    assert!(!got.is_null(), "acknowledged point {id} is missing after the crash");
                    let gv: Vec<f32> =
                        got["vector"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
                    assert_eq!(&gv, v, "point {id} has the wrong vector");
                    assert_eq!(got["payload"]["n"].as_u64(), Some(*n), "point {id} has the wrong payload");
                    live += 1;
                }
                Expect::Gone => {
                    assert!(got.is_null(), "acknowledged delete of {id} was lost: {got}");
                    gone += 1;
                }
                Expect::Unknown => unreachable!(),
            }
        }
    }
    (live, gone)
}

#[test]
fn acknowledged_writes_survive_sigkill() {
    let seed = std::env::var("QUIVER_TEST_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(20261002u64);
    eprintln!("crash test seed {seed} (set QUIVER_TEST_SEED to replay)");
    let mut rng = Rng::new(seed);
    let dir = tmpdir("crash");
    let data = dir.join("data");
    let state: Arc<Mutex<HashMap<u64, Expect>>> = Arc::new(Mutex::new(HashMap::new()));
    let rounds = std::env::var("QUIVER_CRASH_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut total_acks = 0;
    for round in 0..rounds {
        let s = Server::start(&data, &["--compact-interval", "1", "--build-threads", "2"]);
        if round == 0 {
            s.ok(
                "PUT",
                "/collections/c",
                Some(json!({"dim": DIM, "metric": "l2", "segment_rows": 400, "m": 8, "ef_construction": 40})),
            );
        } else {
            let (live, gone) = verify(&s, &state.lock().unwrap());
            eprintln!("round {round}: verified {live} live and {gone} deleted points after SIGKILL");
        }
        let stop = Arc::new(AtomicBool::new(false));
        let writers: Vec<_> = (0..6u64)
            .map(|t| {
                let (addr, stop, state) = (s.addr.clone(), stop.clone(), state.clone());
                let t = t + 1 + round as u64 * 16;
                std::thread::spawn(move || writer(addr, t, seed, stop, state))
            })
            .collect();
        // Maintenance traffic, so that kills also land inside checkpoints and
        // compactions.
        let maint = {
            let (addr, stop) = (s.addr.clone(), stop.clone());
            let mut mrng = Rng::new(seed ^ 0xC0 ^ round as u64);
            std::thread::spawn(move || {
                let mut done = (0, 0);
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(mrng.range(20, 150) as u64));
                    let compact = mrng.bool(0.5);
                    let path = if compact { "/collections/c/compact" } else { "/collections/c/checkpoint" };
                    match request(&addr, "POST", path, None) {
                        Ok(r) if r.status == 200 => {
                            if compact {
                                done.1 += 1
                            } else {
                                done.0 += 1
                            }
                        }
                        Ok(r) => panic!("{path}: {} {}", r.status, r.body),
                        Err(_) => break,
                    }
                }
                done
            })
        };
        std::thread::sleep(Duration::from_millis(rng.range(300, 1500) as u64));
        s.kill9();
        stop.store(true, Ordering::Relaxed);
        let acks: usize = writers.into_iter().map(|w| w.join().unwrap()).sum();
        let (cps, comps) = maint.join().unwrap();
        eprintln!(
            "round {round}: {acks} acknowledged writes, {cps} checkpoints and {comps} compactions before SIGKILL"
        );
        total_acks += acks;
    }
    let s = Server::start(&data, &["--compact-interval", "0"]);
    let st = state.lock().unwrap();
    let (live, gone) = verify(&s, &st);
    eprintln!("final: verified {live} live and {gone} deleted points; {total_acks} acknowledged requests in total");
    assert!(total_acks > 20 && live > 100, "the load was too small to mean anything");
    let stats: Value = s.ok("GET", "/collections/c", None);
    assert!(stats["points"].as_u64().unwrap() as usize >= live);
    let log = std::fs::read_to_string(&s.log).unwrap_or_default();
    eprintln!(
        "server log: {} compactions, {} sealed segments now",
        log.matches("compacted").count(),
        stats["segments"].as_array().map_or(0, |a| a.len() - 1)
    );
    drop(s);
    std::fs::remove_dir_all(&dir).unwrap();
}
