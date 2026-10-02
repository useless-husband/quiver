//! The library API: a durable collection with a payload filter, and the
//! in-memory index on its own.
//!
//!     cargo run --release --example library

use quiver::collection::{Collection, CollectionConfig, Options, Point, Query};
use quiver::payload::Filter;
use quiver::{GraphParams, HnswIndex, Metric};
use serde_json::json;
use std::path::Path;

fn main() -> quiver::Result<()> {
    let dir = std::env::temp_dir().join(format!("quiver-example-{}", std::process::id()));
    let c = Collection::create(&dir, "items", CollectionConfig::new(3, Metric::L2), Options::default())?;
    c.upsert(&[
        Point { id: 1, vector: vec![0.1, 0.2, 0.3], payload: Some(json!({"color": "red"})) },
        Point { id: 2, vector: vec![0.1, 0.2, 0.2], payload: Some(json!({"color": "blue"})) },
        Point { id: 3, vector: vec![0.9, 0.9, 0.9], payload: Some(json!({"color": "red"})) },
    ])?;
    let hits = c.search(&Query::new(&[0.1, 0.2, 0.25], 2))?.hits;
    println!("nearest: {:?}", hits.iter().map(|h| h.id).collect::<Vec<_>>());

    let red = Filter::parse(&json!({"field": "color", "eq": "red"}))?;
    let mut q = Query::new(&[0.1, 0.2, 0.25], 2);
    q.filter = Some(&red);
    println!("nearest red: {:?}", c.search(&q)?.hits.iter().map(|h| h.id).collect::<Vec<_>>());
    drop(c);
    // Reopening replays the write-ahead log.
    let c = Collection::open(&dir, "items", Options::default())?;
    println!("after reopen: {} points", c.len());
    drop(c);
    std::fs::remove_dir_all(Path::new(&dir))?;

    // The in-memory HNSW index alone, filled from 4 threads.
    let idx = HnswIndex::new(2, Metric::Cosine, GraphParams::default(), 1000);
    let data: Vec<f32> = (0..1000).flat_map(|i| [(i as f32).cos(), (i as f32).sin()]).collect();
    idx.insert_batch(&data, 4)?;
    let r = idx.search(&[1.0, 0.0], 3, 64)?;
    println!("index: {:?}", r.iter().map(|n| n.id).collect::<Vec<_>>());
    Ok(())
}
