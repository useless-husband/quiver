//! Helpers for randomised tests: fixed seeds, reported on failure.
//!
//! Every property runs `cases` cases; case `i` draws from its own generator
//! seeded with `seed ^ splitmix64(i)`, so a failure names the one seed needed
//! to replay it. `QUIVER_TEST_SEED` overrides the base seed and
//! `QUIVER_TEST_CASES` scales the number of cases.

use crate::rng::{Rng, splitmix64};
use std::panic::{AssertUnwindSafe, catch_unwind};

pub const DEFAULT_SEED: u64 = 0x5155_4956_4552_0001; // "QUIVER" + 1

pub fn base_seed() -> u64 {
    std::env::var("QUIVER_TEST_SEED")
        .ok()
        .and_then(|s| {
            let s = s.trim();
            if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16).ok() } else { s.parse().ok() }
        })
        .unwrap_or(DEFAULT_SEED)
}

pub fn scaled_cases(default: usize) -> usize {
    std::env::var("QUIVER_TEST_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Runs `f` on `cases` seeded generators; panics with the failing seed.
pub fn property<F>(name: &str, cases: usize, f: F)
where
    F: Fn(&mut Rng) -> Result<(), String>,
{
    let base = base_seed();
    for i in 0..scaled_cases(cases) {
        let seed = base ^ splitmix64(i as u64);
        let mut rng = Rng::new(seed);
        let outcome = catch_unwind(AssertUnwindSafe(|| f(&mut rng)));
        let msg = match outcome {
            Ok(Ok(())) => continue,
            Ok(Err(m)) => m,
            Err(p) => p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into()),
        };
        panic!(
            "property `{name}` failed on case {i} (case seed {seed:#x}, base seed {base:#x}; \
             rerun with QUIVER_TEST_SEED={base:#x}): {msg}"
        );
    }
}

/// Random vectors with a mix of scales, the occasional exact zero and the
/// occasional large value.
pub fn random_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
    let scale = match rng.below(4) {
        0 => 1e-3,
        1 => 1.0,
        2 => 30.0,
        _ => 1e3,
    };
    (0..n)
        .map(|_| match rng.below(20) {
            0 => 0.0,
            1 => rng.normal() * scale * 50.0,
            _ => rng.normal() * scale,
        })
        .collect()
}

/// Gaussian blobs: `n` points around `clusters` centres in `dim` dimensions.
/// Clustered data is harder for HNSW than uniform noise and closer to real
/// embeddings.
pub fn clustered(rng: &mut Rng, n: usize, dim: usize, clusters: usize) -> Vec<f32> {
    // Centres spread 1.5x the within-cluster noise: clusters overlap, so the
    // graph has to route between them.
    let centres: Vec<f32> = (0..clusters * dim).map(|_| rng.normal() * 1.5).collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = rng.below(clusters);
        for d in 0..dim {
            out.push(centres[c * dim + d] + rng.normal());
        }
    }
    out
}
