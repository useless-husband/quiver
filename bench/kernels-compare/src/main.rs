//! Times quiver's kernels against NumKong 7 and SimSIMD 6 with one harness:
//! 32 L1-resident input pairs per dimension, the pair index passed through
//! `black_box`, median of 7 runs of ~2M calls. Every function is called
//! through a function pointer or an FFI call, none is inlined into the loop.
//!
//!     cargo run --release -- [--out kernels-compare.csv]
//!
//! Semantics differ and are part of the comparison: NumKong accumulates f32
//! inputs in f64 and returns f64; SimSIMD and quiver accumulate in f32.

use numkong::{Angular, Dot, Euclidean};
use simsimd::SpatialSimilarity;
use std::hint::black_box;
use std::time::Instant;

fn time<F: Fn(usize) -> f64>(pairs: usize, f: F) -> f64 {
    let reps = (2_000_000 / pairs).max(1);
    let mut sink = 0.0;
    for i in 0..pairs * 200 {
        sink += f(i % pairs);
    }
    let mut s = Vec::new();
    for _ in 0..7 {
        let t0 = Instant::now();
        for _ in 0..reps {
            for i in 0..pairs {
                sink += f(black_box(i));
            }
        }
        s.push(t0.elapsed().as_nanos() as f64 / (reps * pairs) as f64);
    }
    black_box(sink);
    s.sort_by(f64::total_cmp);
    s[3]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    numkong::capabilities::configure_thread();
    // Report what the libraries detected, so a slow result cannot be a
    // silent fall-back to their serial code.
    let nk = numkong::capabilities::available();
    eprintln!(
        "numkong capabilities {nk:#x} (NEON bit {}), dynamic dispatch {}; simsimd uses_neon {} uses_neon_i8 {} uses_sve {}",
        nk & numkong::capabilities::cap::NEON != 0,
        numkong::capabilities::uses_dynamic_dispatch(),
        simsimd::capabilities::uses_neon(),
        simsimd::capabilities::uses_neon_i8(),
        simsimd::capabilities::uses_sve()
    );
    let best = quiver::distance::kernels();
    let scalar = quiver::distance::by_name("scalar").unwrap();
    let mut rng = quiver::rng::Rng::new(7);
    let pairs = 32;
    let mut rows = vec!["kernel,impl,dim,ns_per_call".to_string()];
    println!("{:<9} {:>5} {:>10} {:>10} {:>10} {:>10}", "kernel", "dim", "scalar", best.name, "numkong", "simsimd");
    for dim in [32usize, 100, 128, 384, 768, 784, 1536] {
        let fa: Vec<f32> = (0..pairs * dim).map(|_| rng.normal()).collect();
        let fb: Vec<f32> = (0..pairs * dim).map(|_| rng.normal()).collect();
        let ia: Vec<i8> = (0..pairs * dim).map(|_| (rng.next_u64() % 255) as i8).collect();
        let ib: Vec<i8> = (0..pairs * dim).map(|_| (rng.next_u64() % 255) as i8).collect();
        let f = |v: &[f32], i: usize| -> &[f32] { unsafe { std::slice::from_raw_parts(v.as_ptr().add(i * dim), dim) } };
        let g = |v: &[i8], i: usize| -> &[i8] { unsafe { std::slice::from_raw_parts(v.as_ptr().add(i * dim), dim) } };
        let mut row = |kernel: &str, cells: [(&str, f64); 4]| {
            println!("{kernel:<9} {dim:>5} {:>10.2} {:>10.2} {:>10.2} {:>10.2}", cells[0].1, cells[1].1, cells[2].1, cells[3].1);
            for (imp, ns) in cells {
                rows.push(format!("{kernel},{imp},{dim},{ns:.3}"));
            }
        };
        row(
            "l2sq_f32",
            [
                ("quiver-scalar", time(pairs, |i| (scalar.l2sq_f32)(f(&fa, i), f(&fb, i)) as f64)),
                ("quiver-neon", time(pairs, |i| (best.l2sq_f32)(f(&fa, i), f(&fb, i)) as f64)),
                ("numkong", time(pairs, |i| <f32 as Euclidean>::sqeuclidean(f(&fa, i), f(&fb, i)).unwrap())),
                ("simsimd", time(pairs, |i| <f32 as SpatialSimilarity>::l2sq(f(&fa, i), f(&fb, i)).unwrap())),
            ],
        );
        row(
            "dot_f32",
            [
                ("quiver-scalar", time(pairs, |i| (scalar.dot_f32)(f(&fa, i), f(&fb, i)) as f64)),
                ("quiver-neon", time(pairs, |i| (best.dot_f32)(f(&fa, i), f(&fb, i)) as f64)),
                ("numkong", time(pairs, |i| <f32 as Dot>::dot(f(&fa, i), f(&fb, i)).unwrap())),
                ("simsimd", time(pairs, |i| <f32 as SpatialSimilarity>::dot(f(&fa, i), f(&fb, i)).unwrap())),
            ],
        );
        row(
            "cos_f32",
            [
                ("quiver-scalar", time(pairs, |i| (scalar.cosine_f32)(f(&fa, i), f(&fb, i)) as f64)),
                ("quiver-neon", time(pairs, |i| (best.cosine_f32)(f(&fa, i), f(&fb, i)) as f64)),
                ("numkong", time(pairs, |i| <f32 as Angular>::angular(f(&fa, i), f(&fb, i)).unwrap())),
                ("simsimd", time(pairs, |i| <f32 as SpatialSimilarity>::cos(f(&fa, i), f(&fb, i)).unwrap())),
            ],
        );
        row(
            "dot_i8",
            [
                ("quiver-scalar", time(pairs, |i| (scalar.dot_i8)(g(&ia, i), g(&ib, i)) as f64)),
                ("quiver-neon", time(pairs, |i| (best.dot_i8)(g(&ia, i), g(&ib, i)) as f64)),
                ("numkong", time(pairs, |i| <i8 as Dot>::dot(g(&ia, i), g(&ib, i)).unwrap() as f64)),
                ("simsimd", time(pairs, |i| <i8 as SpatialSimilarity>::dot(g(&ia, i), g(&ib, i)).unwrap())),
            ],
        );
        row(
            "l2sq_i8",
            [
                ("quiver-scalar", time(pairs, |i| (scalar.l2sq_i8)(g(&ia, i), g(&ib, i)) as f64)),
                ("quiver-neon", time(pairs, |i| (best.l2sq_i8)(g(&ia, i), g(&ib, i)) as f64)),
                ("numkong", time(pairs, |i| <i8 as Euclidean>::sqeuclidean(g(&ia, i), g(&ib, i)).unwrap() as f64)),
                ("simsimd", time(pairs, |i| <i8 as SpatialSimilarity>::l2sq(g(&ia, i), g(&ib, i)).unwrap())),
            ],
        );
    }
    // Agreement check: the libraries compute the same quantities.
    let (a, b) = (&fa_check(), &fb_check());
    let q = (best.l2sq_f32)(a, b) as f64;
    let n = <f32 as Euclidean>::sqeuclidean(a, b).unwrap();
    assert!((q - n).abs() <= 1e-4 * n.abs(), "l2sq disagrees: quiver {q}, numkong {n}");
    if let Some(p) = out {
        std::fs::write(&p, rows.join("\n") + "\n").unwrap();
        eprintln!("wrote {p}");
    }
}

fn fa_check() -> Vec<f32> {
    (0..300).map(|i| (i as f32 * 0.37).sin()).collect()
}

fn fb_check() -> Vec<f32> {
    (0..300).map(|i| (i as f32 * 0.11).cos()).collect()
}
