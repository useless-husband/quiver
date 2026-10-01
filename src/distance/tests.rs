//! Every kernel table that this CPU can run is checked against an f64
//! reference (f32 kernels) or the exact scalar code (integer kernels), on
//! random lengths that include empty inputs, short tails, block boundaries and
//! unaligned starting addresses.

use super::*;
use crate::rng::Rng;
use crate::testkit::{property, random_f32};

const U: f64 = f32::EPSILON as f64 / 2.0; // unit round-off of f32

/// Lengths biased towards the cases SIMD code gets wrong: 0, tails of 1..15,
/// exact multiples of the vector width and one either side of them.
fn tricky_len(rng: &mut Rng) -> usize {
    match rng.below(10) {
        0..=4 => rng.below(41),
        5..=7 => {
            let base = 16 * rng.range(1, 40);
            (base as isize + rng.range(0, 2) as isize - 1).max(0) as usize
        }
        _ => rng.below(1101),
    }
}

/// Copies `v` into a fresh buffer at element offset `off`, so the slice
/// returned by `&buf[off..]` starts at an address that is not 16-byte aligned
/// for most offsets.
fn shifted<T: Copy + Default>(v: &[T], off: usize) -> Vec<T> {
    let mut buf = vec![T::default(); v.len() + off];
    buf[off..].copy_from_slice(v);
    buf
}

fn check_close(what: &str, got: f32, want: f64, bound: f64) -> Result<(), String> {
    let err = (got as f64 - want).abs();
    if err <= bound || (got.is_nan() && want.is_nan()) {
        Ok(())
    } else {
        Err(format!(
            "{what}: got {got}, want {want}, error {err:e} > bound {bound:e}"
        ))
    }
}

#[test]
fn f32_kernels_match_reference() {
    let tables = available();
    property("f32 kernels vs f64 reference", 3000, |rng| {
        let n = tricky_len(rng);
        let a = random_f32(rng, n);
        let b = random_f32(rng, n);
        let (oa, ob) = (rng.below(4), rng.below(4));
        let (ba, bb) = (shifted(&a, oa), shifted(&b, ob));
        let (sa, sb) = (&ba[oa..], &bb[ob..]);

        let mut l2 = 0.0f64;
        let (mut ab, mut aa, mut bbs, mut ab_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let (x, y) = (a[i] as f64, b[i] as f64);
            l2 += (x - y) * (x - y);
            ab += x * y;
            ab_abs += (x * y).abs();
            aa += x * x;
            bbs += y * y;
        }
        let gamma = 2.0 * (n as f64 + 4.0) * U;
        let cos_ref = if aa == 0.0 || bbs == 0.0 {
            1.0
        } else {
            (1.0 - ab / (aa * bbs).sqrt()).max(0.0)
        };
        for k in &tables {
            let ctx = |m: &str| format!("{} {m} n={n} offsets=({oa},{ob})", k.name);
            check_close(&ctx("l2sq"), (k.l2sq_f32)(sa, sb), l2, gamma * l2 + 1e-30)?;
            check_close(&ctx("dot"), (k.dot_f32)(sa, sb), ab, gamma * ab_abs + 1e-30)?;
            // |cos error| <= gamma * sum|a_i b_i| / (|a||b|) <= gamma, plus the
            // rounding of the norms and the square root.
            let both_nonzero = a.iter().any(|&x| x != 0.0) && b.iter().any(|&x| x != 0.0);
            if both_nonzero || n == 0 {
                check_close(
                    &ctx("cosine"),
                    (k.cosine_f32)(sa, sb),
                    cos_ref,
                    4.0 * gamma + 1e-6,
                )?;
            }
        }
        Ok(())
    });
}

fn random_i8(rng: &mut Rng, n: usize) -> Vec<i8> {
    let extreme = rng.bool(0.2);
    (0..n)
        .map(|_| {
            if extreme {
                if rng.bool(0.5) { -128 } else { 127 }
            } else {
                rng.next_u64() as i8
            }
        })
        .collect()
}

#[test]
fn i8_kernels_are_exact() {
    let tables = available();
    property("i8 kernels vs scalar", 3000, |rng| {
        let n = tricky_len(rng);
        let a = random_i8(rng, n);
        let b = random_i8(rng, n);
        let (oa, ob) = (rng.below(16), rng.below(16));
        let (ba, bb) = (shifted(&a, oa), shifted(&b, ob));
        let (sa, sb) = (&ba[oa..], &bb[ob..]);
        let dot = scalar::dot_i8(&a, &b);
        let l2 = scalar::l2sq_i8(&a, &b);
        for k in &tables {
            let got = (k.dot_i8)(sa, sb);
            if got != dot {
                return Err(format!("{} dot_i8 n={n}: got {got}, want {dot}", k.name));
            }
            let got = (k.l2sq_i8)(sa, sb);
            if got != l2 {
                return Err(format!("{} l2sq_i8 n={n}: got {got}, want {l2}", k.name));
            }
        }
        Ok(())
    });
}

/// Inputs long enough to cross the 2^16-element accumulator blocks, with the
/// values that maximise every per-lane sum.
#[test]
fn i8_kernels_do_not_overflow_on_long_extreme_inputs() {
    for n in [(1 << 16) - 1, 1 << 16, (1 << 16) + 17, 200_003] {
        let lo = vec![-128i8; n];
        let hi = vec![127i8; n];
        for k in available() {
            assert_eq!(
                (k.dot_i8)(&lo, &lo),
                16384 * n as i64,
                "{} dot n={n}",
                k.name
            );
            assert_eq!(
                (k.dot_i8)(&lo, &hi),
                -16256 * n as i64,
                "{} dot n={n}",
                k.name
            );
            assert_eq!(
                (k.l2sq_i8)(&lo, &hi),
                65025 * n as u64,
                "{} l2 n={n}",
                k.name
            );
        }
    }
}

#[test]
fn hamming_is_exact() {
    let tables = available();
    property("hamming vs scalar", 2000, |rng| {
        let n = match rng.below(3) {
            0 => rng.below(9),
            1 => rng.below(70),
            _ => rng.range(4090, 4110), // crosses the 2048-iteration block
        };
        let a: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
        let b: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
        let off = rng.below(3);
        let (ba, bb) = (shifted(&a, off), shifted(&b, 0));
        let want = scalar::hamming(&a, &b);
        for k in &tables {
            let got = (k.hamming)(&ba[off..], &bb);
            if got != want {
                return Err(format!("{} hamming n={n}: got {got}, want {want}", k.name));
            }
        }
        Ok(())
    });
    let ones = vec![u64::MAX; 9000];
    let zeros = vec![0u64; 9000];
    for k in available() {
        assert_eq!((k.hamming)(&ones, &zeros), 64 * 9000, "{}", k.name);
    }
}

#[test]
fn metric_semantics() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 2.0, 0.0];
    assert_eq!(Metric::L2.distance(&a, &b), 5.0);
    assert_eq!(Metric::Ip.distance(&a, &b), 0.0);
    assert!((Metric::Cosine.distance(&a, &b) - 1.0).abs() < 1e-6);
    let pa = Metric::Cosine.prepare(&[3.0, 4.0]);
    let pb = Metric::Cosine.prepare(&[4.0, 3.0]);
    let k = kernels();
    let d = Metric::Cosine.prepared_distance(k, &pa, &pb);
    assert!((d - (1.0 - 24.0 / 25.0)).abs() < 1e-6, "{d}");
    // Zero vectors stay zero and are at distance 1.
    assert_eq!(Metric::Cosine.prepare(&[0.0, 0.0]), vec![0.0, 0.0]);
    assert_eq!(Metric::Cosine.distance(&[0.0, 0.0], &[1.0, 1.0]), 1.0);
    assert_eq!(Metric::parse("angular"), Some(Metric::Cosine));
    assert_eq!(Metric::parse("nope"), None);
}

#[test]
fn best_table_is_selected() {
    if std::env::var_os("QUIVER_KERNELS").is_some() {
        return;
    }
    let best = kernels();
    assert!(available().iter().any(|k| std::ptr::eq(*k, best)));
    #[cfg(target_arch = "aarch64")]
    assert!(best.name.starts_with("neon"), "{}", best.name);
}
