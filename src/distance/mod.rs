//! Distance metrics and the SIMD kernels behind them.
//!
//! Kernels are plain functions collected in a [`Kernels`] table. The table is
//! chosen once at run time ([`kernels`]): NEON with `SDOT` when the CPU has
//! FEAT_DotProd, NEON with widening multiplies otherwise, AVX2+FMA on x86-64
//! when present, and the scalar reference everywhere else. Setting
//! `QUIVER_KERNELS=scalar` (or `neon`, `neon-dotprod`, `avx2`) forces a table,
//! which is how the benchmarks compare them inside the same binary.
//!
//! Every distance is "smaller is closer":
//! * `l2`: squared Euclidean distance,
//! * `ip`: negative inner product,
//! * `cosine`: `1 - cos(a, b)`; the index stores normalised vectors and uses
//!   `1 - dot`, the one-pass [`Kernels::cosine_f32`] serves unnormalised input.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "x86_64")]
pub mod x86;

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The similarity a collection is searched by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Inner product (maximum inner product search); distance is `-dot`.
    Ip,
    /// Cosine distance `1 - cos`; vectors are normalised on the way in.
    Cosine,
}

impl Metric {
    pub fn parse(s: &str) -> Option<Metric> {
        match s.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Some(Metric::L2),
            "ip" | "dot" | "inner_product" => Some(Metric::Ip),
            "cosine" | "angular" => Some(Metric::Cosine),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Ip => "ip",
            Metric::Cosine => "cosine",
        }
    }

    /// Whether stored and query vectors are normalised before use.
    pub fn normalizes(self) -> bool {
        self == Metric::Cosine
    }

    /// Distance between two vectors already prepared for this metric (that is,
    /// normalised for cosine).
    #[inline]
    pub fn prepared_distance(self, k: &Kernels, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => (k.l2sq_f32)(a, b),
            Metric::Ip => -(k.dot_f32)(a, b),
            Metric::Cosine => 1.0 - (k.dot_f32)(a, b),
        }
    }

    /// Distance between two raw (unprepared) vectors.
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        let k = kernels();
        match self {
            Metric::L2 => (k.l2sq_f32)(a, b),
            Metric::Ip => -(k.dot_f32)(a, b),
            Metric::Cosine => (k.cosine_f32)(a, b),
        }
    }

    /// Prepares a vector for storage or search under this metric.
    pub fn prepare(self, v: &[f32]) -> Vec<f32> {
        let mut out = v.to_vec();
        if self.normalizes() {
            normalize(&mut out);
        }
        out
    }
}

/// Scales `v` to unit length in place; a zero vector is left as it is.
pub fn normalize(v: &mut [f32]) {
    let n = (kernels().dot_f32)(v, v).sqrt();
    if n > 0.0 && n.is_finite() {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// `1 - ab / sqrt(aa * bb)`, with zero vectors at distance 1 from everything.
#[inline]
pub(crate) fn cosine_from_parts(ab: f32, aa: f32, bb: f32) -> f32 {
    if aa == 0.0 || bb == 0.0 {
        return 1.0;
    }
    (1.0 - ab / (aa * bb).sqrt()).max(0.0)
}

/// A table of distance kernels for one instruction set.
#[derive(Debug)]
pub struct Kernels {
    pub name: &'static str,
    pub l2sq_f32: fn(&[f32], &[f32]) -> f32,
    pub dot_f32: fn(&[f32], &[f32]) -> f32,
    pub cosine_f32: fn(&[f32], &[f32]) -> f32,
    pub dot_i8: fn(&[i8], &[i8]) -> i64,
    pub l2sq_i8: fn(&[i8], &[i8]) -> u64,
    pub hamming: fn(&[u64], &[u64]) -> u64,
}

static SCALAR: Kernels = Kernels {
    name: "scalar",
    l2sq_f32: scalar::l2sq_f32,
    dot_f32: scalar::dot_f32,
    cosine_f32: scalar::cosine_f32,
    dot_i8: scalar::dot_i8,
    l2sq_i8: scalar::l2sq_i8,
    hamming: scalar::hamming,
};

#[cfg(target_arch = "aarch64")]
static NEON: Kernels = Kernels {
    name: "neon",
    l2sq_f32: neon::l2sq_f32,
    dot_f32: neon::dot_f32,
    cosine_f32: neon::cosine_f32,
    dot_i8: neon::dot_i8_widening,
    l2sq_i8: neon::l2sq_i8_widening,
    hamming: neon::hamming,
};

// Only reachable through `available()`, which checks FEAT_DotProd first, so the
// safe wrappers below are never called on a CPU without it.
#[cfg(target_arch = "aarch64")]
static NEON_DOTPROD: Kernels = Kernels {
    name: "neon-dotprod",
    l2sq_f32: neon::l2sq_f32,
    dot_f32: neon::dot_f32,
    cosine_f32: neon::cosine_f32,
    dot_i8: |a, b| unsafe { neon::dot_i8_dotprod(a, b) },
    l2sq_i8: |a, b| unsafe { neon::l2sq_i8_dotprod(a, b) },
    hamming: neon::hamming,
};

// Only reachable through `available()`, which checks AVX2 and FMA first.
#[cfg(target_arch = "x86_64")]
static AVX2: Kernels = Kernels {
    name: "avx2",
    l2sq_f32: |a, b| unsafe { x86::l2sq_f32(a, b) },
    dot_f32: |a, b| unsafe { x86::dot_f32(a, b) },
    cosine_f32: |a, b| unsafe { x86::cosine_f32(a, b) },
    dot_i8: |a, b| unsafe { x86::dot_i8(a, b) },
    l2sq_i8: scalar::l2sq_i8,
    hamming: scalar::hamming,
};

/// Every kernel table this CPU can run, best last.
pub fn available() -> Vec<&'static Kernels> {
    #[allow(unused_mut)]
    let mut v: Vec<&'static Kernels> = vec![&SCALAR];
    #[cfg(target_arch = "aarch64")]
    {
        v.push(&NEON);
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            v.push(&NEON_DOTPROD);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            v.push(&AVX2);
        }
    }
    v
}

/// Looks up an available kernel table by name.
pub fn by_name(name: &str) -> Option<&'static Kernels> {
    available().into_iter().find(|k| k.name == name)
}

/// The kernel table used by the index: the best available one, unless
/// `QUIVER_KERNELS` names another.
pub fn kernels() -> &'static Kernels {
    static CHOSEN: OnceLock<&'static Kernels> = OnceLock::new();
    CHOSEN.get_or_init(|| {
        if let Ok(name) = std::env::var("QUIVER_KERNELS") {
            if let Some(k) = by_name(&name) {
                return k;
            }
            eprintln!(
                "quiver: QUIVER_KERNELS={name} is not available here; using the best kernels"
            );
        }
        *available()
            .last()
            .expect("scalar kernels are always available")
    })
}

#[cfg(test)]
mod tests;
