//! quiver: a compact vector database.
//!
//! * [`distance`]: L2 / inner-product / cosine kernels (NEON, AVX2, scalar).
//! * [`index`]: a brute-force flat index and a concurrent HNSW graph.

pub mod distance;
pub mod error;
pub mod index;
pub mod rng;

#[cfg(test)]
pub(crate) mod testkit;

pub use distance::Metric;
pub use error::{Error, Result};
pub use index::{FlatIndex, GraphParams, HnswIndex, Neighbor};
