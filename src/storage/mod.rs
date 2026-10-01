//! Durable storage: write-ahead log, segment files, tombstone files and the
//! file-system primitives (checksums, F_FULLFSYNC, mmap) under them.

pub mod crc;
pub mod fs;
pub mod segment;
pub mod wal;

// Segment files are little-endian and mapped directly into typed slices.
#[cfg(not(target_endian = "little"))]
compile_error!("quiver's on-disk format assumes a little-endian target");

pub use fs::SyncMode;
