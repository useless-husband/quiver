//! Benchmark helpers shared by `quiver bench` and the test suite: dataset
//! files, timing and memory measurement.
//!
//! Datasets use the big-ann-benchmarks binary layout: `.fbin` is
//! `u32 rows, u32 dim, rows * dim f32`; `.ibin` is `u32 rows, u32 k,
//! rows * k i32` (ground-truth neighbour ids). All little-endian.

use crate::error::{Error, Result};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

fn header(buf: &[u8], path: &Path, elem: usize) -> Result<(usize, usize)> {
    if buf.len() < 8 {
        return Err(Error::Corrupt(format!(
            "{}: shorter than its header",
            path.display()
        )));
    }
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    if buf.len() != 8 + n * d * elem {
        return Err(Error::Corrupt(format!(
            "{}: header says {n} x {d} but the file has {} bytes",
            path.display(),
            buf.len()
        )));
    }
    Ok((n, d))
}

/// Reads an `.fbin` file: returns `(rows, dim, data)`.
pub fn read_fbin(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let (n, d) = header(&buf, path, 4)?;
    let data = buf[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((n, d, data))
}

/// Reads an `.ibin` file: returns `(rows, k, ids)`.
pub fn read_ibin(path: &Path) -> Result<(usize, usize, Vec<u32>)> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let (n, k) = header(&buf, path, 4)?;
    let data = buf[8..]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
        .collect();
    Ok((n, k, data))
}

pub fn write_fbin(path: &Path, dim: usize, data: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&((data.len() / dim) as u32).to_le_bytes())?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    for x in data {
        f.write_all(&x.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub fn write_ibin(path: &Path, k: usize, ids: &[u32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&((ids.len() / k) as u32).to_le_bytes())?;
    f.write_all(&(k as u32).to_le_bytes())?;
    for &x in ids {
        f.write_all(&(x as i32).to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// Peak resident set size of this process in bytes.
pub fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage writes into the struct we pass.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let v = ru.ru_maxrss as u64;
    // macOS reports bytes, Linux kilobytes.
    if cfg!(target_os = "macos") {
        v
    } else {
        v * 1024
    }
}

/// Runs `f(i)` for `i in 0..n` on `threads` threads (dynamic chunks of 16)
/// and returns the wall time in seconds.
pub fn parallel_for<F: Fn(usize) + Sync>(n: usize, threads: usize, f: F) -> f64 {
    let t0 = Instant::now();
    if threads <= 1 {
        for i in 0..n {
            f(i);
        }
        return t0.elapsed().as_secs_f64();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(16, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    for j in i..(i + 16).min(n) {
                        f(j);
                    }
                }
            });
        }
    });
    t0.elapsed().as_secs_f64()
}

/// Percentile of a sample (nearest-rank), `p` in `[0, 100]`.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbin_ibin_round_trip_and_reject_truncation() {
        let dir = std::env::temp_dir().join(format!("quiver-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data: Vec<f32> = (0..30).map(|i| i as f32 * 0.5).collect();
        write_fbin(&dir.join("a.fbin"), 3, &data).unwrap();
        assert_eq!(read_fbin(&dir.join("a.fbin")).unwrap(), (10, 3, data));
        let ids: Vec<u32> = (0..20).collect();
        write_ibin(&dir.join("a.ibin"), 4, &ids).unwrap();
        assert_eq!(read_ibin(&dir.join("a.ibin")).unwrap(), (5, 4, ids));
        let bytes = std::fs::read(dir.join("a.fbin")).unwrap();
        std::fs::write(dir.join("b.fbin"), &bytes[..bytes.len() - 1]).unwrap();
        assert!(read_fbin(&dir.join("b.fbin")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 99.0), 4.0);
    }
}
