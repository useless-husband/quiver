//! Durable file operations.
//!
//! On macOS `fsync(2)` only moves data to the drive, which may keep it in a
//! volatile cache; `fcntl(F_FULLFSYNC)` asks the drive to flush that cache
//! too. [`SyncMode::Full`] uses it on Apple platforms (and `fdatasync` on
//! Linux, where the kernel already issues the cache flush).

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SyncMode {
    /// F_FULLFSYNC on macOS, fdatasync elsewhere. Survives power loss.
    #[default]
    Full,
    /// Plain fsync. On macOS this survives a crash of the process or the
    /// kernel but not necessarily a power cut.
    Fsync,
    /// No syncing: acknowledged writes survive a crash of the quiver process
    /// (they are in the page cache) but not of the machine. Benchmarks only.
    None,
}

impl SyncMode {
    pub fn parse(s: &str) -> Option<SyncMode> {
        match s {
            "full" => Some(SyncMode::Full),
            "fsync" => Some(SyncMode::Fsync),
            "none" | "off" => Some(SyncMode::None),
            _ => None,
        }
    }
}

#[cfg(target_vendor = "apple")]
fn full_sync(f: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: plain fcntl on a valid descriptor.
    let r = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_FULLFSYNC) };
    if r == -1 {
        // Not every file system supports it (for example some network
        // mounts); fall back to fsync rather than failing the write.
        return f.sync_all();
    }
    Ok(())
}

#[cfg(not(target_vendor = "apple"))]
fn full_sync(f: &File) -> io::Result<()> {
    f.sync_data()
}

/// Makes the contents of `f` durable according to `mode`.
pub fn sync_file(f: &File, mode: SyncMode) -> io::Result<()> {
    match mode {
        SyncMode::Full => full_sync(f),
        SyncMode::Fsync => f.sync_all(),
        SyncMode::None => Ok(()),
    }
}

/// Makes a directory's entries (creations, renames, deletions) durable.
pub fn sync_dir(dir: &Path, mode: SyncMode) -> io::Result<()> {
    if mode == SyncMode::None {
        return Ok(());
    }
    let d = File::open(dir)?;
    match mode {
        SyncMode::Full => full_sync(&d),
        _ => d.sync_all(),
    }
}

/// Replaces `path` with `bytes` atomically: write a temporary file, sync it,
/// rename it over the target, sync the directory. A crash leaves either the
/// old or the new contents, never a mixture.
pub fn write_atomic(path: &Path, bytes: &[u8], mode: SyncMode) -> io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let tmp = path.with_extension(format!("{}.tmp", path.extension().and_then(|e| e.to_str()).unwrap_or("")));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        sync_file(&f, mode)?;
    }
    std::fs::rename(&tmp, path)?;
    sync_dir(dir, mode)
}

/// An exclusive advisory lock (`flock`) on `dir/LOCK`, held until dropped.
/// It stops a second process (or a second open in this one) from using the
/// same directory, which would corrupt the WAL and the manifest.
pub struct DirLock {
    _file: File,
}

impl DirLock {
    pub fn acquire(dir: &Path) -> io::Result<DirLock> {
        use std::os::fd::AsRawFd;
        let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("LOCK"))?;
        // SAFETY: flock on a descriptor we own.
        let r = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if r != 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("{} is in use by another quiver instance", dir.display()),
            ));
        }
        Ok(DirLock { _file: file })
    }
}

/// A read-only memory map of a whole file.
pub struct Mmap {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is read-only and immutable for its lifetime (segment
// files are never modified after they are written).
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(f: &File) -> io::Result<Mmap> {
        use std::os::fd::AsRawFd;
        let len = f.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mmap { ptr: std::ptr::null_mut(), len: 0 });
        }
        // SAFETY: mapping a regular file read-only; the result is checked.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Mmap { ptr, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` maps `len` readable bytes until `drop`.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// Hints that `range` will be accessed randomly (graph traversal), so the
    /// kernel should not read ahead.
    pub fn advise_random(&self, range: std::ops::Range<usize>) {
        if self.len == 0 || range.is_empty() {
            return;
        }
        let page = 16384usize;
        let start = range.start / page * page;
        let end = range.end.min(self.len);
        // SAFETY: advisory call on our own mapping; errors are ignored.
        unsafe {
            libc::madvise((self.ptr as *mut u8).add(start) as *mut libc::c_void, end - start, libc::MADV_RANDOM);
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: unmapping exactly what `open` mapped.
            unsafe { libc::munmap(self.ptr, self.len) };
        }
    }
}

/// Plain-old-data types that may be viewed directly in a mapped file.
pub trait Pod: Copy + 'static {}
impl Pod for u8 {}
impl Pod for i8 {}
impl Pod for u32 {}
impl Pod for u64 {}
impl Pod for f32 {}

/// Reinterprets bytes as a slice of `T`; `None` if misaligned or ragged.
pub fn cast<T: Pod>(b: &[u8]) -> Option<&[T]> {
    let size = std::mem::size_of::<T>();
    if !b.len().is_multiple_of(size) || !(b.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return None;
    }
    // SAFETY: alignment and length checked; every bit pattern is a valid `T`
    // for the Pod types above; the files are little-endian and so is every
    // supported target (see the compile-time check in storage/mod.rs).
    Some(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, b.len() / size) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_and_mmap() {
        let dir = std::env::temp_dir().join(format!("quiver-fs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.json");
        write_atomic(&p, b"hello", SyncMode::Full).unwrap();
        write_atomic(&p, b"world!!!", SyncMode::Fsync).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"world!!!");
        assert!(!dir.join("x.json.tmp").exists());
        let m = Mmap::open(&File::open(&p).unwrap()).unwrap();
        assert_eq!(m.as_slice(), b"world!!!");
        m.advise_random(0..8);
        let empty = dir.join("e");
        File::create(&empty).unwrap();
        assert!(Mmap::open(&File::open(&empty).unwrap()).unwrap().as_slice().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
        let words = [1u64, 2, 3];
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, 24) };
        assert_eq!(cast::<u64>(bytes), Some(&words[..]));
        assert!(cast::<u64>(&bytes[1..9]).is_none());
        assert_eq!(SyncMode::parse("full"), Some(SyncMode::Full));
    }
}
