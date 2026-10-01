//! A one-byte lock for per-node link lists.
//!
//! An HNSW graph needs one lock per node (an insert rewrites the neighbour
//! lists of up to `M` other nodes), so a million-node graph needs a million
//! locks. `std::sync::Mutex` is 16 bytes on macOS; this lock is 1. Critical
//! sections are short (copying at most `2M` ids, or one neighbour-selection
//! pass), so the lock spins briefly and then yields to the scheduler. Readers
//! never take it: link lists are read with atomic loads (see `graph.rs`).

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub struct SpinLock(AtomicBool);

pub struct SpinGuard<'a>(&'a SpinLock);

impl SpinLock {
    pub const fn new() -> SpinLock {
        SpinLock(AtomicBool::new(false))
    }

    #[inline]
    pub fn lock(&self) -> SpinGuard<'_> {
        let mut spins = 0u32;
        loop {
            if self.0.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                return SpinGuard(self);
            }
            while self.0.load(Ordering::Relaxed) {
                if spins < 100 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                spins = spins.saturating_add(1);
            }
        }
    }
}

impl Drop for SpinGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.0.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::UnsafeCell;
    use std::sync::Arc;

    struct Shared {
        lock: SpinLock,
        value: UnsafeCell<u64>,
    }
    // SAFETY: `value` is only touched while `lock` is held.
    unsafe impl Sync for Shared {}

    #[test]
    fn mutual_exclusion() {
        let s = Arc::new(Shared { lock: SpinLock::new(), value: UnsafeCell::new(0) });
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let s = s.clone();
                std::thread::spawn(move || {
                    for _ in 0..100_000 {
                        let _g = s.lock.lock();
                        // SAFETY: guarded by the lock.
                        unsafe { *s.value.get() += 1 };
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(unsafe { *s.value.get() }, 400_000);
    }
}
