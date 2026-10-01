//! Bitsets: a plain one for filter results and an atomic one for tombstones.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
}

impl Bitset {
    pub fn new(len: usize) -> Bitset {
        Bitset { words: vec![0; len.div_ceil(64)], len }
    }

    pub fn full(len: usize) -> Bitset {
        let mut b = Bitset { words: vec![u64::MAX; len.div_ceil(64)], len };
        b.trim();
        b
    }

    fn trim(&mut self) {
        let rem = self.len % 64;
        if rem != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        debug_assert!(i < self.len);
        self.words[i / 64] |= 1 << (i % 64);
    }

    #[inline]
    pub fn clear(&mut self, i: usize) {
        self.words[i / 64] &= !(1 << (i % 64));
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        i < self.len && self.words[i / 64] >> (i % 64) & 1 == 1
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn and_with(&mut self, o: &Bitset) {
        for (a, b) in self.words.iter_mut().zip(&o.words) {
            *a &= b;
        }
    }

    pub fn or_with(&mut self, o: &Bitset) {
        for (a, b) in self.words.iter_mut().zip(&o.words) {
            *a |= b;
        }
    }

    pub fn negate(&mut self) {
        for w in self.words.iter_mut() {
            *w = !*w;
        }
        self.trim();
    }

    /// Clears every bit that is set in the atomic set `o`.
    pub fn and_not_atomic(&mut self, o: &AtomicBitset) {
        for (i, w) in self.words.iter_mut().enumerate() {
            *w &= !o.word(i);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                Some(wi * 64 + b)
            })
        })
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

/// A fixed-size bitset that many threads can set concurrently.
pub struct AtomicBitset {
    words: Box<[AtomicU64]>,
    len: usize,
}

impl AtomicBitset {
    pub fn new(len: usize) -> AtomicBitset {
        AtomicBitset { words: (0..len.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(), len }
    }

    pub fn from_words(words: &[u64], len: usize) -> AtomicBitset {
        let mut w: Vec<AtomicU64> = words.iter().map(|&x| AtomicU64::new(x)).collect();
        w.resize_with(len.div_ceil(64), || AtomicU64::new(0));
        AtomicBitset { words: w.into_boxed_slice(), len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sets bit `i`; returns true if it was clear before.
    #[inline]
    pub fn set(&self, i: usize) -> bool {
        assert!(i < self.len, "bit {i} out of range {}", self.len);
        let m = 1u64 << (i % 64);
        self.words[i / 64].fetch_or(m, Ordering::AcqRel) & m == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        i < self.len && self.words[i / 64].load(Ordering::Acquire) >> (i % 64) & 1 == 1
    }

    #[inline]
    fn word(&self, i: usize) -> u64 {
        self.words.get(i).map_or(0, |w| w.load(Ordering::Acquire))
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.load(Ordering::Acquire).count_ones() as usize).sum()
    }

    pub fn snapshot(&self) -> Vec<u64> {
        self.words.iter().map(|w| w.load(Ordering::Acquire)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_ops_and_iteration() {
        let mut a = Bitset::new(130);
        for i in [0, 5, 63, 64, 129] {
            a.set(i);
        }
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![0, 5, 63, 64, 129]);
        assert_eq!(a.count(), 5);
        let mut n = a.clone();
        n.negate();
        assert_eq!(n.count(), 125);
        assert!(!n.get(129) && n.get(128) && !n.get(130));
        let full = Bitset::full(130);
        assert_eq!(full.count(), 130);
        let mut b = full.clone();
        b.and_with(&a);
        assert_eq!(b, a);
        let t = AtomicBitset::new(130);
        assert!(t.set(64));
        assert!(!t.set(64));
        let mut c = a.clone();
        c.and_not_atomic(&t);
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![0, 5, 63, 129]);
    }
}
