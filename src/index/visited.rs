//! The "visited" set of a graph search.
//!
//! A search touches a few thousand nodes out of millions, so clearing a bitmap
//! per search would dominate. Instead each slot holds the epoch in which it was
//! last visited; starting a search bumps the epoch, and the array is only
//! cleared when the 16-bit epoch wraps (once every 65535 searches).

pub struct VisitedSet {
    marks: Vec<u16>,
    epoch: u16,
}

impl VisitedSet {
    pub fn new(capacity: usize) -> VisitedSet {
        VisitedSet {
            marks: vec![0; capacity],
            epoch: 0,
        }
    }

    /// Starts a new search over ids `0..capacity`.
    pub fn reset(&mut self, capacity: usize) {
        if self.marks.len() < capacity {
            self.marks.resize(capacity, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// Marks `id`; returns true if it was not yet visited in this search.
    #[inline]
    pub fn insert(&mut self, id: u32) -> bool {
        let slot = &mut self.marks[id as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.marks[id as usize] == self.epoch
    }

    #[inline]
    pub fn prefetch(&self, id: u32) {
        if let Some(p) = self.marks.get(id as usize) {
            super::prefetch_read(p as *const u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_isolate_searches_across_wraparound() {
        let mut v = VisitedSet::new(8);
        for round in 0..70_000u32 {
            v.reset(8);
            let id = round % 8;
            assert!(v.insert(id), "round {round}");
            assert!(!v.insert(id));
            assert!(v.contains(id));
            assert!(!v.contains((id + 1) % 8));
        }
    }
}
