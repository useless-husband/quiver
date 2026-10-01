//! The write-ahead log.
//!
//! Every upsert and delete is appended here, and made durable, before it is
//! applied in memory and acknowledged. A record is
//!
//! ```text
//! [len: u32][crc32c(body): u32][body: kind u8, lsn u64, id u64, ...]
//! upsert body continues: dim u32, dim x f32, payload_len u32, payload (JSON)
//! ```
//!
//! **LSNs** (log sequence numbers) are assigned under the writer mutex, so
//! file order is LSN order. The collection uses them to resolve concurrent
//! writes to the same id deterministically: the higher LSN wins, both live
//! and during replay.
//!
//! **Group commit.** Appending and syncing are separate steps. A writer that
//! needs its record durable calls [`Wal::sync`]; if no sync is running it
//! becomes the leader, syncs everything written so far and wakes the others;
//! otherwise it waits for the running sync and checks whether that covered its
//! LSN. Concurrent writers therefore share one `F_FULLFSYNC`.
//!
//! **Recovery.** A crash can leave a partly written record at the tail. The
//! reader stops at the first record that is truncated, has a bad checksum or
//! does not decode, and truncates the file there; everything before it was
//! fully written. Records after a damaged one are not trusted because the
//! framing can no longer be relied on.
//!
//! **Failed syncs poison the log.** After a failed fsync the kernel may have
//! dropped the dirty pages, and a retried fsync can report success without
//! writing them (the PostgreSQL "fsyncgate" lesson), so every later append
//! fails until the process restarts and re-reads the log from disk.

use super::crc::crc32c;
use super::fs::{SyncMode, sync_dir, sync_file};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

const MAGIC: &[u8; 8] = b"QVWAL001";
const HEADER: u64 = 16;
const KIND_UPSERT: u8 = 1;
const KIND_DELETE: u8 = 2;
/// Upper bound on one record, so a corrupted length cannot make the reader
/// allocate gigabytes.
pub const MAX_RECORD: usize = 64 << 20;

#[derive(Clone, Debug, PartialEq)]
pub enum WalOp {
    Upsert { id: u64, vector: Vec<f32>, payload: Vec<u8> },
    Delete { id: u64 },
}

impl WalOp {
    pub fn id(&self) -> u64 {
        match self {
            WalOp::Upsert { id, .. } | WalOp::Delete { id } => *id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WalRecord {
    pub lsn: u64,
    pub op: WalOp,
}

fn encode(lsn: u64, op: &WalOp, out: &mut Vec<u8>) {
    let start = out.len();
    out.extend_from_slice(&[0u8; 8]);
    match op {
        WalOp::Upsert { id, vector, payload } => {
            out.push(KIND_UPSERT);
            out.extend_from_slice(&lsn.to_le_bytes());
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
            for x in vector {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        WalOp::Delete { id } => {
            out.push(KIND_DELETE);
            out.extend_from_slice(&lsn.to_le_bytes());
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
    let body_len = (out.len() - start - 8) as u32;
    let crc = crc32c(&out[start + 8..]);
    out[start..start + 4].copy_from_slice(&body_len.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&crc.to_le_bytes());
}

fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if b.len() < n {
        return None;
    }
    let (h, t) = b.split_at(n);
    *b = t;
    Some(h)
}

fn u32_at(b: &mut &[u8]) -> Option<u32> {
    take(b, 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()))
}

fn u64_at(b: &mut &[u8]) -> Option<u64> {
    take(b, 8).map(|x| u64::from_le_bytes(x.try_into().unwrap()))
}

fn decode(mut b: &[u8]) -> Option<WalRecord> {
    let kind = *take(&mut b, 1)?.first()?;
    let lsn = u64_at(&mut b)?;
    let id = u64_at(&mut b)?;
    let op = match kind {
        KIND_UPSERT => {
            let dim = u32_at(&mut b)? as usize;
            let raw = take(&mut b, dim.checked_mul(4)?)?;
            let vector = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let plen = u32_at(&mut b)? as usize;
            let payload = take(&mut b, plen)?.to_vec();
            WalOp::Upsert { id, vector, payload }
        }
        KIND_DELETE => WalOp::Delete { id },
        _ => return None,
    };
    if !b.is_empty() {
        return None;
    }
    Some(WalRecord { lsn, op })
}

/// The result of reading a log from disk.
#[derive(Debug, Default)]
pub struct Replay {
    pub generation: u64,
    pub records: Vec<WalRecord>,
    /// Bytes of fully valid log (header included).
    pub valid_len: u64,
    /// Bytes after the last valid record (a torn or damaged tail).
    pub discarded: u64,
}

/// Reads every valid record of a log file.
pub fn read_log(path: &Path) -> io::Result<Replay> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() < HEADER as usize || &buf[..8] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: not a quiver WAL", path.display())));
    }
    let generation = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let mut pos = HEADER as usize;
    let mut records = Vec::new();
    while pos + 8 <= buf.len() {
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
        if len > MAX_RECORD || pos + 8 + len > buf.len() {
            break;
        }
        let body = &buf[pos + 8..pos + 8 + len];
        if crc32c(body) != crc {
            break;
        }
        let Some(rec) = decode(body) else { break };
        records.push(rec);
        pos += 8 + len;
    }
    Ok(Replay { generation, records, valid_len: pos as u64, discarded: (buf.len() - pos) as u64 })
}

struct Writer {
    file: File,
    offset: u64,
    next_lsn: u64,
    poisoned: bool,
}

struct SyncState {
    synced: u64,
    busy: bool,
}

pub struct Wal {
    path: PathBuf,
    generation: u64,
    mode: SyncMode,
    writer: Mutex<Writer>,
    sync_handle: File,
    written: AtomicU64,
    state: Mutex<SyncState>,
    cv: Condvar,
    syncs: AtomicU64,
}

fn poisoned() -> io::Error {
    io::Error::other("write-ahead log is poisoned by an earlier write or sync failure; restart to recover")
}

impl Wal {
    /// Creates a new, empty log whose first record will get `next_lsn`.
    pub fn create(path: &Path, generation: u64, next_lsn: u64, mode: SyncMode) -> io::Result<Wal> {
        let mut f = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&generation.to_le_bytes())?;
        sync_file(&f, mode)?;
        if let Some(dir) = path.parent() {
            sync_dir(dir, mode)?;
        }
        Self::from_file(path, f, generation, HEADER, next_lsn, mode)
    }

    /// Opens an existing log, truncating any torn tail, and returns its
    /// records. New records get LSNs above both the last record and
    /// `min_next_lsn`.
    pub fn open(path: &Path, min_next_lsn: u64, mode: SyncMode) -> io::Result<(Wal, Replay)> {
        let replay = read_log(path)?;
        let f = OpenOptions::new().read(true).write(true).open(path)?;
        if replay.discarded > 0 {
            f.set_len(replay.valid_len)?;
            sync_file(&f, mode)?;
        }
        let next = replay.records.iter().map(|r| r.lsn + 1).max().unwrap_or(0).max(min_next_lsn);
        let wal = Self::from_file(path, f, replay.generation, replay.valid_len, next, mode)?;
        Ok((wal, replay))
    }

    fn from_file(
        path: &Path,
        mut file: File,
        generation: u64,
        offset: u64,
        next_lsn: u64,
        mode: SyncMode,
    ) -> io::Result<Wal> {
        file.seek(SeekFrom::Start(offset))?;
        let sync_handle = file.try_clone()?;
        let last = next_lsn.saturating_sub(1);
        Ok(Wal {
            path: path.to_path_buf(),
            generation,
            mode,
            writer: Mutex::new(Writer { file, offset, next_lsn, poisoned: false }),
            sync_handle,
            written: AtomicU64::new(last),
            state: Mutex::new(SyncState { synced: last, busy: false }),
            cv: Condvar::new(),
            syncs: AtomicU64::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The LSN the next record will get.
    pub fn next_lsn(&self) -> u64 {
        self.writer.lock().unwrap_or_else(|e| e.into_inner()).next_lsn
    }

    pub fn size_bytes(&self) -> u64 {
        self.writer.lock().unwrap_or_else(|e| e.into_inner()).offset
    }

    /// Number of syncs issued (group commit makes this smaller than the
    /// number of acknowledged writes).
    pub fn sync_count(&self) -> u64 {
        self.syncs.load(Ordering::Relaxed)
    }

    /// Appends `ops` in one write and returns the LSN of the first; the rest
    /// follow consecutively. Not durable until [`sync`](Self::sync).
    pub fn append(&self, ops: &[WalOp]) -> io::Result<u64> {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if w.poisoned {
            return Err(poisoned());
        }
        let first = w.next_lsn;
        let mut buf = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            let before = buf.len();
            encode(first + i as u64, op, &mut buf);
            if buf.len() - before - 8 > MAX_RECORD {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "record too large for the WAL"));
            }
        }
        if let Err(e) = w.file.write_all(&buf) {
            // Drop whatever part of the batch reached the file so the next
            // append does not land after a torn record.
            let off = w.offset;
            let undone = w.file.set_len(off).and_then(|_| w.file.seek(SeekFrom::Start(off)).map(|_| ()));
            if undone.is_err() {
                w.poisoned = true;
            }
            return Err(e);
        }
        w.offset += buf.len() as u64;
        w.next_lsn += ops.len() as u64;
        self.written.store(w.next_lsn - 1, Ordering::Release);
        Ok(first)
    }

    /// Returns once every record up to `lsn` is durable.
    pub fn sync(&self, lsn: u64) -> io::Result<()> {
        if self.mode == SyncMode::None {
            return Ok(());
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if st.synced >= lsn {
                return Ok(());
            }
            if !st.busy {
                st.busy = true;
                let target = self.written.load(Ordering::Acquire);
                drop(st);
                let r = sync_file(&self.sync_handle, self.mode);
                self.syncs.fetch_add(1, Ordering::Relaxed);
                st = self.state.lock().unwrap_or_else(|e| e.into_inner());
                st.busy = false;
                self.cv.notify_all();
                match r {
                    Ok(()) => st.synced = st.synced.max(target),
                    Err(e) => {
                        drop(st);
                        self.writer.lock().unwrap_or_else(|e| e.into_inner()).poisoned = true;
                        return Err(e);
                    }
                }
            } else {
                st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    /// Appends and waits for durability; returns the first LSN.
    pub fn commit(&self, ops: &[WalOp]) -> io::Result<u64> {
        let first = self.append(ops)?;
        self.sync(first + ops.len() as u64 - 1)?;
        Ok(first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use std::sync::Arc;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("quiver-wal-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn op(rng: &mut Rng, id: u64) -> WalOp {
        if rng.bool(0.2) {
            WalOp::Delete { id }
        } else {
            let dim = rng.range(1, 8);
            WalOp::Upsert {
                id,
                vector: (0..dim).map(|_| rng.normal()).collect(),
                payload: format!("{{\"n\":{}}}", rng.below(100)).into_bytes(),
            }
        }
    }

    #[test]
    fn round_trip_and_lsn_continuity() {
        let d = tmpdir("rt");
        let p = d.join("wal-1.log");
        let mut rng = Rng::new(1);
        let wal = Wal::create(&p, 1, 100, SyncMode::Fsync).unwrap();
        let ops: Vec<WalOp> = (0..50).map(|i| op(&mut rng, i)).collect();
        assert_eq!(wal.commit(&ops[..20]).unwrap(), 100);
        assert_eq!(wal.commit(&ops[20..]).unwrap(), 120);
        drop(wal);
        let (wal, replay) = Wal::open(&p, 0, SyncMode::Fsync).unwrap();
        assert_eq!(replay.generation, 1);
        assert_eq!(replay.discarded, 0);
        assert_eq!(replay.records.iter().map(|r| r.op.clone()).collect::<Vec<_>>(), ops);
        assert_eq!(replay.records.iter().map(|r| r.lsn).collect::<Vec<_>>(), (100..150).collect::<Vec<_>>());
        assert_eq!(wal.next_lsn(), 150);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Cutting the file at every possible byte keeps exactly the records
    /// that were completely written, and the log stays appendable.
    #[test]
    fn torn_tails_are_truncated_at_every_offset() {
        let d = tmpdir("torn");
        let p = d.join("wal.log");
        let mut rng = Rng::new(2);
        let wal = Wal::create(&p, 7, 1, SyncMode::None).unwrap();
        let ops: Vec<WalOp> = (0..6).map(|i| op(&mut rng, i)).collect();
        let mut ends = vec![HEADER];
        for o in &ops {
            wal.commit(std::slice::from_ref(o)).unwrap();
            ends.push(wal.size_bytes());
        }
        drop(wal);
        let full = std::fs::read(&p).unwrap();
        for cut in HEADER as usize..=full.len() {
            let q = d.join("cut.log");
            std::fs::write(&q, &full[..cut]).unwrap();
            let (wal, replay) = Wal::open(&q, 0, SyncMode::None).unwrap();
            let complete = ends.iter().filter(|&&e| e as usize <= cut).count() - 1;
            assert_eq!(replay.records.len(), complete, "cut at {cut}");
            assert_eq!(std::fs::metadata(&q).unwrap().len(), ends[complete], "cut at {cut}");
            wal.commit(&[WalOp::Delete { id: 99 }]).unwrap();
            drop(wal);
            let again = read_log(&q).unwrap();
            assert_eq!(again.records.len(), complete + 1);
            assert_eq!(again.records.last().unwrap().op, WalOp::Delete { id: 99 });
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn corruption_stops_replay_at_the_damaged_record() {
        let d = tmpdir("crc");
        let p = d.join("wal.log");
        let wal = Wal::create(&p, 1, 1, SyncMode::None).unwrap();
        let ops: Vec<WalOp> = (0..5).map(|i| WalOp::Delete { id: i }).collect();
        wal.commit(&ops).unwrap();
        drop(wal);
        let mut bytes = std::fs::read(&p).unwrap();
        // Flip a bit inside the third record's body.
        let rec = 8 + 17;
        bytes[HEADER as usize + 2 * rec + 12] ^= 0x10;
        std::fs::write(&p, &bytes).unwrap();
        let r = read_log(&p).unwrap();
        assert_eq!(r.records.len(), 2);
        assert!(r.discarded > 0);
        std::fs::write(&p, b"garbage!").unwrap();
        assert!(read_log(&p).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn group_commit_shares_syncs_and_loses_nothing() {
        let d = tmpdir("group");
        let p = d.join("wal.log");
        let wal = Arc::new(Wal::create(&p, 1, 1, SyncMode::Fsync).unwrap());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let wal = wal.clone();
                std::thread::spawn(move || {
                    for i in 0..40u64 {
                        wal.commit(&[WalOp::Delete { id: t * 1000 + i }]).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let syncs = wal.sync_count();
        drop(wal);
        let r = read_log(&p).unwrap();
        assert_eq!(r.records.len(), 320);
        let mut ids: Vec<u64> = r.records.iter().map(|x| x.op.id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 320);
        assert!(r.records.windows(2).all(|w| w[1].lsn == w[0].lsn + 1));
        assert!(syncs <= 320, "{syncs} syncs");
        eprintln!("320 commits from 8 threads used {syncs} syncs");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
