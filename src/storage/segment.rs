//! Immutable segment files.
//!
//! A segment holds a sealed batch of rows: external ids, LSNs, full-precision
//! vectors, the frozen HNSW graph, payloads and (optionally) quantized codes.
//! The file is memory-mapped read-only; the graph and the vectors are used in
//! place, so opening a segment does not copy or rebuild anything.
//!
//! ```text
//! 0     header (512 bytes): magic "QVSEG001", version, dim, rows, metric,
//!       quantization, graph parameters, entry point, section table
//!       (offset, length, crc32c) x 10, header crc32c in the last 4 bytes
//! 512.. sections, each 64-byte aligned:
//!       0 ids u64[rows]        1 lsns u64[rows]      2 vectors f32[rows*dim]
//!       3 levels u8[rows]      4 links0 u32[rows*(1+2M)]
//!       5 upper_off u32[rows]  6 upper u32[..]
//!       7 payload_off u64[rows+1]  8 payload bytes (JSON per row)
//!       9 quantization blob
//! ```
//!
//! Every section is checksummed and every length and offset is validated on
//! open, so a damaged file is rejected instead of crashing a search.

use super::crc::{crc32c, crc32c_append};
use super::fs::{Mmap, Pod, SyncMode, cast, sync_dir, sync_file};
use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::index::graph::{FrozenGraph, FrozenParts, GraphParams};
use crate::index::hnsw::F32Space;
use crate::quant::{Codes, Quantization, pq::Pq, sq8::Sq8};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"QVSEG001";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 512;
const NSEC: usize = 10;
const ALIGN: u64 = 64;

const IDS: usize = 0;
const LSNS: usize = 1;
const VECTORS: usize = 2;
const LEVELS: usize = 3;
const LINKS0: usize = 4;
const UPPER_OFF: usize = 5;
const UPPER: usize = 6;
const PAYLOAD_OFF: usize = 7;
const PAYLOAD: usize = 8;
const QUANT: usize = 9;

fn metric_code(m: Metric) -> u8 {
    match m {
        Metric::L2 => 0,
        Metric::Ip => 1,
        Metric::Cosine => 2,
    }
}

fn metric_from(c: u8) -> Option<Metric> {
    [Metric::L2, Metric::Ip, Metric::Cosine].get(c as usize).copied()
}

pub(crate) fn bytes_of<T: Pod>(s: &[T]) -> &[u8] {
    // SAFETY: Pod types have no padding and any byte view of them is valid.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Everything needed to write a segment.
pub struct SegmentWrite<'a> {
    pub dim: usize,
    pub metric: Metric,
    pub params: GraphParams,
    pub ids: &'a [u64],
    pub lsns: &'a [u64],
    /// Prepared (normalised, for cosine) vectors, `ids.len() * dim` floats.
    pub vectors: &'a [f32],
    pub graph: &'a FrozenParts,
    /// JSON bytes per row; empty means "no payload".
    pub payloads: &'a [Vec<u8>],
    pub codes: Option<&'a Codes>,
}

fn quant_blob(codes: Option<&Codes>) -> (u8, u16, Vec<u8>) {
    let mut b = Vec::new();
    match codes {
        None => (0, 0, b),
        Some(Codes::Int8(s)) => {
            b.extend_from_slice(&(s.center().len() as u32).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(bytes_of(s.center()));
            b.extend_from_slice(bytes_of(s.scales()));
            b.extend_from_slice(bytes_of(s.codes()));
            (1, 0, b)
        }
        Some(Codes::Pq(p)) => {
            b.extend_from_slice(&(p.m() as u32).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(bytes_of(p.centroids()));
            b.extend_from_slice(p.codes());
            (2, p.m() as u16, b)
        }
    }
}

/// Writes a segment to `path` (via a temporary file and rename) and returns
/// its size in bytes.
pub fn write_segment(path: &Path, s: &SegmentWrite<'_>, mode: SyncMode) -> Result<u64> {
    let rows = s.ids.len();
    assert_eq!(s.lsns.len(), rows);
    assert_eq!(s.vectors.len(), rows * s.dim);
    assert_eq!(s.graph.levels.len(), rows);
    assert_eq!(s.payloads.len(), rows);
    let mut pay_off = Vec::with_capacity(rows + 1);
    let mut acc = 0u64;
    pay_off.push(0u64);
    for p in s.payloads {
        acc += p.len() as u64;
        pay_off.push(acc);
    }
    let (qkind, qm, qblob) = quant_blob(s.codes);

    let tmp = path.with_extension("qseg.tmp");
    let mut f = std::io::BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    f.write_all(&[0u8; HEADER_LEN])?;
    let mut pos = HEADER_LEN as u64;
    let mut table = [(0u64, 0u64, 0u32); NSEC];
    let mut put = |f: &mut std::io::BufWriter<File>, idx: usize, parts: &[&[u8]]| -> std::io::Result<()> {
        let pad = (ALIGN - pos % ALIGN) % ALIGN;
        f.write_all(&vec![0u8; pad as usize])?;
        pos += pad;
        let mut crc = 0u32;
        let mut len = 0u64;
        for p in parts {
            f.write_all(p)?;
            crc = crc32c_append(crc, p);
            len += p.len() as u64;
        }
        table[idx] = (pos, len, crc);
        pos += len;
        Ok(())
    };
    put(&mut f, IDS, &[bytes_of(s.ids)])?;
    put(&mut f, LSNS, &[bytes_of(s.lsns)])?;
    put(&mut f, VECTORS, &[bytes_of(s.vectors)])?;
    put(&mut f, LEVELS, &[&s.graph.levels])?;
    put(&mut f, LINKS0, &[bytes_of(&s.graph.links0)])?;
    put(&mut f, UPPER_OFF, &[bytes_of(&s.graph.upper_off)])?;
    put(&mut f, UPPER, &[bytes_of(&s.graph.upper)])?;
    put(&mut f, PAYLOAD_OFF, &[bytes_of(&pay_off)])?;
    let pay_parts: Vec<&[u8]> = s.payloads.iter().map(|p| p.as_slice()).collect();
    put(&mut f, PAYLOAD, &pay_parts)?;
    put(&mut f, QUANT, &[&qblob])?;

    let mut h = vec![0u8; HEADER_LEN];
    h[0..8].copy_from_slice(MAGIC);
    h[8..12].copy_from_slice(&VERSION.to_le_bytes());
    h[12..16].copy_from_slice(&(s.dim as u32).to_le_bytes());
    h[16..24].copy_from_slice(&(rows as u64).to_le_bytes());
    h[24] = metric_code(s.metric);
    h[25] = qkind;
    h[26..28].copy_from_slice(&qm.to_le_bytes());
    h[28..32].copy_from_slice(&(s.params.m as u32).to_le_bytes());
    h[32..36].copy_from_slice(&(s.params.ef_construction as u32).to_le_bytes());
    h[40..48].copy_from_slice(&s.params.seed.to_le_bytes());
    let (eid, elv) = s.graph.entry.map_or((u32::MAX, u32::MAX), |(e, l)| (e, l as u32));
    h[48..52].copy_from_slice(&eid.to_le_bytes());
    h[52..56].copy_from_slice(&elv.to_le_bytes());
    h[56..60].copy_from_slice(&(NSEC as u32).to_le_bytes());
    for (i, (off, len, crc)) in table.iter().enumerate() {
        let b = 64 + i * 24;
        h[b..b + 8].copy_from_slice(&off.to_le_bytes());
        h[b + 8..b + 16].copy_from_slice(&len.to_le_bytes());
        h[b + 16..b + 20].copy_from_slice(&crc.to_le_bytes());
    }
    let hcrc = crc32c(&h[..HEADER_LEN - 4]);
    h[HEADER_LEN - 4..].copy_from_slice(&hcrc.to_le_bytes());
    let mut file = f.into_inner().map_err(|e| e.into_error())?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&h)?;
    sync_file(&file, mode)?;
    drop(file);
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        sync_dir(dir, mode)?;
    }
    Ok(pos)
}

/// A mapped, validated segment file.
pub struct SegmentFile {
    map: Mmap,
    dim: usize,
    rows: usize,
    metric: Metric,
    params: GraphParams,
    entry: Option<(u32, usize)>,
    sec: [(usize, usize); NSEC],
    codes: Option<Codes>,
}

fn corrupt<T>(path: &Path, what: impl std::fmt::Display) -> Result<T> {
    Err(Error::Corrupt(format!("{}: {what}", path.display())))
}

impl SegmentFile {
    /// Maps and validates a segment (checksums of the header and of every
    /// section, section sizes, graph shape and payload offsets).
    pub fn open(path: &Path) -> Result<SegmentFile> {
        let f = File::open(path)?;
        let map = Mmap::open(&f)?;
        let b = map.as_slice();
        if b.len() < HEADER_LEN || &b[0..8] != MAGIC {
            return corrupt(path, "not a quiver segment");
        }
        let h = &b[..HEADER_LEN];
        let rd32 = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let rd64 = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        if crc32c(&h[..HEADER_LEN - 4]) != rd32(HEADER_LEN - 4) {
            return corrupt(path, "header checksum mismatch");
        }
        if rd32(8) != VERSION {
            return corrupt(path, format!("unsupported version {}", rd32(8)));
        }
        let dim = rd32(12) as usize;
        let rows = rd64(16) as usize;
        let Some(metric) = metric_from(h[24]) else { return corrupt(path, "unknown metric") };
        let qkind = h[25];
        let qm = u16::from_le_bytes([h[26], h[27]]) as usize;
        let params = GraphParams { m: rd32(28) as usize, ef_construction: rd32(32) as usize, seed: rd64(40) };
        if params.validate().is_err() || dim == 0 || rd32(56) as usize != NSEC {
            return corrupt(path, "invalid parameters in header");
        }
        let (eid, elv) = (rd32(48), rd32(52));
        let entry = if eid == u32::MAX { None } else { Some((eid, elv as usize)) };
        let mut sec = [(0usize, 0usize); NSEC];
        for (i, s) in sec.iter_mut().enumerate() {
            let o = 64 + i * 24;
            let (off, len, crc) = (rd64(o) as usize, rd64(o + 8) as usize, rd32(o + 16));
            if off % ALIGN as usize != 0 || off < HEADER_LEN || off.checked_add(len).is_none_or(|e| e > b.len()) {
                return corrupt(path, format!("section {i} out of bounds"));
            }
            if crc32c(&b[off..off + len]) != crc {
                return corrupt(path, format!("section {i} checksum mismatch"));
            }
            *s = (off, len);
        }
        let m0 = 2 * params.m;
        let expect = [
            (IDS, rows * 8),
            (LSNS, rows * 8),
            (VECTORS, rows * dim * 4),
            (LEVELS, rows),
            (LINKS0, rows * (1 + m0) * 4),
            (UPPER_OFF, rows * 4),
            (PAYLOAD_OFF, (rows + 1) * 8),
        ];
        for (i, want) in expect {
            if sec[i].1 != want {
                return corrupt(path, format!("section {i} has {} bytes, expected {want}", sec[i].1));
            }
        }
        let mut seg = SegmentFile { map, dim, rows, metric, params, entry, sec, codes: None };
        if cast::<u32>(seg.bytes(UPPER)).is_none() {
            return corrupt(path, "upper links section is ragged");
        }
        seg.graph().validate_shape().or_else(|e| corrupt(path, e))?;
        let offs = seg.payload_offsets();
        if offs[0] != 0 || offs.windows(2).any(|w| w[0] > w[1]) || offs[rows] as usize != seg.sec[PAYLOAD].1 {
            return corrupt(path, "payload offsets are inconsistent");
        }
        seg.codes = seg.decode_codes(qkind, qm).or_else(|e| corrupt(path, e))?;
        // Graph traversal touches vectors at random; don't read ahead.
        let (vo, vl) = seg.sec[VECTORS];
        seg.map.advise_random(vo..vo + vl);
        Ok(seg)
    }

    fn bytes(&self, i: usize) -> &[u8] {
        let (o, l) = self.sec[i];
        &self.map.as_slice()[o..o + l]
    }

    fn typed<T: Pod>(&self, i: usize) -> &[T] {
        cast(self.bytes(i)).expect("section validated on open")
    }

    fn decode_codes(&self, kind: u8, qm: usize) -> std::result::Result<Option<Codes>, String> {
        let b = self.bytes(QUANT);
        let (rows, dim) = (self.rows, self.dim);
        match kind {
            0 => Ok(None),
            1 => {
                let clen = u32::from_le_bytes(b.get(0..4).ok_or("short int8 blob")?.try_into().unwrap()) as usize;
                let need = 8 + clen * 4 + rows * 4 + rows * dim;
                if b.len() != need || !(clen == 0 || clen == dim) {
                    return Err("int8 blob has the wrong size".into());
                }
                let f32s = |r: std::ops::Range<usize>| -> Vec<f32> {
                    b[r].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
                };
                let center = f32s(8..8 + clen * 4);
                let scales = f32s(8 + clen * 4..8 + clen * 4 + rows * 4);
                let codes = b[8 + clen * 4 + rows * 4..].iter().map(|&x| x as i8).collect();
                Sq8::from_parts(dim, self.metric, center, codes, scales)
                    .map(Codes::Int8)
                    .map(Some)
                    .ok_or("bad int8 codes".into())
            }
            2 => {
                if qm == 0 || dim % qm != 0 {
                    return Err("bad pq m".into());
                }
                let clen = qm * crate::quant::pq::KS * (dim / qm);
                if b.len() != 8 + clen * 4 + rows * qm {
                    return Err("pq blob has the wrong size".into());
                }
                let cent = b[8..8 + clen * 4].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                let codes = b[8 + clen * 4..].to_vec();
                Pq::from_parts(dim, qm, self.metric, cent, codes).map(Codes::Pq).map(Some).ok_or("bad pq codes".into())
            }
            k => Err(format!("unknown quantization kind {k}")),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn params(&self) -> GraphParams {
        self.params
    }

    pub fn quantization(&self) -> Quantization {
        self.codes.as_ref().map_or(Quantization::None, |c| c.kind())
    }

    pub fn file_bytes(&self) -> usize {
        self.map.as_slice().len()
    }

    pub fn ids(&self) -> &[u64] {
        self.typed(IDS)
    }

    pub fn lsns(&self) -> &[u64] {
        self.typed(LSNS)
    }

    pub fn vectors(&self) -> &[f32] {
        self.typed(VECTORS)
    }

    pub fn vector(&self, row: u32) -> &[f32] {
        &self.vectors()[row as usize * self.dim..(row as usize + 1) * self.dim]
    }

    pub fn space(&self) -> F32Space<'_> {
        F32Space::over(self.vectors(), self.dim, self.metric)
    }

    pub fn graph(&self) -> FrozenGraph<'_> {
        FrozenGraph {
            m: self.params.m,
            m0: 2 * self.params.m,
            entry: self.entry,
            levels: self.typed(LEVELS),
            links0: self.typed(LINKS0),
            upper_off: self.typed(UPPER_OFF),
            upper: self.typed(UPPER),
        }
    }

    fn payload_offsets(&self) -> &[u64] {
        self.typed(PAYLOAD_OFF)
    }

    /// The JSON payload of a row (empty if it has none).
    pub fn payload(&self, row: u32) -> &[u8] {
        let o = self.payload_offsets();
        let (s, e) = (o[row as usize] as usize, o[row as usize + 1] as usize);
        &self.bytes(PAYLOAD)[s..e]
    }

    pub fn codes(&self) -> Option<&Codes> {
        self.codes.as_ref()
    }
}

const DEL_MAGIC: &[u8; 8] = b"QVDEL001";

/// Writes a tombstone bitmap for a segment of `rows` rows (atomically).
pub fn write_tombstones(path: &Path, rows: usize, words: &[u64], mode: SyncMode) -> Result<()> {
    let body = bytes_of(words);
    let mut b = Vec::with_capacity(24 + body.len());
    b.extend_from_slice(DEL_MAGIC);
    b.extend_from_slice(&(rows as u64).to_le_bytes());
    b.extend_from_slice(&crc32c(body).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(body);
    super::fs::write_atomic(path, &b, mode)?;
    Ok(())
}

/// Reads a tombstone bitmap written by [`write_tombstones`].
pub fn read_tombstones(path: &Path, rows: usize) -> Result<Vec<u64>> {
    let b = std::fs::read(path)?;
    let words = rows.div_ceil(64);
    if b.len() != 24 + words * 8
        || &b[..8] != DEL_MAGIC
        || u64::from_le_bytes(b[8..16].try_into().unwrap()) != rows as u64
    {
        return corrupt(path, "malformed tombstone file");
    }
    if crc32c(&b[24..]) != u32::from_le_bytes(b[16..20].try_into().unwrap()) {
        return corrupt(path, "tombstone checksum mismatch");
    }
    Ok(b[24..].as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect())
}

#[cfg(test)]
mod tests;
