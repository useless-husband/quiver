# quiver design

This document explains how quiver is put together, which problems were hard,
and which alternatives were rejected. Numbers quoted here come from the
benchmark files in `bench/results/`; see [benchmarks.md](benchmarks.md) for how
they were produced.

## Layers

```
 HTTP (tiny_http)  CLI (quiver ...)  Rust API
        \              |              /
         +----- db::Database ------+          a directory of collections
                       |
              collection::Collection          WAL, id map, segments, checkpoint,
               /                \             compaction, recovery
     segments::Growing     segments::Sealed   (x N)
     HnswIndex (RAM)       SegmentFile (mmap) + tombstones + payload index
               \                /
          index::graph  (HNSW topology, generic over a Space)
               |
   Space impls: F32Space / Sq8Space / PqSpace    quant::{sq8, pq}
               |
          distance::Kernels (NEON / AVX2 / scalar, chosen at run time)
```

Every layer is usable on its own: `HnswIndex` is a complete in-memory index,
`Collection` is a durable store without a server, and the server is a thin
router over `Database`.

## Distance kernels

`distance::Kernels` is a table of plain functions. At start-up `kernels()`
picks the best table the CPU supports: `neon-dotprod` (FEAT_DotProd, i.e.
`SDOT`/`UDOT`), `neon` (widening multiplies, any AArch64), `avx2` on x86-64,
`scalar` everywhere. `QUIVER_KERNELS=scalar` forces a table, which is how the
benchmarks compare kernels inside one binary. The tables are private and only
reachable through `available()`, which checks CPU features first, so the safe
wrappers around `#[target_feature]` functions can never run on a CPU without
the feature.

**f32.** Four independent FMA accumulators of four lanes, unaligned loads
(`vld1q_f32` has no alignment requirement), a 4-lane loop for the remainder
and a scalar tail. Four chains matter: SimSIMD 6.5's NEON `l2sq_f32` uses one
accumulator and is 1.8-2.9x slower on the same harness. Eight chains were
tried and measured no faster, so the simpler version stayed.

**int8.** `dot_i8` uses `SDOT` (16 multiply-accumulates per instruction);
`l2sq_i8` computes `|a - b|` with `SABD` and squares it with `UDOT`. Each
kernel works in blocks of 2^16 elements and widens its 32-bit lane
accumulators to 64 bits between blocks; the block size is chosen so that no
lane can overflow for any input (worst case 4 * 255^2 per instruction for L2).
A test runs 200,003 elements of `-128` / `127` through every table.

**Testing.** f32 kernels are compared with an f64 reference under the bound
`|error| <= 2 (n + 4) u * sum |terms|` (u = 2^-24), on 3,000 random cases per
run whose lengths are biased towards 0, tails of 1-15, multiples of 16 and one
either side, at 0-3 element offsets (unaligned addresses). Integer kernels
must equal the scalar code exactly. Fixed seeds; a failure prints the seed.

**Measured, not assumed** (`bench/results/kernels*.csv`):

* NEON f32 is about 4x the scalar loop at 128 dimensions and 7x at 768.
* The "scalar" int8 dot product is not scalar on Apple targets: LLVM
  auto-vectorises the integer loop (with `SDOT`, since the target enables
  dotprod). The hand-written kernel is 15% slower than that at 128-d and
  10-30% faster at 768-1536-d; for int8 L2, where LLVM does not find the
  `SABD`/`UDOT` idiom, it is 3-4.6x faster.
* NumKong 7 accumulates f32 inputs in f64 (two lanes per instruction, by
  design, for accuracy), so its f32 kernels are slower than quiver's and
  SimSIMD's here; they compute a more precise result. Both libraries were
  checked to have detected NEON before timing.

**SME.** The M5 has SME/SME2. A prior investigation on this machine found SME
a poor fit for single-pair distances: entering and leaving streaming mode
costs more than a 128-dimension dot product, and HNSW computes one pair at a
time on a data-dependent path. SME pays off for batched work (many queries
against a block of vectors), which the graph search does not do. quiver does
not implement SME kernels.

## HNSW

`index::graph` implements Malkov & Yashunin (2018): a geometric level per node
(`floor(-ln U / ln M)`), greedy descent through the upper layers, beam search
of width `ef` on layer 0, and the neighbour-selection heuristic (keep a
candidate only if it is closer to the base than to every neighbour already
kept, as in hnswlib). Layer 0 allows `2M` links, upper layers `M`.

The graph stores topology only. Distances come from a `Space` (`dist(q, id)`),
so the same search code traverses f32 vectors, int8 codes or PQ codes, and
the same code runs over the mutable `Graph` and over a `FrozenGraph` whose
arrays are slices of a memory-mapped file.

### Concurrent inserts and searches

* Link lists are `[count, id...]` arrays of `AtomicU32`. Writers hold a
  one-byte spin-then-yield lock per node (a `std::sync::Mutex` is 16 bytes on
  macOS, a million nodes need a million locks). Readers take no lock: they
  load the count with `Acquire`, the ids with `Relaxed`, then issue one
  `Acquire` fence; writers store ids and count with `Release`. A reader may
  see a list half-rewritten, but every slot always holds a node that was
  fully initialised before it was first linked, so the worst case is a stale
  neighbourhood.
* A row's vector is written exactly once, before the row is linked anywhere,
  so the `Release`/`Acquire` chain above also publishes the vector. This is
  the safety argument for the raw-pointer arena in `HnswIndex`; the arena is
  only exposed inside the crate.
* An insert whose level exceeds the current top level holds a global lock for
  its whole insertion (the hnswlib scheme), so two inserts never race to
  raise the entry point.

**A race the stress test found.** Once a new node is wired on layer `l`, a
concurrent insert can find it there, descend, and link to it on layer `l-1`
before the new node's own `l-1` list is written. The new node's search on
`l-1` could then reach itself and select itself as a neighbour (a self loop;
hnswlib throws in this situation). The fix excludes the node from its own
result set, and the node's own list is merged with, not overwritten by, its
selection. `concurrent_stress_keeps_invariants` (8 threads, M = 4, tight
clusters, 12 rounds) checks the invariants after every round.

### Deletes: tombstones plus rebuild

A delete sets a tombstone bit. Tombstoned nodes keep routing searches (their
links are what keeps the graph connected) but never enter the result set.
Rebuilding happens in compaction, which builds a fresh graph over the live
rows of the sealed segments. In-place repair (hnswlib's `markDelete` +
`replaceDeleted`, which reuses a deleted slot for a new vector) was rejected:
the slot's old in-links then point at a vector that has changed, which hurts
recall unless every in-neighbour is revisited. With 33% of rows tombstoned the
recall test still measures 1.0 on its data, and searches never return a
tombstoned row.

### Invariants

`check_invariants` verifies: no self loops, no duplicate links, no dangling
links (to rows not in the graph), degrees within `2M`/`M`, every link on layer
`l` points to a node that reaches layer `l`, and the entry point is on the top
layer. Frozen graphs read from disk are additionally shape-checked
(`validate_shape`) so a damaged file cannot cause an out-of-range read; a
property test corrupts frozen graphs at random and checks that every graph
that passes validation can be searched without panicking.

## Quantization

**int8, per-vector scale.** Each prepared vector `x` (mean-centred for L2) is
stored as `s * q` with `q` in `[-127, 127]` and `s = max|x_i| / 127`; the query
is quantized the same way, so a distance is one integer dot product (`SDOT`)
and two multiplications. Per-dimension scales were rejected because they turn
the distance into a weighted sum that no integer instruction computes. For L2
the norms of the *dequantized* vectors are used, so the estimate is exactly
the distance between the two reconstructions and is never negative.
Centring is done for L2 only (it changes inner products).

**PQ.** `m` sub-quantizers of 256 centroids each, trained with k-means
(Lloyd, 20 iterations, empty clusters re-seeded by splitting the largest, on
at most 25,600 sampled rows). Search uses ADC: one `m x 256` table per query,
then `m` lookups per row, summed in four independent accumulators.

**Re-ranking.** A quantized search walks the graph with compressed distances,
keeps `k * rerank` candidates and re-ranks them with the full-precision
vectors, which live in the memory-mapped segment and are only read for those
candidates. The graph itself is always built with full-precision distances.

## Storage

### Write path

1. Validate the request (dimension, finite values, payload is an object,
   batch size). Rejected requests never reach the WAL.
2. Reserve rows in the growing segment; if it is full, seal it first.
3. Append the operations to the WAL in one `write`, then sync. Syncs use
   group commit: one writer becomes the leader and syncs everything written
   so far, the rest wait and check whether that sync covered their LSN.
   `F_FULLFSYNC` on Apple platforms (plain `fsync` there does not flush the
   drive's cache), `fdatasync` elsewhere.
4. Insert the vectors into the growing segment's HNSW graph (concurrently
   with other writers). The rows are invisible to searches.
5. Under the id-map lock, commit: point the id at its new row, tombstone the
   old one, make the row visible. If the id already maps to an operation with
   a higher LSN (another writer got there first), the new row is tombstoned
   instead. So the in-memory state is the one a sequential replay of the WAL
   produces, whatever order the graph inserts finished in.
6. Answer the client.

### Files

* `wal-N.log`: CRC-32C framed records `[len][crc][kind, lsn, id, ...]`.
* `seg-N.qseg`: a 512-byte header with a table of ten sections (ids, LSNs,
  vectors, levels, layer-0 links, upper-link offsets, upper links, payload
  offsets, payload JSON, quantization), each 64-byte aligned with its own
  CRC-32C. Opening a segment maps it read-only and validates the header, every
  section checksum, every size, the graph shape and the payload offsets; the
  vectors and links are then used in place.
* `seg-N.del`: the segment's tombstone bitmap, replaced atomically.
* `MANIFEST.json`: the sealed segment ids, the current WAL generation, the
  next segment id and the LSN mark; replaced atomically (write temp, sync,
  rename, sync directory).
* `LOCK`: an exclusive `flock`, so a second process cannot open the same
  collection or database.

### Checkpoint (sealing the growing segment)

Writers are paused (the "gate"). The growing segment is written as a sealed
file (quantized if configured), dirty tombstone bitmaps are written, a fresh
WAL is created, and the manifest is replaced to point at the new segment and
the new WAL. Then the in-memory state switches over and the old WAL is
deleted. Crash points:

| crash after | state on restart |
|---|---|
| segment / tombstones written, before manifest | old manifest + old WAL; the new files are not referenced and are deleted on open; the WAL replay rebuilds the growing segment |
| new WAL created, before manifest | as above; the empty new WAL is deleted |
| manifest replaced | new segment + new (empty) WAL; the old WAL is deleted on open |

A failure *after* the manifest has been replaced cannot be undone in memory
without losing writes, so the collection stops accepting requests; a restart
recovers from the (consistent) files.

### Compaction

Compaction merges all sealed segments into one, dropping tombstoned rows and
building a fresh graph. The long part (copying live rows, building the graph,
writing the file) runs without blocking writers or searches. Then writers are
paused briefly: rows deleted since the snapshot are carried over as
tombstones of the new segment, the manifest is replaced, and id-map entries
that still point at the old rows are moved to the new ones. Only compaction
removes sealed segments, so its inputs stay valid while it runs, even if a
checkpoint adds a segment meanwhile. The server compacts automatically when
more than 20% of sealed rows are tombstoned or there are more than 8 sealed
segments.

### Recovery

Open the manifest, delete files it does not reference, open and validate the
sealed segments and their tombstones, rebuild the id map from their live rows
(an id that appears twice keeps its higher LSN), then replay the WAL: the
graph inserts in parallel, the commits in LSN order through the same commit
function as the live path. A torn final record (a crash mid-write) is
truncated away; anything before it was completely written.

### What the tests check

* `storage::wal`: cutting the log at every byte offset keeps exactly the
  complete records and leaves the log appendable; a flipped bit stops replay
  at the damaged record; 8 threads x 40 commits lose nothing and share syncs.
* `storage::segment`: round trips for every metric and quantization; 300
  random byte flips are either rejected or provably harmless (alignment
  padding is not checksummed); truncations are rejected.
* `collection::tests::model_based_random_operations`: 40 random sequences of
  60 steps (upserts with colliding ids, deletes, checkpoints, compactions,
  reopening) compared with a plain map after every step, including an exact
  brute-force search. Disabling tombstone persistence makes it fail.
* `out_of_order_commits_keep_the_highest_lsn`: commits delivered in the wrong
  order, deterministically. Removing the LSN check makes it fail.
* `tests/crash.rs`: SIGKILL by PID under concurrent upserts, deletes,
  checkpoints and compactions, four rounds on the same directory; every
  acknowledged upsert must be present with its exact vector and payload, every
  acknowledged delete gone. Disabling tombstone persistence makes it fail.

Note what SIGKILL does not test: the data of an acknowledged write is in the
OS page cache when the process dies, so this test would also pass with syncing
disabled. Power-loss durability rests on `F_FULLFSYNC` and on the ordering
argument above, not on a test.

## Filtered search

Each segment keeps a payload index: posting lists for strings and booleans,
and a `(value, row)` list per numeric field, sorted once the segment is
sealed. A filter (`eq`, `in`, ranges, `and`/`or`/`not`, dotted paths, arrays
as multi-valued fields) evaluates to a bitset over the segment's rows, ANDed
with the live rows. Its population count is the exact number of matches, so
the strategy is chosen per segment from a known selectivity, not an estimate:

* **Brute force** over the matching rows costs about `matched` distances.
* **Filtered graph traversal** (non-matching nodes are expanded but never
  enter the result set) must expand roughly `ef / s` nodes of `2M` neighbours
  to collect `ef` matches, about `ef * 2M * live / matched` distances.

The two are equal at `matched = sqrt(ef * 2M * live)`; quiver uses brute force
below `brute_force_factor` times that (default 1.0) and graph traversal above.
For 60,000 rows, `ef = 64` and `M = 16` the crossover is about 11,000 matching
rows (18% selectivity); for 1.18 million rows about 49,000 (4.2%). On both
datasets the measured crossover falls on the same side of every measured
selectivity as the model's, so the automatic choice was the faster strategy
at each point. `quiver bench filter` measures recall and throughput of
both strategies, the automatic choice and naive post-filtering on either side
of it ([benchmarks.md](benchmarks.md)).

Rejected: payload-aware extra edges (Qdrant builds additional HNSW links per
indexed payload value), and ACORN-style two-hop expansion. Both make the
graph traversal hold up at lower selectivity; with exact match counts and a
brute-force fallback, quiver does not need the graph to work at very low
selectivity, at the cost of linear scans for filters that match tens of
thousands of rows in a large segment.

## Server

`tiny_http` with a fixed pool of worker threads; routing, limits (64 MB
body, 10,000 points per upsert, 1 MB payload per point) and error mapping
(400/404/405/409/413/500, JSON bodies) are quiver's. Writes are answered after
they are durable. SIGINT/SIGTERM checkpoint every collection before exit, so
the next start has nothing to replay. Port 0 plus `--port-file` lets tests run
servers on ephemeral ports.

## Known limitations

* One growing segment per collection: concurrent writers share it, but all
  inserts into one collection go through one HNSW graph.
* Searches visit segments one after another on the calling thread.
* Compaction always merges every sealed segment (no size-tiered policy), so
  its cost grows with the collection.
* Ids are `u64`; payload values are strings, numbers, booleans and arrays of
  them; filters cannot test for a field's existence.
* Cosine collections store and return normalised vectors.
* No authentication, TLS or replication.
