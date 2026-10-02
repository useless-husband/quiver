# quiver

**A compact vector database in Rust: a hand-written HNSW index, Arm NEON distance kernels, int8 and product quantization, a write-ahead log with crash recovery, filtered search and an HTTP API, benchmarked against hnswlib and FAISS.**

quiver answers "which stored vectors are closest to this one?" for collections of `(id, vector, JSON payload)` points, fast enough to serve from one process and without losing an acknowledged write when that process is killed. It is not a new algorithm: hnswlib, FAISS, Qdrant and others do the same job at much larger scale. What it offers is a compact, well-tested implementation (about 9,500 lines of Rust including tests) in which every claim below is backed by a command you can run, and benchmarks that say where it loses.

[繁體中文說明](README.zh-TW.md) · [Design](docs/DESIGN.md) · [Benchmarks](docs/benchmarks.md) · [導讀（中文白話）](docs/導讀.zh-TW.md)

## What is in it

- **HNSW** (Malkov & Yashunin) with the neighbour-selection heuristic, configurable `M` / `efConstruction` / `ef`, concurrent inserts and searches (lock-free reads, a one-byte lock per node), tombstone deletes with rebuild-on-compaction, and a structural invariant checker. A brute-force flat index serves as the baseline and as ground truth in tests.
- **Distance kernels** for L2, inner product and cosine on f32, dot product and L2 on int8, and Hamming distance: hand-written NEON (with `SDOT`/`UDOT` when the CPU has FEAT_DotProd), AVX2+FMA on x86-64, and a scalar reference, chosen at run time. Property-tested against an f64 reference on random lengths, tails and unaligned inputs.
- **Quantization**: int8 with per-vector scaling (distances by `SDOT`) and product quantization (k-means codebooks, table lookups), both used for graph traversal with re-ranking on full-precision vectors.
- **Storage**: a CRC-checked write-ahead log with group commit and `F_FULLFSYNC` on macOS, immutable memory-mapped segment files with per-section checksums, atomic manifests, background compaction, and recovery that replays the log. A test kills the server with SIGKILL under write load and checks every acknowledged write afterwards.
- **Filtering** on payload fields (equality, `in`, ranges, `and`/`or`/`not`), with the strategy chosen per segment from the exact number of matching rows: brute force over the matches when there are few, filtered graph traversal otherwise.
- **Interfaces**: an HTTP/JSON server (`quiver serve`), a CLI, and the Rust library.

## Demo

A real session (`quiver serve` on an ephemeral port, then the CLI):

```console
$ quiver serve --data ./data --listen 127.0.0.1:0 --port-file port &
quiver 0.1.0 listening on http://127.0.0.1:58185 (data: ./data)
$ export QUIVER_URL=http://$(cat port)
$ quiver create docs --dim 4 --metric cosine
{ "created": "docs" }
$ cat points.jsonl
{"id":1,"vector":[1,0,0,0],"payload":{"lang":"en","year":2019}}
{"id":2,"vector":[0.9,0.1,0,0],"payload":{"lang":"de","year":2021}}
{"id":3,"vector":[0,1,0,0],"payload":{"lang":"en","year":2023}}
{"id":4,"vector":[0.7,0.7,0,0],"payload":{"lang":"en","year":2024,"tags":["new"]}}
$ quiver import docs points.jsonl
imported 4 points in 0.0s
$ quiver search docs --vector 1,0.2,0,0 --k 2 \
    --filter '{"and":[{"field":"lang","eq":"en"},{"field":"year","gte":2020}]}'
{
  "hits": [
    { "distance": 0.16794973611831665, "id": 4,
      "payload": { "lang": "en", "tags": ["new"], "year": 2024 } },
    { "distance": 0.8038838505744934, "id": 3,
      "payload": { "lang": "en", "year": 2023 } }
  ],
  "plan": [ { "live": 4, "matched": 2, "segment": 1, "strategy": "brute" } ],
  "took_us": 25
}
$ quiver delete docs 4
{ "deleted": 1, "lsn": 5 }
$ kill -TERM %1        # SIGTERM checkpoints before exit
quiver: shutting down, checkpointing collections
$ ls data/collections/docs
LOCK  MANIFEST.json  seg-00000001.del  seg-00000001.qseg  wal-00000002.log
```

The `plan` field shows, per segment, how many rows matched the filter and which strategy answered.

## How it works

```
 write ─► validate ─► WAL append + group-commit sync ─► HNSW insert (growing segment)
                                                        ─► commit by LSN ─► reply
 growing segment full ─► seal: segment file (+ int8/PQ codes), new WAL, atomic manifest
 tombstones > 20% or > 8 segments ─► compaction: fresh graph over live rows, short swap
 search ─► per segment: filter bitset ─► brute force or HNSW (± quantized + re-rank) ─► merge
```

- **Concurrency.** Link lists are atomic arrays: writers lock one node at a time, readers take no locks and rely on `Release`/`Acquire` ordering, so a reader can see a stale neighbourhood but never an uninitialised node. A stress test found a real race (a new node could select itself as a neighbour; hnswlib raises an exception in the same situation); it is fixed and covered.
- **Durability.** A write is logged and synced before it touches memory and is answered after it has been applied. Concurrent writers share syncs. Races on the same id are resolved by log sequence number, so memory always matches a sequential replay of the log.
- **Filtering.** Each segment evaluates a filter to an exact bitset through its payload index. Brute force over `matched` rows costs about `matched` distances; a filtered graph search costs about `ef * 2M * live / matched`; quiver picks brute force below the crossover `sqrt(ef * 2M * live)`.

The [design document](docs/DESIGN.md) covers the hard parts in detail: the memory-ordering argument, the checkpoint crash points, compaction under concurrent writes, and the alternatives that were rejected.

## Results

Measured on an Apple M5 (macOS 27, shared with other work), M = 16,
efConstruction = 200 for every HNSW, 6 build threads; full method, commands and
all numbers in [docs/benchmarks.md](docs/benchmarks.md), raw data in
`bench/results/`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/results/ann-glove-100-angular-dark.svg">
  <img alt="GloVe-100 recall@10 vs queries per second: quiver, hnswlib, FAISS HNSW and IVF-PQ, 1 and 6 query threads" src="bench/results/ann-glove-100-angular-light.svg">
</picture>

Best QPS among measured points reaching the recall level (single query thread / 6 threads):

| | GloVe-100 (1.18M, cosine) recall 0.90 | Fashion-MNIST (60k, L2) recall 0.99 | build s (GloVe / Fashion) |
|---|---|---|---|
| quiver HNSW f32 | 3,006 / 13,058 | 15,685 / 57,712 | 50.6 / 3.6 |
| quiver HNSW int8 + re-rank | 3,860 / 16,544 | 26,525 / 104,997 | 50.6 / 3.6 |
| FAISS HNSWFlat | 2,095 / 9,602 | 17,566 / 50,646 | 59.1 / 3.7 |
| hnswlib | 1,730 / 7,322 | 5,942 / 28,322 | 95.5 / 7.5 |
| quiver HNSW f32, scalar kernels | 2,276 / 10,105 | 5,959 / 28,810 | 71.3 / 7.7 |
| FAISS IVF-PQ + refine | (max recall 0.895) | 2,760 / 10,428 | 11.7 / 7.0 |

What these numbers do and do not show:

- **hnswlib has no NEON kernels** (its SIMD code is x86-only; the arm64 build runs a plain loop). Run with its own scalar kernels, quiver matches hnswlib within 3% on Fashion-MNIST, so quiver's lead over hnswlib on this machine comes from its distance kernels, not from a better graph.
- **Where quiver loses:** FAISS HNSWFlat is 12-15% faster single-threaded on Fashion-MNIST; quiver's PQ mode reaches only recall 0.79 on GloVe (FAISS IVF-PQ + refine: 0.895) and keeps the whole HNSW graph in memory (222 MB vs FAISS IVF-PQ's 41 MB); FAISS IVF-PQ builds 4x faster than any HNSW.
- **int8 + re-ranking** is quiver's best mode here: faster than f32 at equal recall, with 57 MB resident instead of 198 MB on Fashion-MNIST (the f32 vectors stay in the memory-mapped file for re-ranking).
- On GloVe all HNSW implementations stop near recall 0.925 at M = 16; higher recall needs a larger M.

**Kernels** (ns per call, L1-resident inputs, same harness for all; NumKong 7 accumulates f32 in f64 by design, SimSIMD 6.5 uses one accumulator):

| | dim | scalar | quiver NEON | NumKong 7 | SimSIMD 6 |
|---|---|---|---|---|---|
| L2 f32 | 768 | 298.5 | 40.3 | 262.4 | 115.2 |
| dot f32 | 768 | 303.1 | 40.7 | 145.0 | 101.6 |
| L2 int8 | 768 | 45.2 | 9.9 | 15.1 | 15.2 |

**Filtered search** picks brute force or graph traversal per segment from the exact match count. On both datasets it chose the faster strategy at every measured selectivity (50% down to 0.1%), with recall 1.0 below the crossover; naive post-filtering falls to recall 0.40-0.42 at 0.1% selectivity.

**Durability:** the SIGKILL test (4 kill rounds under concurrent writes, checkpoints and compactions) verifies about 8,000 acknowledged points and 300 acknowledged deletes per run; six seeds passed.

**SME:** the M5 has SME2, but a prior investigation on this machine found streaming-mode entry costs more than a single-pair distance, and HNSW computes one pair at a time, so quiver does not use it.

## API

| method | path | body |
|---|---|---|
| `PUT` | `/collections/{name}` | `{"dim": 768, "metric": "cosine", "m": 16, "ef_construction": 200, "ef_search": 64, "quantization": {"type": "int8"}, "rerank": 4, "segment_rows": 100000}` (all but `dim`/`metric` optional) |
| `GET` / `DELETE` | `/collections/{name}` | stats / drop |
| `GET` | `/collections` | stats of all collections |
| `PUT` | `/collections/{name}/points` | `{"points": [{"id": 1, "vector": [...], "payload": {...}}]}` (insert or replace, up to 10,000) |
| `POST` | `/collections/{name}/points/delete` | `{"ids": [1, 2]}` |
| `POST` | `/collections/{name}/points/get` | `{"ids": [1, 2], "with_vector": true}` |
| `GET` | `/collections/{name}/points/{id}` | |
| `POST` | `/collections/{name}/search` | `{"vector": [...], "k": 10, "ef": 64, "filter": {...}, "with_payload": true, "strategy": "auto"}` |
| `POST` | `/collections/{name}/checkpoint` / `/compact` | |

Distances are "smaller is closer": squared L2, `-dot` for inner product, `1 - cos` for cosine. Filters: `{"field": "color", "eq": "red"}`, `{"field": "price", "gte": 10, "lt": 20}`, `{"field": "tags", "in": ["a", "b"]}`, combined with `{"and": [...]}`, `{"or": [...]}`, `{"not": {...}}`; fields are dotted paths and arrays match if any element does.

Library use ([examples/library.rs](examples/library.rs), `cargo run --release --example library`):

```rust
let c = Collection::create(&dir, "items", CollectionConfig::new(3, Metric::L2), Options::default())?;
c.upsert(&[Point { id: 1, vector: vec![0.1, 0.2, 0.3], payload: Some(json!({"color": "red"})) }])?;
let red = Filter::parse(&json!({"field": "color", "eq": "red"}))?;
let mut q = Query::new(&[0.1, 0.2, 0.25], 2);
q.filter = Some(&red);
let hits = c.search(&q)?.hits;
```

`quiver::HnswIndex` is also usable on its own as an in-memory index.

## Build and test

Requires Rust 1.98 or newer (the NEON dot-product intrinsics were stabilised in 1.98).

```sh
make build        # cargo build --release
make test         # unit, property and integration tests, including the SIGKILL test
make lint         # rustfmt --check and clippy -D warnings
QUIVER_KERNELS=scalar cargo test --release --lib   # the same tests on the scalar kernels
```

CI runs the tests on macOS (Apple Silicon), Linux AArch64 and Linux x86-64 (the AVX2 path). Randomised tests use fixed seeds and print the seed on failure; `QUIVER_TEST_SEED` replays one, `QUIVER_TEST_CASES` scales the number of cases, `QUIVER_CRASH_ROUNDS` the kill rounds.

## Limitations

- One growing segment per collection, and searches visit segments one after another on the calling thread.
- Compaction always merges every sealed segment; there is no size-tiered policy, so its cost grows with the collection.
- Very selective filters on very large segments fall back to a linear scan of the matching rows (exact, but linear); there are no payload-aware graph edges as in Qdrant.
- The SIGKILL test covers process crashes. Power-loss durability relies on `F_FULLFSYNC`/`fdatasync` and the write ordering described in the design document; it is not tested by pulling the plug.
- No authentication, TLS, replication or sharding. Ids are `u64`.
- No SME kernels (see the benchmarks for why).

## Related work

- [hnswlib](https://github.com/nmslib/hnswlib): the reference HNSW implementation by the algorithm's authors; quiver's graph code follows its structure (heuristic, per-node locks, entry-point lock). Its SIMD kernels are x86-only.
- [FAISS](https://github.com/facebookresearch/faiss): HNSW, IVF, PQ and much more, with highly optimised kernels; the main baseline here.
- [USearch](https://github.com/unum-cloud/usearch): a compact single-file HNSW engine using SimSIMD/NumKong kernels.
- [Qdrant](https://github.com/qdrant/qdrant), [Milvus](https://github.com/milvus-io/milvus), [LanceDB](https://github.com/lancedb/lancedb), [pgvector](https://github.com/pgvector/pgvector): production vector databases with segments, WALs, payload filtering, replication and much more than quiver has.
- [NumKong](https://github.com/ashvardanian/NumKong) / [SimSIMD](https://github.com/ashvardanian/SimSIMD): portable SIMD distance kernels; measured against quiver's below.

quiver claims no novelty beyond being a compact, well-tested implementation with transparent benchmarks.

## License

MIT. Copyright (c) 2026 useless-husband.
