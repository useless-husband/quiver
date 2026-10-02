# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Distance kernels: L2, inner product and cosine for f32; dot product and L2
  for int8; Hamming distance. NEON (with `SDOT`/`UDOT` when FEAT_DotProd is
  present), AVX2+FMA and scalar versions, chosen at run time.
- HNSW index with the neighbour-selection heuristic, concurrent inserts and
  searches, tombstone deletes and rebuild; brute-force flat index.
- int8 scalar quantization (per-vector scale, `SDOT` distances) and product
  quantization (k-means codebooks, ADC), with full-precision re-ranking.
- Storage: CRC-32C framed write-ahead log with group commit and
  `F_FULLFSYNC`, checksummed memory-mapped segment files, tombstone files,
  atomic manifests, checkpoints, compaction and crash recovery; `flock`
  directory locks.
- Payload filters (`eq`, `in`, ranges, `and`/`or`/`not`) with per-segment
  payload indexes and a cost-based brute-force / graph strategy choice.
- `quiver serve` (HTTP/JSON API), CLI client commands, Rust library API and
  `examples/library.rs`.
- Tests: kernel property tests, HNSW invariants and concurrency stress,
  WAL torn-tail and corruption tests, segment byte-flip tests, a model-based
  random-operation test, and a SIGKILL durability test.
- Benchmarks: `quiver bench ann|kernels|filter`, hnswlib / FAISS baselines,
  a NumKong / SimSIMD kernel comparison, results and charts for
  Fashion-MNIST and GloVe-100.
