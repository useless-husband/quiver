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
- `quiver bench ann` and the hnswlib / FAISS baseline scripts.
