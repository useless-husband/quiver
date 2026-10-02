#!/usr/bin/env bash
# Runs the recall/QPS benchmarks behind the README.
#
#   bench/ann/run.sh DATA_DIR [dataset ...]
#
# DATA_DIR must hold NAME.base.fbin / NAME.query.fbin / NAME.gt.ibin, as
# written by bench/ann/datasets.py. PY must be a Python with numpy, hnswlib
# and faiss-cpu. Results are appended to bench/results/ann-NAME.csv (the file
# is replaced). Build threads and query threads are fixed below so every
# library runs under the same limits.
set -euo pipefail
DATA=${1:?usage: bench/ann/run.sh DATA_DIR [dataset ...]}
shift
PY=${PY:-python3}
Q=${Q:-./target/release/quiver}
BUILD_THREADS=${BUILD_THREADS:-4}
THREADS=${THREADS:-1,4}
EF=${EF:-10,16,24,32,48,64,96,128,192,256,384,512}
mkdir -p bench/results
datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=(fashion-mnist-784-euclidean glove-100-angular)
for ds in "${datasets[@]}"; do
  case $ds in
    *angular) metric=cosine ;;
    *euclidean) metric=l2 ;;
    *) echo "unknown metric for $ds" >&2; exit 1 ;;
  esac
  base=$DATA/$ds.base.fbin
  dim=$(od -An -tu4 -j4 -N4 "$base" | tr -d ' ')
  pq=$((dim / 4))
  csv=bench/results/ann-$ds.csv
  rm -f "$csv"
  echo "== $ds (dim $dim, $metric) -> $csv"
  "$Q" bench ann --base "$base" --queries "$DATA/$ds.query.fbin" --truth "$DATA/$ds.gt.ibin" \
    --metric $metric --m 16 --efc 200 --ef "$EF" --threads "$THREADS" --build-threads "$BUILD_THREADS" \
    --quant "f32,int8,pq$pq" --rerank 4 --dataset "$ds" --out "$csv"
  # The same index with scalar distance kernels: hnswlib has no NEON kernels,
  # so this run separates the graph implementation from the SIMD code.
  QUIVER_KERNELS=scalar "$Q" bench ann --base "$base" --queries "$DATA/$ds.query.fbin" --truth "$DATA/$ds.gt.ibin" \
    --metric $metric --m 16 --efc 200 --ef "$EF" --threads "$THREADS" --build-threads "$BUILD_THREADS" \
    --quant f32 --dataset "$ds" --out "$csv"
  for lib in hnswlib faiss-hnsw faiss-ivfpq; do
    "$PY" bench/ann/baselines.py --data-dir "$DATA" --dataset "$ds" --metric $metric --library $lib \
      --m 16 --efc 200 --ef "$EF" --threads "$THREADS" --build-threads "$BUILD_THREADS" --pq-m $pq --out "$csv"
  done
done
