#!/usr/bin/env python3
"""hnswlib and FAISS baselines, writing the same CSV rows as `quiver bench ann`.

    python bench/ann/baselines.py --data-dir DIR --dataset glove-100-angular \
        --metric cosine --library hnswlib --threads 1,4 --build-threads 4 --out results.csv

Libraries: hnswlib, faiss-hnsw (IndexHNSWFlat), faiss-ivfpq (IndexIVFPQ, with
and without IndexRefineFlat re-ranking). Searches are issued as one batch
per (parameter, thread count), so per-query latency is not available for the
baselines (p50/p99 are left empty). Thread counts are enforced with
hnswlib's num_threads and faiss.omp_set_num_threads.
"""
import argparse
import os
import resource
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from datasets import read_fbin, read_ibin  # noqa: E402

HEADER = ("library,variant,dataset,metric,m,ef_construction,ef,k,query_threads,build_threads,"
          "recall,qps,p50_us,p99_us,build_s,index_mb,peak_rss_mb")


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r * 1024 / 1e6


def recall(found, gt, k):
    hits = 0
    for f, g in zip(found, gt):
        hits += len(set(f[:k].tolist()) & set(g[:k].tolist()))
    return hits / (len(gt) * k)


class Out:
    def __init__(self, path):
        self.f = None
        if path:
            new = not os.path.exists(path)
            self.f = open(path, "a")
            if new:
                self.f.write(HEADER + "\n")
        else:
            print(HEADER)

    def row(self, **r):
        line = ",".join(str(r[c]) for c in HEADER.split(","))
        print(line, file=sys.stderr)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()
        else:
            print(line)


def timed_batch(fn, queries, warm=500):
    fn(queries[:warm])
    t0 = time.perf_counter()
    labels = fn(queries)
    return labels, len(queries) / (time.perf_counter() - t0)


def run_hnswlib(a, base, queries, gt, out):
    import hnswlib

    space = {"l2": "l2", "ip": "ip", "cosine": "cosine"}[a.metric]
    n, d = base.shape
    p = hnswlib.Index(space=space, dim=d)
    p.init_index(max_elements=n, ef_construction=a.efc, M=a.m, random_seed=1)
    p.set_num_threads(a.build_threads)
    t0 = time.perf_counter()
    p.add_items(base, np.arange(n))
    build_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "h.bin")
        p.save_index(path)
        index_mb = os.path.getsize(path) / 1e6
    print(f"hnswlib built in {build_s:.1f}s", file=sys.stderr)
    for t in a.threads:
        p.set_num_threads(t)
        for ef in a.ef:
            p.set_ef(max(ef, a.k))
            labels, qps = timed_batch(lambda q: p.knn_query(q, k=a.k, num_threads=t)[0], queries)
            out.row(library="hnswlib", variant="hnsw-f32", dataset=a.dataset, metric=a.metric, m=a.m,
                    ef_construction=a.efc, ef=ef, k=a.k, query_threads=t, build_threads=a.build_threads,
                    recall=f"{recall(labels, gt, a.k):.4f}", qps=f"{qps:.1f}", p50_us="", p99_us="",
                    build_s=f"{build_s:.2f}", index_mb=f"{index_mb:.1f}", peak_rss_mb=f"{peak_rss_mb():.1f}")


def faiss_inputs(a, base, queries):
    import faiss

    if a.metric == "cosine":
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    mt = faiss.METRIC_L2 if a.metric == "l2" else faiss.METRIC_INNER_PRODUCT
    return np.ascontiguousarray(base, dtype=np.float32), np.ascontiguousarray(queries, dtype=np.float32), mt


def run_faiss_hnsw(a, base, queries, gt, out):
    import faiss

    base, queries, mt = faiss_inputs(a, base, queries)
    d = base.shape[1]
    idx = faiss.IndexHNSWFlat(d, a.m, mt)
    idx.hnsw.efConstruction = a.efc
    faiss.omp_set_num_threads(a.build_threads)
    t0 = time.perf_counter()
    idx.add(base)
    build_s = time.perf_counter() - t0
    index_mb = faiss.serialize_index(idx).nbytes / 1e6
    print(f"faiss-hnsw built in {build_s:.1f}s", file=sys.stderr)
    for t in a.threads:
        faiss.omp_set_num_threads(t)
        for ef in a.ef:
            idx.hnsw.efSearch = max(ef, a.k)
            labels, qps = timed_batch(lambda q: idx.search(q, a.k)[1], queries)
            out.row(library="faiss", variant="hnsw-f32", dataset=a.dataset, metric=a.metric, m=a.m,
                    ef_construction=a.efc, ef=ef, k=a.k, query_threads=t, build_threads=a.build_threads,
                    recall=f"{recall(labels, gt, a.k):.4f}", qps=f"{qps:.1f}", p50_us="", p99_us="",
                    build_s=f"{build_s:.2f}", index_mb=f"{index_mb:.1f}", peak_rss_mb=f"{peak_rss_mb():.1f}")


def run_faiss_ivfpq(a, base, queries, gt, out):
    import faiss

    base, queries, mt = faiss_inputs(a, base, queries)
    n, d = base.shape
    nlist = a.nlist or int(4 * np.sqrt(n))
    pq_m = a.pq_m or d // 4
    quant = faiss.IndexFlat(d, mt)
    ivf = faiss.IndexIVFPQ(quant, d, nlist, pq_m, 8, mt)
    faiss.omp_set_num_threads(a.build_threads)
    t0 = time.perf_counter()
    rng = np.random.default_rng(1)
    sample = base[rng.choice(n, size=min(n, max(nlist * 39, 65536)), replace=False)]
    ivf.train(sample)
    ivf.add(base)
    build_s = time.perf_counter() - t0
    index_mb = faiss.serialize_index(ivf).nbytes / 1e6
    print(f"faiss-ivfpq nlist={nlist} m={pq_m} built in {build_s:.1f}s", file=sys.stderr)
    refine = faiss.IndexRefineFlat(ivf, faiss.swig_ptr(base))
    refine.k_factor = a.refine
    for t in a.threads:
        faiss.omp_set_num_threads(t)
        for nprobe in a.nprobe:
            ivf.nprobe = nprobe
            for variant, index in ((f"ivfpq{pq_m}", ivf), (f"ivfpq{pq_m}+refine{a.refine}", refine)):
                labels, qps = timed_batch(lambda q: index.search(q, a.k)[1], queries)
                out.row(library="faiss", variant=variant, dataset=a.dataset, metric=a.metric, m=nlist,
                        ef_construction=0, ef=nprobe, k=a.k, query_threads=t, build_threads=a.build_threads,
                        recall=f"{recall(labels, gt, a.k):.4f}", qps=f"{qps:.1f}", p50_us="", p99_us="",
                        build_s=f"{build_s:.2f}", index_mb=f"{index_mb:.1f}", peak_rss_mb=f"{peak_rss_mb():.1f}")


def ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", required=True, choices=["l2", "ip", "cosine"])
    ap.add_argument("--library", required=True, choices=["hnswlib", "faiss-hnsw", "faiss-ivfpq"])
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", type=ints, default=ints("10,16,24,32,48,64,96,128,192,256,384,512"))
    ap.add_argument("--nprobe", type=ints, default=ints("1,2,4,8,16,32,64,128"))
    ap.add_argument("--nlist", type=int, default=0)
    ap.add_argument("--pq-m", type=int, default=0)
    ap.add_argument("--refine", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=ints, default=[1])
    ap.add_argument("--build-threads", type=int, default=4)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    base = read_fbin(os.path.join(a.data_dir, a.dataset + ".base.fbin"))
    queries = read_fbin(os.path.join(a.data_dir, a.dataset + ".query.fbin"))
    gt = read_ibin(os.path.join(a.data_dir, a.dataset + ".gt.ibin"))
    if a.nq:
        queries, gt = queries[: a.nq], gt[: a.nq]
    out = Out(a.out)
    {"hnswlib": run_hnswlib, "faiss-hnsw": run_faiss_hnsw, "faiss-ivfpq": run_faiss_ivfpq}[a.library](
        a, np.ascontiguousarray(base), np.ascontiguousarray(queries), gt, out)


if __name__ == "__main__":
    main()
