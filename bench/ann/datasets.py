#!/usr/bin/env python3
"""Download ann-benchmarks datasets and convert them to .fbin/.ibin.

    python bench/ann/datasets.py --data-dir DIR glove-100-angular fashion-mnist-784-euclidean
    python bench/ann/datasets.py --data-dir DIR synthetic --n 100000 --dim 64

Files written per dataset NAME: NAME.base.fbin, NAME.query.fbin and
NAME.gt.ibin (the 100 true neighbours from the HDF5 file). Nothing is
written inside the repository unless DIR points there (bench/data/ is
git-ignored).
"""
import argparse
import os
import struct
import sys
import urllib.request

import numpy as np

URL = "http://ann-benchmarks.com/{}.hdf5"


def write_fbin(path, a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        f.write(a.tobytes())


def write_ibin(path, a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        f.write(a.tobytes())


def read_fbin(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.float32).reshape(n, d)


def read_ibin(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.int32).reshape(n, d)


def fetch(name, data_dir):
    import h5py

    h5 = os.path.join(data_dir, name + ".hdf5")
    if not os.path.exists(h5):
        print(f"downloading {URL.format(name)}", file=sys.stderr)
        tmp = h5 + ".part"
        # The server rejects urllib's default User-Agent.
        req = urllib.request.Request(URL.format(name), headers={"User-Agent": "quiver-bench/0.1"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.rename(tmp, h5)
    with h5py.File(h5, "r") as f:
        base, query, gt = f["train"][:], f["test"][:], f["neighbors"][:]
        print(f"{name}: base {base.shape}, queries {query.shape}, gt {gt.shape}, distance {f.attrs.get('distance')}",
              file=sys.stderr)
    write_fbin(os.path.join(data_dir, name + ".base.fbin"), base)
    write_fbin(os.path.join(data_dir, name + ".query.fbin"), query)
    write_ibin(os.path.join(data_dir, name + ".gt.ibin"), gt)


def synthetic(data_dir, n, dim, nq, seed):
    """Gaussian clusters; exact ground truth from FAISS IndexFlatL2."""
    import faiss

    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(100, dim)).astype(np.float32) * 1.5
    lab = rng.integers(0, 100, size=n + nq)
    x = centres[lab] + rng.normal(size=(n + nq, dim)).astype(np.float32)
    base, query = x[:n], x[n:]
    flat = faiss.IndexFlatL2(dim)
    flat.add(base)
    _, gt = flat.search(query, 100)
    name = f"synthetic-{dim}-l2"
    write_fbin(os.path.join(data_dir, name + ".base.fbin"), base)
    write_fbin(os.path.join(data_dir, name + ".query.fbin"), query)
    write_ibin(os.path.join(data_dir, name + ".gt.ibin"), gt)
    print(f"{name}: {n} x {dim}, {nq} queries", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    for name in a.names:
        if name == "synthetic":
            synthetic(a.data_dir, a.n, a.dim, a.nq, a.seed)
        else:
            fetch(name, a.data_dir)


if __name__ == "__main__":
    main()
