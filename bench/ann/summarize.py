#!/usr/bin/env python3
"""Markdown tables from bench/results/ann-*.csv.

    python bench/ann/summarize.py [bench/results]

For each dataset and query-thread count: the highest QPS among the measured
points whose recall@10 reaches 0.90 / 0.95 / 0.99 ("-" if no point does),
plus build time and index size as reported by each library.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot import SERIES, classify  # noqa: E402

TARGETS = (0.80, 0.90, 0.95, 0.99)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "bench/results"
    for path in sorted(glob.glob(os.path.join(d, "ann-*.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        print(f"\n### {rows[0]['dataset']}\n")
        threads = sorted({int(r["query_threads"]) for r in rows})
        build_threads = rows[0]["build_threads"]
        head = "| index | threads | " + " | ".join(f"QPS @ recall {t:.2f}" for t in TARGETS)
        print(head + f" | max recall | build s ({build_threads} thr) | index MB |")
        print("|" + "---|" * (len(TARGETS) + 5))
        order = {label: i for i, label in enumerate(SERIES)}
        groups = {}
        for r in rows:
            c = classify(r)
            if c:
                groups.setdefault((c[0], int(r["query_threads"])), []).append(r)
        for th in threads:
            for label in sorted({k[0] for k in groups}, key=lambda x: order[x]):
                pts = groups.get((label, th))
                if not pts:
                    continue
                cells = []
                for t in TARGETS:
                    ok = [float(p["qps"]) for p in pts if float(p["recall"]) >= t]
                    cells.append(f"{max(ok):,.0f}" if ok else "-")
                best = max(float(p["recall"]) for p in pts)
                print(f"| {label} | {th} | " + " | ".join(cells)
                      + f" | {best:.4f} | {float(pts[0]['build_s']):.1f} | {float(pts[0]['index_mb']):.0f} |")


if __name__ == "__main__":
    main()
