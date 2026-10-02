#!/usr/bin/env python3
"""Recall@10 vs queries/second charts from bench/results/ann-*.csv.

    python bench/ann/plot.py [bench/results]

Writes ann-NAME-light.svg and ann-NAME-dark.svg next to each CSV (the README
picks one per GitHub theme with <picture>). Colours are the fixed categorical
order of the validated default palette; every series also has its own marker
shape, so identity never depends on colour alone. The CSV is the table view.
"""
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# label -> (palette slot, marker). Colour follows the series, never its rank,
# so every chart uses the same mapping.
SERIES = {
    "quiver HNSW f32": (0, "o"),
    "hnswlib": (1, "s"),
    "FAISS HNSWFlat": (2, "^"),
    "quiver HNSW int8 + rerank": (3, "D"),
    "quiver HNSW PQ + rerank": (4, "P"),
    "FAISS IVF-PQ": (5, "X"),
    "FAISS IVF-PQ + refine": (6, "v"),
    "quiver HNSW f32, scalar kernels": (7, "h"),
}


THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}


def classify(row):
    lib, v = row["library"], row["variant"]
    label = None
    if lib == "quiver":
        if v == "hnsw-f32-scalar":
            label = "quiver HNSW f32, scalar kernels"
        elif v.endswith("-scalar"):
            label = None
        else:
            label = {"hnsw-f32": "quiver HNSW f32", "hnsw-int8": "quiver HNSW int8 + rerank",
                     "hnsw-pq": "quiver HNSW PQ + rerank"}.get(next((p for p in ("hnsw-f32", "hnsw-int8", "hnsw-pq")
                                                                   if v.startswith(p)), ""))
    elif lib == "hnswlib":
        label = "hnswlib"
    elif lib == "faiss":
        if v.startswith("hnsw"):
            label = "FAISS HNSWFlat"
        elif v.startswith("ivfpq"):
            label = "FAISS IVF-PQ + refine" if "refine" in v else "FAISS IVF-PQ"
    if label is None:
        return None
    slot, marker = SERIES[label]
    return label, slot, marker


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot(path, rows, theme_name):
    t = THEMES[theme_name]
    ds = rows[0]["dataset"]
    threads = sorted({int(r["query_threads"]) for r in rows})
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "svg.fonttype": "none",
        "axes.edgecolor": t["grid"],
        "axes.labelcolor": t["muted"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "text.color": t["text"],
    })
    fig, axes = plt.subplots(1, len(threads), figsize=(5.2 * len(threads), 4.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    minx = 1.0
    handles = {}
    for ax, th in zip(axes[0], threads):
        ax.set_facecolor(t["surface"])
        groups = {}
        for r in rows:
            if int(r["query_threads"]) != th:
                continue
            c = classify(r)
            if c is None:
                continue
            groups.setdefault(c, []).append((float(r["recall"]), float(r["qps"])))
        for (label, slot, marker), pts in sorted(groups.items(), key=lambda kv: kv[0][1]):
            pts.sort()
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            minx = min(minx, min(xs))
            (h,) = ax.plot(xs, ys, color=t["series"][slot], lw=2, marker=marker, ms=6,
                           mec=t["surface"], mew=1.5, label=label, zorder=3)
            handles[label] = (slot, h)
        ax.set_yscale("log")
        ax.grid(True, which="major", color=t["grid"], lw=0.6)
        ax.grid(True, which="minor", axis="y", color=t["grid"], lw=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{th} query thread{'s' if th > 1 else ''}", color=t["text"], loc="left", fontsize=11)
        ax.set_xlabel("recall@10")
    axes[0][0].set_ylabel("queries per second (log)")
    lo = max(0.0, (int(minx * 20) / 20.0))
    for ax in axes[0]:
        ax.set_xlim(lo, 1.002)
    fig.suptitle(f"{ds}: recall vs throughput (M=16, efConstruction=200; up and to the right is better)",
                 x=0.01, ha="left", color=t["text"], fontsize=12)
    ordered = [h for _, (slot, h) in sorted(handles.items(), key=lambda kv: kv[1][0])]
    leg = fig.legend(handles=ordered, loc="lower center", ncol=4, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, -0.01))
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    out = path.replace(".csv", f"-{theme_name}.svg")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "bench/results"
    for path in sorted(glob.glob(os.path.join(d, "ann-*.csv"))):
        rows = load(path)
        if rows:
            for theme in THEMES:
                plot(path, rows, theme)


if __name__ == "__main__":
    main()
