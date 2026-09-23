#!/usr/bin/env python3
"""Visualization for audited ImageHash train/test reports.

No image is opened and no hash is recomputed here. This script reads CSV output
from train_test_hash_leakage.py only.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/Users/jason/Desktop/OrganelleClassifyDataset/hash_leakage_results")
    p.add_argument("--output", default=None)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def save(fig, out, stem, dpi):
    png = out / (stem + ".png")
    pdf = out / (stem + ".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def main():
    import matplotlib.pyplot as plt

    args = parse_args()
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve() if args.output else inp / "figures"
    out.mkdir(parents=True, exist_ok=True)

    hashes = read_csv(inp / "image_hashes.csv")
    candidates = read_csv(inp / "train_test_hash_matches.csv")
    nearest = read_csv(inp / "nearest_train_matches.csv")
    hist = read_csv(inp / "distance_histogram.csv")
    classes = sorted({r["class"] for r in hashes if r.get("class")})
    generated = []

    # 1. Class counts
    counts = Counter((r["split"], r["class"]) for r in hashes)
    x = list(range(len(classes)))
    w = 0.38
    tr = [counts[("train", c)] for c in classes]
    te = [counts[("test", c)] for c in classes]
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.15), 5.5))
    ax.bar([i-w/2 for i in x], tr, width=w, label="Train")
    ax.bar([i+w/2 for i in x], te, width=w, label="Test")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Number of images"); ax.set_xlabel("Class")
    ax.set_title("Train and test image counts by class"); ax.legend(); ax.grid(axis="y", alpha=.25)
    generated += save(fig, out, "class_counts", args.dpi); plt.close(fig)

    # 2. Complete pairwise Hamming-distance histogram
    dx = [int(float(r["distance"])) for r in hist]
    dy = [int(r["pair_count"]) for r in hist]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(dx, dy)
    ax.set_xlabel("pHash Hamming distance")
    ax.set_ylabel("Number of train-test pairs")
    ax.set_title("All train-test pHash distances")
    ax.grid(axis="y", alpha=.25)
    generated += save(fig, out, "all_pair_distance_histogram", args.dpi); plt.close(fig)

    # 3. One nearest distance per test image (ties collapsed by test path)
    nearest_by_test = {}
    for r in nearest:
        p = r["test_path"]
        d = int(float(r["distance"]))
        nearest_by_test[p] = min(d, nearest_by_test.get(p, d))
    nd = Counter(nearest_by_test.values())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(sorted(nd), [nd[d] for d in sorted(nd)])
    ax.set_xlabel("Nearest training-image pHash distance")
    ax.set_ylabel("Number of test images")
    ax.set_title("Nearest train image for each test image")
    ax.grid(axis="y", alpha=.25)
    generated += save(fig, out, "nearest_distance_histogram", args.dpi); plt.close(fig)

    # 4. Candidate matrix at the threshold used by the audit
    import numpy as np
    idx = {c:i for i,c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    for r in candidates:
        a, b = r.get("train_class", ""), r.get("test_class", "")
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1
    fig, ax = plt.subplots(figsize=(max(7, len(classes)*.85), max(6, len(classes)*.75)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Test class"); ax.set_ylabel("Train class")
    ax.set_title("Train-test candidate pairs at selected threshold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(int(mat[i,j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Candidate pairs")
    generated += save(fig, out, "candidate_match_matrix", args.dpi); plt.close(fig)

    print("Figures generated:")
    for p in generated:
        print("  - %s" % p)
    print("Visualization only: no image hash was recalculated or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
