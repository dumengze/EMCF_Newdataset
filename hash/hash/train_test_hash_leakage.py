#!/usr/bin/env python3
"""
Train/test adapter for the upstream ImageHash project.

IMPORTANT
---------
This adapter does NOT modify the hashing implementation in imagehash/__init__.py.
For pHash, the value is computed exactly by the upstream call:

    imagehash.phash(Image.open(path))

The adapter only:
  1) recursively discovers images under train/ and test/;
  2) calls the upstream hash function;
  3) compares hashes across train/test using upstream ImageHash.__sub__;
  4) writes reports.

Why this audited version exists
-------------------------------
A previous adapter only wrote pairs whose distance was <= --threshold. With
--threshold 0, a result of zero only means "no identical pHash values"; it says
nothing about pairs at distance 1, 2, 3, ... . That can make a leakage audit
look falsely empty.

This version ALWAYS computes the complete cross-split distance distribution and
the nearest training image(s) for every test image. --threshold is used only to
mark/report candidate pairs; it does not change pHash computation or distance.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import imagehash


DEFAULT_TRAIN = "/Users/jason/Desktop/OrganelleClassifyDataset/train"
DEFAULT_TEST = "/Users/jason/Desktop/OrganelleClassifyDataset/test"
DEFAULT_OUTPUT = "/Users/jason/Desktop/OrganelleClassifyDataset/hash_leakage_results"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".svg"
}


def get_hashfunc(method):
    """Return an ORIGINAL ImageHash function; no hash formula is reimplemented."""
    if method == "ahash":
        return imagehash.average_hash
    if method == "phash":
        return imagehash.phash
    if method == "dhash":
        return imagehash.dhash
    if method == "whash-haar":
        return imagehash.whash
    if method == "whash-db4":
        def hashfunc(img):
            return imagehash.whash(img, mode="db4")
        return hashfunc
    if method == "colorhash":
        return imagehash.colorhash
    raise ValueError("Unsupported method: %s" % method)


def collect_images(root):
    """Recursively collect supported image files; first subfolder is class."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Directory does not exist: %s" % root)

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        class_name = rel.parts[0] if len(rel.parts) > 1 else ""
        rows.append({
            "class": class_name,
            "relative_path": str(rel),
            "path": str(path),
        })
    return rows


def compute_hashes(items, hashfunc, split):
    """Compute hashes through the upstream library call only."""
    output = []
    errors = []
    total = len(items)

    for index, item in enumerate(items, 1):
        path = item["path"]
        try:
            # Same computation path as upstream find_similar_images.py.
            with Image.open(path) as img:
                hash_value = hashfunc(img)
            output.append({
                "split": split,
                "class": item["class"],
                "relative_path": item["relative_path"],
                "path": path,
                "hash_obj": hash_value,
                "hash": str(hash_value),
            })
        except Exception as exc:
            errors.append({
                "split": split,
                "class": item["class"],
                "relative_path": item["relative_path"],
                "path": path,
                "error": repr(exc),
            })

        if index == total or index % 100 == 0:
            print("[%s] hashed %d/%d" % (split, index, total))

    return output, errors


def make_match_row(train, test, distance):
    return {
        "train_class": train["class"],
        "test_class": test["class"],
        "train_relative_path": train["relative_path"],
        "test_relative_path": test["relative_path"],
        "train_path": train["path"],
        "test_path": test["path"],
        "train_hash": train["hash"],
        "test_hash": test["hash"],
        "distance": distance,
        "same_class": train["class"] == test["class"],
    }


def compare_all(train_rows, test_rows, threshold, same_class_only=False, top_n=200):
    """
    Audit every allowed train-test pair with the ORIGINAL distance operator:

        distance = train_hash - test_hash

    Returns:
      candidates: all pairs with distance <= threshold
      nearest_rows: nearest train image(s) for each test image
      histogram: distance -> number of train-test pairs
      top_pairs: globally smallest-distance pairs (up to top_n)
      comparison_count: number of pairs actually compared
    """
    candidates = []
    nearest_rows = []
    histogram = Counter()
    top_buffer = []
    comparison_count = 0

    total = len(test_rows)
    for i, test in enumerate(test_rows, 1):
        best_distance = None
        best_trains = []

        for train in train_rows:
            if same_class_only and train["class"] != test["class"]:
                continue

            # Upstream ImageHash.__sub__; no custom distance formula.
            distance = train["hash_obj"] - test["hash_obj"]
            comparison_count += 1
            histogram[distance] += 1

            if distance <= threshold:
                candidates.append(make_match_row(train, test, distance))

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_trains = [train]
            elif distance == best_distance:
                best_trains.append(train)

            # Keep a bounded global list of the smallest distances.
            # This affects reporting only, never hash/distance computation.
            top_buffer.append((distance, train, test))
            if len(top_buffer) > max(top_n * 4, 1000):
                top_buffer.sort(key=lambda x: (x[0], x[2]["path"], x[1]["path"]))
                del top_buffer[max(top_n * 2, 500):]

        if best_distance is not None:
            for train in best_trains:
                row = make_match_row(train, test, best_distance)
                row["nearest_tie_count"] = len(best_trains)
                nearest_rows.append(row)

        if i == total or i % 50 == 0:
            print("[compare] processed test image %d/%d" % (i, total))

    candidates.sort(key=lambda r: (float(r["distance"]), r["test_path"], r["train_path"]))
    nearest_rows.sort(key=lambda r: (float(r["distance"]), r["test_path"], r["train_path"]))
    top_buffer.sort(key=lambda x: (x[0], x[2]["path"], x[1]["path"]))
    top_pairs = [make_match_row(train, test, distance) for distance, train, test in top_buffer[:top_n]]

    return candidates, nearest_rows, histogram, top_pairs, comparison_count


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(output_dir, method, threshold, train_rows, test_rows,
                 candidates, nearest_rows, histogram, top_pairs,
                 comparison_count, errors, same_class_only):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hash_fields = ["split", "class", "relative_path", "path", "hash"]
    serializable_hashes = [{k: row[k] for k in hash_fields} for row in train_rows + test_rows]
    write_csv(output_dir / "image_hashes.csv", serializable_hashes, hash_fields)

    match_fields = [
        "train_class", "test_class", "train_relative_path", "test_relative_path",
        "train_path", "test_path", "train_hash", "test_hash", "distance", "same_class"
    ]
    write_csv(output_dir / "train_test_hash_matches.csv", candidates, match_fields)

    nearest_fields = match_fields + ["nearest_tie_count"]
    write_csv(output_dir / "nearest_train_matches.csv", nearest_rows, nearest_fields)

    write_csv(
        output_dir / "top_nearest_pairs.csv",
        top_pairs,
        match_fields,
    )

    hist_rows = [{"distance": d, "pair_count": histogram[d]} for d in sorted(histogram)]
    write_csv(output_dir / "distance_histogram.csv", hist_rows, ["distance", "pair_count"])

    error_fields = ["split", "class", "relative_path", "path", "error"]
    write_csv(output_dir / "processing_errors.csv", errors, error_fields)

    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for row in train_rows:
        train_counts[row["class"]] += 1
    for row in test_rows:
        test_counts[row["class"]] += 1

    nearest_min = min((r["distance"] for r in nearest_rows), default=None)
    nearest_by_test = {}
    for r in nearest_rows:
        key = r["test_path"]
        if key not in nearest_by_test or r["distance"] < nearest_by_test[key]:
            nearest_by_test[key] = r["distance"]
    nearest_values = list(nearest_by_test.values())

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("ImageHash train/test leakage audit\n")
        f.write("==================================\n")
        f.write("Hash implementation: upstream imagehash package (unmodified)\n")
        f.write("Method: %s\n" % method)
        f.write("Candidate threshold: %s\n" % threshold)
        f.write("Comparison: %s\n" % ("same class only" if same_class_only else "all train vs all test classes"))
        f.write("Train images successfully hashed: %d\n" % len(train_rows))
        f.write("Test images successfully hashed: %d\n" % len(test_rows))
        f.write("Processing errors: %d\n" % len(errors))
        f.write("Pairwise comparisons performed: %d\n" % comparison_count)
        f.write("Candidate pairs (distance <= %s): %d\n" % (threshold, len(candidates)))
        if nearest_min is not None:
            f.write("Smallest train-test distance observed: %s\n" % nearest_min)
        if nearest_values:
            sv = sorted(nearest_values)
            f.write("Median nearest-train distance across test images: %s\n" % sv[len(sv)//2])
        f.write("\nTrain class counts:\n")
        for cls in sorted(train_counts):
            f.write("  %s: %d\n" % (cls, train_counts[cls]))
        f.write("\nTest class counts:\n")
        for cls in sorted(test_counts):
            f.write("  %s: %d\n" % (cls, test_counts[cls]))
        f.write("\nInterpretation note:\n")
        f.write("  pHash itself is unchanged. Distance is upstream ImageHash.__sub__.\n")
        f.write("  --threshold only selects candidate rows; all pair distances are audited.\n")
        if threshold == 0:
            f.write("  Zero candidates means only that no train/test pair has identical hash bits.\n")
            f.write("  Inspect nearest_train_matches.csv and distance_histogram.csv for near matches.\n")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit train/test overlap using the unmodified upstream ImageHash implementation."
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="Training directory")
    parser.add_argument("--test", default=DEFAULT_TEST, help="Test directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument(
        "--method", default="phash",
        choices=["ahash", "phash", "dhash", "whash-haar", "whash-db4", "colorhash"],
        help="Upstream ImageHash method (default: phash)",
    )
    parser.add_argument(
        "--threshold", type=int, default=0,
        help="Reporting threshold only. Hashes/distances are always computed unchanged (default: 0).",
    )
    parser.add_argument(
        "--same-class-only", action="store_true",
        help="Only compare train/test images with the same class folder name",
    )
    parser.add_argument(
        "--top-n", type=int, default=200,
        help="Number of globally nearest train/test pairs to save (default: 200)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threshold < 0:
        print("ERROR: --threshold must be >= 0", file=sys.stderr)
        return 2
    if args.top_n < 1:
        print("ERROR: --top-n must be >= 1", file=sys.stderr)
        return 2

    hashfunc = get_hashfunc(args.method)

    print("Using upstream ImageHash method: %s" % args.method)
    print("Candidate threshold: %d" % args.threshold)
    print("Train: %s" % args.train)
    print("Test: %s" % args.test)
    print("Output: %s" % args.output)
    print("Hashing algorithm is NOT modified by this adapter.\n")

    train_items = collect_images(args.train)
    test_items = collect_images(args.test)
    print("Found train images: %d" % len(train_items))
    print("Found test images:  %d" % len(test_items))

    train_rows, train_errors = compute_hashes(train_items, hashfunc, "train")
    test_rows, test_errors = compute_hashes(test_items, hashfunc, "test")

    candidates, nearest_rows, histogram, top_pairs, comparison_count = compare_all(
        train_rows, test_rows, args.threshold, args.same_class_only, args.top_n
    )

    output_dir = save_outputs(
        args.output, args.method, args.threshold,
        train_rows, test_rows, candidates, nearest_rows, histogram, top_pairs,
        comparison_count, train_errors + test_errors, args.same_class_only,
    )

    print("\nDone.")
    print("Pairwise comparisons: %d" % comparison_count)
    print("Candidate pairs (distance <= %d): %d" % (args.threshold, len(candidates)))
    if nearest_rows:
        print("Smallest train-test distance: %s" % min(r["distance"] for r in nearest_rows))
    print("Results: %s" % output_dir)
    print("  - summary.txt")
    print("  - image_hashes.csv")
    print("  - train_test_hash_matches.csv")
    print("  - nearest_train_matches.csv")
    print("  - top_nearest_pairs.csv")
    print("  - distance_histogram.csv")
    print("  - processing_errors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
