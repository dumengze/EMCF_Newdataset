# Train/Test adapter for the original ImageHash code

This package keeps the upstream ImageHash calculation code unchanged.

## What was added

- `train_test_hash_leakage.py`: recursively reads `train/<class>/...` and `test/<class>/...`, calls the original ImageHash API, compares only across train and test, and writes CSV reports.
- `run_organelle_hash_check.sh`: convenience command for the requested dataset paths.

## What was NOT changed

- `imagehash/__init__.py` — unchanged.
- The original hash formulas — unchanged.
- Image preprocessing — the adapter does not resize, normalize, convert RGB, or otherwise alter an image before hashing.
- Original `find_similar_images.py` — unchanged.
- Original tests — unchanged.

The adapter computes pHash exactly through:

```python
with Image.open(path) as img:
    hash_value = imagehash.phash(img)
```

This is the same API path used by the upstream demo (`hashfunc(Image.open(img))`).

## Default comparison

The default is `--threshold 0`, which requires exact equality of ImageHash values and therefore preserves the original demo's equality-based duplicate grouping behavior.

If you intentionally want near-duplicate search, provide a positive `--threshold N`; distance is then calculated by the upstream expression `hash1 - hash2`. The adapter does not implement its own Hamming-distance formula.

## Figures

`plot_hash_results.py` is a visualization-only interface. It reads the CSV files already written by `train_test_hash_leakage.py`; it does **not** reopen images, recalculate hashes, or change the matching criterion.

It creates separate PNG and PDF figures:

- `class_counts`: train/test image counts for each class.
- `train_test_match_matrix`: number of reported train/test ImageHash matches for every class pair. With the default `--threshold 0`, this visualizes exact ImageHash equality only.
- `hash_uniqueness`: percentage of unique stored ImageHash values within each split/class. This is descriptive only and is not used to decide leakage.

Install plotting support separately (the upstream package metadata is intentionally left unchanged):

```bash
python3 -m pip install matplotlib
```

Then either run the full pipeline:

```bash
bash run_organelle_hash_check.sh
```

or plot an existing result folder without rerunning the hash calculation:

```bash
python3 plot_hash_results.py \
  --input "/Users/jason/Desktop/OrganelleClassifyDataset/hash_leakage_results"
```
