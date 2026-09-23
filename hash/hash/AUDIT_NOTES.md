# Audit notes

The upstream hashing implementation in `imagehash/__init__.py` is unchanged.

## What was wrong with the previous adapter

The previous adapter only saved pairs satisfying `distance <= threshold`.
When run with `--threshold 0`, the output could only answer whether any two
images had exactly the same pHash bits. It discarded all nonzero distances, so
`0 matches` could not show whether there were near-duplicates at distance 1, 2,
3, etc. The plotted all-zero match matrix therefore looked more conclusive than
the data supported.

## What this version changes

Only the adapter/reporting interface:

- Computes the complete train-test pHash Hamming-distance histogram.
- Saves the nearest training image(s) for every test image.
- Saves the globally nearest pairs for manual review.
- Keeps `--threshold` only as a reporting filter for candidate pairs.

The pHash calculation remains `imagehash.phash(Image.open(path))`, and distance
remains the upstream `hash1 - hash2` operation.
