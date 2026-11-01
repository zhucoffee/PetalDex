#!/usr/bin/env python
"""Copy selected episodes into a smaller Zarr dataset.

Preserve data keys and recompute cumulative episode boundaries.
Example: python scripts/data/extract_episode.py --src data/flowers.zarr
    --dst data/episode87.zarr --episodes 87"""

# Add the repository root for local package imports.
import sys as _sys
from pathlib import Path as _Path
_entry = _Path(__file__)
_project_root = _entry.resolve().parents[2]
_sys.path.insert(0, str(_project_root))

import argparse
from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--episodes", required=True,
                    help="Comma-separated episode indices, such as '87' or '87,76,56'")
    args = ap.parse_args()

    if Path(args.dst).exists():
        ap.error("Output already exists; choose a new destination.")
    eps = [int(x) for x in args.episodes.split(",")]
    src = zarr.open(args.src, "r")
    ends = np.asarray(src["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])

    if not eps or any(e < 0 or e >= len(ends) for e in eps):
        ap.error(f"Episode indices must be between 0 and {len(ends)-1}.")
    dst = zarr.open(args.dst, "w")
    dst_data = dst.create_group("data")
    dst_meta = dst.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


    segs = [(int(starts[e]), int(ends[e])) for e in eps]
    total = sum(b - a for a, b in segs)
    new_ends = np.cumsum([b - a for a, b in segs]).astype(np.int64)

    for key in src["data"].array_keys():
        sarr = src[f"data/{key}"]
        shape = (total,) + sarr.shape[1:]
        chunks = (min(sarr.chunks[0], total),) + sarr.chunks[1:]
        darr = dst_data.create_dataset(
            key, shape=shape, chunks=chunks, dtype=sarr.dtype, compressor=compressor,
        )
        off = 0
        for a, b in segs:
            darr[off:off + (b - a)] = sarr[a:b]
            off += (b - a)
        print(f"  data/{key:22s} {sarr.shape} -> {shape}")

    dst_meta.create_dataset("episode_ends", data=new_ends, dtype=np.int64)
    print(f"[done] {args.dst}  episodes={eps}  total_frames={total}  episode_ends={new_ends.tolist()}")


if __name__ == "__main__":
    main()
