#!/usr/bin/env python
"""Downsample a zarr dataset along time by an integer stride, per episode.

Keep every `--stride` frame within each episode and recompute episode
boundaries. For uniformly sampled 120 Hz data, a stride of 4 produces 30 Hz data.

Example:
  python scripts/data/downsample_zarr_fps.py \
    --input  data/recordings/flowers.zarr \
    --output data/recordings/flowers_30hz.zarr \
    --stride 4 --overwrite
"""

# Add the repository root for local package imports.
import sys as _sys
from pathlib import Path as _Path
_entry = _Path(__file__)
_project_root = _entry.resolve().parents[2]
_sys.path.insert(0, str(_project_root))

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="source zarr")
    p.add_argument("--output", required=True, help="target zarr")
    p.add_argument("--stride", type=int, required=True, help="keep every Nth frame (120/30 = 4)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--block", type=int, default=2048, help="how many source rows to read per copy block")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stride < 1 or args.block < 1:
        raise ValueError("--stride and --block must be positive integers")
    source_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if (source_path == output_path or source_path in output_path.parents
            or output_path in source_path.parents):
        raise ValueError("Input and output paths must not overlap")

    src = zarr.open(str(source_path), mode="r")
    ee = src["meta/episode_ends"][:].astype(np.int64)
    total = int(ee[-1])

    # Build the kept-index list per episode, recompute episode_ends.
    starts = np.concatenate([[0], ee[:-1]])
    kept_idx = []
    new_ends = []
    cum = 0
    for s, e in zip(starts, ee):
        idx = np.arange(int(s), int(e), args.stride)
        kept_idx.append(idx)
        cum += len(idx)
        new_ends.append(cum)
    kept_idx = np.concatenate(kept_idx)
    new_ends = np.asarray(new_ends, dtype=np.int64)
    n_new = len(kept_idx)
    print(f"[plan] {len(ee)} episodes, {total} -> {n_new} frames (stride={args.stride})")

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)

    dst = zarr.open(str(output_path), mode="w")
    dst_data = dst.create_group("data")
    dst_meta = dst.create_group("meta")

    data_keys = [k for k, _ in src["data"].arrays()]
    for key in data_keys:
        a = src["data"][key]
        out = dst_data.create_dataset(
            name=key,
            shape=(n_new,) + a.shape[1:],
            chunks=a.chunks,
            dtype=a.dtype,
        )
        # Copy in source-contiguous blocks; gather the kept rows that fall in each block.
        write_pos = 0
        for b0 in range(0, total, args.block):
            b1 = min(b0 + args.block, total)
            mask = (kept_idx >= b0) & (kept_idx < b1)
            if not mask.any():
                continue
            local = kept_idx[mask] - b0
            chunk = a[b0:b1][local]
            out[write_pos:write_pos + len(local)] = chunk
            write_pos += len(local)
        assert write_pos == n_new, (key, write_pos, n_new)
        print(f"[copy] {key}: {a.shape} -> {out.shape}")

    dst_meta.create_dataset("episode_ends", data=new_ends)
    print(f"[done] wrote {args.output}  (episodes={len(new_ends)}, frames={n_new})")


if __name__ == "__main__":
    main()
