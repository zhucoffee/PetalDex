# Data format

Download the released demonstrations from [jasonGUself/PetalDex](https://huggingface.co/datasets/jasonGUself/PetalDex). The dataset repository controls dataset terms, versioning, and download availability.

## Released HDF5 episodes

`scripts/data/convert_hdf5_to_zarr.py` reads sorted `episode_*.h5` files from one directory. Each episode must contain:

| HDF5 key | Content |
| --- | --- |
| `observations/state` | Finite `(T, 54)` joint states |
| `action` | Finite `(T, 54)` joint targets |
| `observations/images/head` | `T` JPEG-encoded 224 × 224 frames |
| `observations/images/right_wrist` | `T` JPEG-encoded 224 × 224 frames |

Each HDF5 file becomes one episode. Images are decoded to RGB; conversion rejects invalid shapes and frame-count mismatches. Conversion writes a temporary sibling directory, then renames it to the requested new destination after completion.

## Training Zarr store

| Zarr key | Shape | Type |
| --- | --- | --- |
| `data/action` | `(N, 54)` | float32 |
| `data/left_arm_states` | `(N, 7)` | float32 |
| `data/right_arm_states` | `(N, 7)` | float32 |
| `data/left_hand_states` | `(N, 20)` | float32 |
| `data/right_hand_states` | `(N, 20)` | float32 |
| `data/cam_head_color` | `(N, 224, 224, 3)` | uint8 RGB |
| `data/cam_right_wrist_color` | `(N, 224, 224, 3)` | uint8 RGB |
| `meta/episode_ends` | `(number of episodes,)` | int64 |

`N` is the total number of frames across episodes. Episode ends are cumulative, exclusive boundaries. The 54 joint dimensions are ordered left arm (7), right arm (7), left hand (20), right hand (20). The converter preserves the dataset's numerical joint units; it does not convert units or enforce hardware limits.

The dataset loader keeps image arrays on disk and loads low-dimensional arrays into memory. Images are converted from HWC uint8 RGB to CHW float32 in `[0, 1]` before policy normalization. The supplied policy uses one observation step and a 16-step action horizon; sequences do not cross episode boundaries.

## Custom ROS 2 recordings

Edit `configs/mcap_to_zarr.yaml` to match your camera, state, command, and timestamp topics. The converter synchronizes streams using the configured reference stream and timestamp tolerance. It reports bags it skips, so inspect the conversion summary. It processes one bag at a time; one complete bag and its aligned arrays can still require substantial RAM.

`scripts/data/downsample_zarr_fps.py` keeps every `--stride` frame within each episode. A stride specifies an integer reduction; it does not infer the original frame rate. Input and output paths must not overlap. Prefer a new output directory and keep the original recordings.
