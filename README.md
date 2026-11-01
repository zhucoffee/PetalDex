# 🌸 PetalDex

Policy training and action prediction for bimanual flower arrangement with **Tianji Arms and Wuji Hands**.

[Project Website](https://petaldex.github.io/) · [Dataset](https://huggingface.co/datasets/jasonGUself/PetalDex)

**Download data → Train a policy → Predict actions.**

## Repository layout

| Path | Purpose |
|---|---|
| `psi_policy/` | Models, datasets and training configuration |
| `configs/` | ROS recording conversion settings |
| `scripts/data/` | Data conversion and preparation |
| `scripts/training/` | Single-GPU and multi-GPU training |
| `scripts/inference/` | Model loading and action prediction |
| `train.py` | Training entry point |
| `pyproject.toml`, `uv.lock` | Python environment and dependencies |
| `accelerate_config.yaml` | Training launch settings |

## 🚀 Quick Start

### 1. Install

Use Linux x86_64 with an NVIDIA GPU and a driver supporting CUDA 12.8. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/vinkda/PetalDex.git
cd PetalDex
uv sync --frozen
source .venv/bin/activate
export PROJECT_ROOT="$PWD"
```

Python 3.9.23 and PyTorch 2.8.0 are specified in `pyproject.toml`. Run the commands below from the project root with this environment activated.

### 2. Download and prepare data

Download the human–robot co-creation subset and convert it to Zarr:

```bash
hf download jasonGUself/PetalDex --repo-type dataset \
  --include 'hdf5/robot_human_co-creation/*' --local-dir data/PetalDex
python scripts/data/convert_hdf5_to_zarr.py \
  --input data/PetalDex/hdf5/robot_human_co-creation \
  --output data/flowers.zarr
```

Replace `robot_human_co-creation` with `robot_auto` in both paths to use the other subset.

### 3. Train

```bash
bash scripts/training/run_train_ddp.sh \
  "$PROJECT_ROOT/data/flowers.zarr" arrange_flowers 64 1000 0
```

Arguments: dataset path, experiment name, batch size per GPU, epochs and GPU IDs. For two GPUs, replace `0` with `0,1`.

Checkpoints and `normalizer.pkl` are saved under `data/outputs/`. Training settings are in `psi_policy/config/psi_policy_workspace.yaml`. W&B logging defaults to offline; the image encoder downloads pretrained weights on first use.

### 4. Predict actions

Select the checkpoint and normalizer from your training run, then predict from an observation in the converted dataset:

```bash
python scripts/inference/policy.py \
  --ckpt /path/to/training_run/checkpoints/latest.ckpt \
  --normalizer /path/to/training_run/normalizer.pkl \
  --zarr data/flowers.zarr --episode 0 --frame 0 \
  --device cuda:0 --output predictions/actions.npy
```

Replace `/path/to/training_run` with your output directory under `data/outputs/`. Episode and frame indices start at zero.

The script prints and saves **16 × 54 joint targets**, ordered as **left arm (7), right arm (7), left hand (20), right hand (20)**, in the same units as the dataset.

## Use your own recordings

Configure the camera and joint topics in `configs/mcap_to_zarr.yaml`, then convert your ROS 2 bags:

```bash
export BAG_ROOT=/path/to/recordings
python scripts/data/convert_mcap_to_wuji_bimanual_zarr.py \
  --config configs/mcap_to_zarr.yaml --output data/custom.zarr
```

`BAG_ROOT` contains the `rosbag_*` recording folders. Use the resulting Zarr path in the training command above.
