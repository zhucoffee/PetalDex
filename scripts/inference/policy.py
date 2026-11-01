#!/usr/bin/env python3
"""Load a trained policy and predict an action chunk from a Zarr observation."""
import argparse
import copy
import os
from pathlib import Path
import pickle
import sys

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(checkpoint, normalizer=None, device="cuda:0", weights="auto"):
    """Load a policy without constructing a training workspace or optimizer.

    Checkpoints and optional normalizers use pickle; load only trusted files.
    """
    with Path(checkpoint).expanduser().open("rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    cfg = copy.deepcopy(payload["cfg"])
    cfg.optimizer.start_ckpt_path = None
    OmegaConf.update(cfg, "policy.obs_encoder.pretrained", False, force_add=True)
    if weights not in {"auto", "model", "ema"}:
        raise ValueError("weights must be auto, model, or ema")
    use_ema = bool(cfg.training.use_ema) if weights == "auto" else weights == "ema"
    state_key = "ema_model" if use_ema else "model"
    if state_key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint has no {state_key} weights")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][state_key])
    if normalizer is not None:
        with Path(normalizer).expanduser().open("rb") as stream:
            policy.set_normalizer(pickle.load(stream))
    return policy.eval().to(device), cfg


def read_observation(dataset, cfg, episode, frame, device):
    """Read observation history, repeating the first episode frame for padding."""
    root = zarr.open(str(dataset), mode="r")
    ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
    if not 0 <= episode < len(ends):
        raise ValueError(f"episode must be between 0 and {len(ends) - 1}")
    start = int(ends[episode - 1]) if episode else 0
    length = int(ends[episode]) - start
    if not 0 <= frame < length:
        raise ValueError(f"frame must be between 0 and {length - 1} for episode {episode}")
    indices = np.maximum(start, np.arange(start + frame - int(cfg.n_obs_steps) + 1, start + frame + 1))
    observation = {}
    for key, spec in cfg.task.shape_meta.obs.items():
        array = root[f"data/{key}"]
        values = np.stack([array[int(index)] for index in indices])
        if spec.type == "rgb":
            if values.dtype != np.uint8 or values.ndim != 4:
                raise ValueError(f"{key} must contain uint8 HWC images")
            values = values.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        else:
            values = values.astype(np.float32)
        expected = (int(cfg.n_obs_steps), *list(spec.shape))
        if values.shape != expected:
            raise ValueError(f"{key}: expected shape {expected}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains non-finite values")
        observation[key] = torch.from_numpy(values.copy()).unsqueeze(0).to(device)
    return observation


@torch.inference_mode()
def predict_actions(policy, observation):
    """Return unnormalized (horizon, action_dim) targets in training-data joint order."""
    actions = policy.predict_action(observation)["action"][0].cpu().numpy().astype(np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Predicted actions contain non-finite values")
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Training checkpoint")
    parser.add_argument("--normalizer", help="Optional normalizer.pkl from the same training run")
    parser.add_argument("--weights", choices=("auto", "model", "ema"), default="auto",
                        help="auto uses EMA when the checkpoint training configuration enables it")
    parser.add_argument("--zarr", required=True, help="Converted demonstration dataset")
    parser.add_argument("--episode", type=int, default=0, help="Zero-based episode index")
    parser.add_argument("--frame", type=int, default=0, help="Zero-based frame index within the episode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Seed for latent sampling")
    parser.add_argument("--output", default="actions.npy", help="Output NumPy file")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    policy, cfg = load_policy(args.ckpt, args.normalizer, args.device, args.weights)
    torch.manual_seed(args.seed)
    observation = read_observation(args.zarr, cfg, args.episode, args.frame, args.device)
    actions = predict_actions(policy, observation)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, actions)
    print(f"Saved actions: {output} | shape={actions.shape}")
    print("Joint order: left arm (7), right arm (7), left hand (20), right hand (20)")
    with np.printoptions(threshold=np.inf, precision=6):
        print(actions)


if __name__ == "__main__":
    main()
