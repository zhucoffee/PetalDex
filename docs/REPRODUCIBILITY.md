# Release scope and reproducibility

This repository packages the available PetalDex source for community use. Its initial source snapshot is `vinkda/PetalDex` commit `f4c550fb1b51ee93d9f229989740a6773aeb9b20` (2026-10-31).

## Included

- HDF5 and ROS 2 MCAP conversion to the policy's Zarr schema.
- The policy, dataset loader, and a bimanual head/right-wrist camera configuration.
- Single-GPU and multi-GPU training launcher.
- Offline action prediction from an observation in a converted dataset.
- The original pinned Linux x86_64 / Python 3.9.23 dependency environment.

## Current evidence and limits

The community release received static source review and targeted fixes. It has not been validated with a fresh environment installation, end-to-end conversion/training, GPU inference, or physical robot execution. The commands in the README describe intended usage, not a completed independent reproduction of paper results.

The repository does not include pretrained PetalDex weights, a paper-results checkpoint manifest, robot drivers, a live observation adapter, or a hardware execution loop. Training uses pretrained timm image-encoder weights downloaded separately on first use.

The supplied dataset class currently supports `val_ratio: 0` only. It does not construct a held-out validation split. For reported evaluation, split demonstrations at the episode level and use a separately implemented evaluation protocol; training loss and offline action chunks do not measure task success.

`CKPT=/path/to/model.ckpt` passed to the launcher initializes weights with a fresh optimizer by default. Full optimizer resume additionally requires `training.resume=True` and a checkpoint that includes optimizer state. Checkpoints made with the default `training.resume: False` omit optimizer state. Keep the checkpoint and normalizer from the same run together.

The pinned packages and lock file use the Aliyun PyPI mirror, retained from the source environment. A different package source or platform requires resolving and checking a new environment. No claim is made that the lock file supports macOS, Windows, or CPU training.

## Recording a reproduction

Record the code commit, dataset revision and subset, preprocessing arguments, full resolved training config, seed, environment, GPU count, checkpoint, and normalizer. Distinguish a successful installation, data conversion, training run, offline prediction, and physical task evaluation in any report.
