# Usage notes and reproducibility

PetalDex supports demonstration conversion, policy training, and offline action prediction. Use this guide to configure your environment, manage checkpoints, and plan evaluation.

## Requirements

The dependency lock file targets Linux x86_64, Python 3.9.23, and PyTorch 2.8.0 with CUDA 12.8. Packages are downloaded from the Aliyun PyPI mirror. Using another platform or package source requires resolving an environment for that setup.

Training uses pretrained timm image-encoder weights downloaded separately on first use. The repository does not include pretrained PetalDex policy weights, robot drivers, a live observation adapter, or a hardware execution loop.

## Checkpoints

`CKPT=/path/to/model.ckpt` passed to the launcher initializes weights with a fresh optimizer by default. Full optimizer resume additionally requires `training.resume=True` and a checkpoint that includes optimizer state. Checkpoints made with the default `training.resume: False` omit optimizer state. Keep the checkpoint and normalizer from the same run together.

Serialized configurations and normalizers depend on the Python module paths used when they were created. Use artifacts generated with the `petal` package and the matching training configuration.

## Evaluation

The supplied dataset class currently supports `val_ratio: 0` only. It does not construct a held-out validation split. For reported evaluation, split demonstrations at the episode level and implement an evaluation protocol suited to your task. Training loss and offline action chunks do not measure task success. A checkpoint manifest for reproducing the paper's results is not provided.

## Runtime coverage

Fresh environment installation, end-to-end conversion and training, GPU inference, and physical robot execution have not been validated for this release.

## Recording a reproduction

Record the code commit, dataset revision and subset, preprocessing arguments, full resolved training config, seed, environment, GPU count, checkpoint, and normalizer. Distinguish a successful installation, data conversion, training run, offline prediction, and physical task evaluation in any report.
