# Usage guide

PetalDex supports demonstration conversion, policy training, and offline action prediction. Use this guide to configure your environment, manage checkpoints, and plan evaluation.

## Requirements

The dependency lock file targets Linux x86_64, Python 3.9.23, and PyTorch 2.8.0 with CUDA 12.8. Packages are downloaded from the Aliyun PyPI mirror. Using another platform or package source requires resolving an environment for that setup.

The image encoder downloads pretrained timm weights on first use. Pretrained PetalDex policy checkpoints are not included; follow the [training steps](../README.md#3-train) to train a policy on your demonstrations.

## Checkpoints

`CKPT=/path/to/model.ckpt` passed to the launcher initializes weights with a fresh optimizer by default. Full optimizer resume additionally requires `training.resume=True` and a checkpoint that includes optimizer state. Checkpoints made with the default `training.resume: False` omit optimizer state.

Keep the checkpoint, `normalizer.pkl`, and training configuration from the same run together. Load saved artifacts with a matching PetalDex version. See [action prediction](../README.md#4-predict-actions) for loading a checkpoint and selecting model or EMA weights.

## Evaluation

The supplied dataset class supports `val_ratio: 0` only. It does not construct a held-out validation split. For evaluation, split demonstrations at the episode level and implement an evaluation protocol suited to your task. Training loss and offline action chunks do not measure task success.

## Robot integration

Offline prediction produces joint targets in the dataset's ordering and units. Executing them on a robot requires an observation adapter and controller for your hardware. Robot drivers and a hardware execution loop are not included. See the [data format](DATA.md) for observation keys and joint dimensions.

## Experiment records

For repeatable experiments, record the code commit, dataset revision and subset, preprocessing arguments, resolved training configuration, random seed, environment, GPU count, checkpoint, and normalizer. Include the held-out episode IDs and task success criteria with evaluation results.
