# Third-party notices

## Diffusion Policy

PetalDex adapts replay-buffer, sampling, normalization, checkpointing, logging,
workspace and model-interface utilities from
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
These portions are licensed under the MIT License. The complete upstream
license is preserved in [third_party/DIFFUSION_POLICY_LICENSE](third_party/DIFFUSION_POLICY_LICENSE)
and its copyright notice is also retained in the root [LICENSE](LICENSE).

The following file mapping records the adapted utilities. Upstream paths are
relative to the upstream repository; PetalDex paths are relative to this
repository.

| PetalDex file | Diffusion Policy source file |
| --- | --- |
| `petal/common/replay_buffer.py` | `diffusion_policy/common/replay_buffer.py` |
| `petal/common/sampler.py` | `diffusion_policy/common/sampler.py` |
| `petal/common/pytorch_util.py` | `diffusion_policy/common/pytorch_util.py` |
| `petal/common/normalize_util.py` | `diffusion_policy/common/normalize_util.py` |
| `petal/common/checkpoint_util.py` | `diffusion_policy/common/checkpoint_util.py` |
| `petal/common/json_logger.py` | `diffusion_policy/common/json_logger.py` |
| `petal/common/nested_dict_util.py` | `diffusion_policy/common/nested_dict_util.py` |
| `petal/workspace/base_workspace.py` | `diffusion_policy/workspace/base_workspace.py` |
| `petal/model/common/normalizer.py` | `diffusion_policy/model/common/normalizer.py` |
| `petal/model/common/module_attr_mixin.py` | `diffusion_policy/model/common/module_attr_mixin.py` |
| `petal/model/common/dict_of_tensor_mixin.py` | `diffusion_policy/model/common/dict_of_tensor_mixin.py` |
| `petal/model/common/lr_scheduler.py` | `diffusion_policy/model/common/lr_scheduler.py` |
| `petal/model/diffusion/ema_model.py` | `diffusion_policy/model/diffusion/ema_model.py` |
| `petal/policy/base_image_policy.py` | `diffusion_policy/policy/base_image_policy.py` |
| `petal/dataset/base_dataset.py` | `diffusion_policy/dataset/base_dataset.py` |

## Dependencies and pretrained weights

Python dependencies are installed separately under their own licenses; they are
not relicensed by the PetalDex MIT License. In particular, PetalDex uses
[PyTorch](https://github.com/pytorch/pytorch),
[timm](https://github.com/huggingface/pytorch-image-models),
[Diffusers](https://github.com/huggingface/diffusers), and
[Accelerate](https://github.com/huggingface/accelerate).
Any pretrained image-encoder weights downloaded by timm remain subject to the
terms supplied with those weights. No pretrained weights are bundled here.

## Dataset

The [PetalDex dataset](https://huggingface.co/datasets/jasonGUself/PetalDex)
is hosted separately. See its dataset card for the applicable license and terms.
The code license in this repository does not change them.
