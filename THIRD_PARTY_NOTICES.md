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
repository. The `diffusion_policy` import namespace was changed to `psi_policy`,
and several files contain PetalDex-specific changes.

| PetalDex file | Diffusion Policy source file |
| --- | --- |
| `psi_policy/common/replay_buffer.py` | `diffusion_policy/common/replay_buffer.py` |
| `psi_policy/common/sampler.py` | `diffusion_policy/common/sampler.py` |
| `psi_policy/common/pytorch_util.py` | `diffusion_policy/common/pytorch_util.py` |
| `psi_policy/common/normalize_util.py` | `diffusion_policy/common/normalize_util.py` |
| `psi_policy/common/checkpoint_util.py` | `diffusion_policy/common/checkpoint_util.py` |
| `psi_policy/common/json_logger.py` | `diffusion_policy/common/json_logger.py` |
| `psi_policy/common/nested_dict_util.py` | `diffusion_policy/common/nested_dict_util.py` |
| `psi_policy/workspace/base_workspace.py` | `diffusion_policy/workspace/base_workspace.py` |
| `psi_policy/model/common/normalizer.py` | `diffusion_policy/model/common/normalizer.py` |
| `psi_policy/model/common/module_attr_mixin.py` | `diffusion_policy/model/common/module_attr_mixin.py` |
| `psi_policy/model/common/dict_of_tensor_mixin.py` | `diffusion_policy/model/common/dict_of_tensor_mixin.py` |
| `psi_policy/model/common/lr_scheduler.py` | `diffusion_policy/model/common/lr_scheduler.py` |
| `psi_policy/model/diffusion/ema_model.py` | `diffusion_policy/model/diffusion/ema_model.py` |
| `psi_policy/policy/base_image_policy.py` | `diffusion_policy/policy/base_image_policy.py` |
| `psi_policy/dataset/base_dataset.py` | `diffusion_policy/dataset/base_dataset.py` |

Source review: these files were compared with the corresponding files in
[`real-stanford/diffusion_policy` on `main`](https://github.com/real-stanford/diffusion_policy/tree/main/diffusion_policy)
on 2026-11-01. The imported snapshot does not record the original upstream
commit. The upstream license was retrieved from
[the official repository](https://github.com/real-stanford/diffusion_policy/blob/main/LICENSE).

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
is hosted separately. Its dataset card declares Apache-2.0 as of 2026-11-01.
Consult that dataset repository for its applicable terms; the code license in
this repository does not change them.
