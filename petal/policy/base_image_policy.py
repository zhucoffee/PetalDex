from typing import Dict
import torch
import torch.nn as nn
from petal.model.common.module_attr_mixin import ModuleAttrMixin
from petal.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Accept named observation tensors with shape (B, To, ...).
        Return a dictionary containing predicted action tensors (B, Ta, Da).
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
