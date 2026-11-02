import logging

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from timm.models.vision_transformer import Mlp, PatchEmbed
from einops.layers.torch import Rearrange

from petal.model.common.module_attr_mixin import ModuleAttrMixin
from petal.common.nested_dict_util import obs_keys_by_type


logger = logging.getLogger(__name__)

# Backbone feature dimensions
feature_dim_dict = {
    'vit_small_r26_s32_224': 384,
    'vit_small_r26_s32_384': 384,
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,

    'vit_tiny_patch16_224': 192,
    'deit_tiny_patch16_224': 192,
    'vit_tiny_patch16_384': 192,
    'test_vit.r160_in1k': 192,
    'mobilevit_xxs': 320,
    'mobilevit_xs': 384,
}


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            pretrained: bool = True,
            model_name: str = "vit_small_r26_s32_224",  # Backbone used by the two-camera policy
            use_augmentation: bool = True,
            crop_ratio: float = 0.95,
            image_size: int = None,  # Input image size; infer from shape_meta when omitted.
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        self.model_name = model_name
        self.use_augmentation = use_augmentation
        derived = obs_keys_by_type(shape_meta, "rgb")
        self.rgb_camera_keys = derived if derived else (
            "rgb_head", "rgb_left_hand", "rgb_right_hand"
        )

        # Infer image resolution from the RGB schema unless explicitly configured.

        if image_size is None:
            rgb_shapes = [tuple(v['shape']) for v in shape_meta['obs'].values()
                          if v.get('type') == 'rgb']
            image_size = int(rgb_shapes[0][-1]) if rgb_shapes else 224
        self.image_size = int(image_size)

        # timm adapts pretrained positional embeddings to the configured image size.
        self.vit_head = timm.create_model(
            model_name,
            pretrained=pretrained,
            global_pool="",
            num_classes=0, # No classification head is used, output feature vectors
            img_size=self.image_size,
        )

        # Apply random image augmentation during training.
        image_size = self.image_size
        crop_size = int(image_size * crop_ratio)

        self.train_transforms = T.Compose([
            # from [-1, 1] to [0, 1]
            T.Lambda(lambda x: (x + 1) / 2),
            T.RandomCrop(size=crop_size),
            T.Resize(size=image_size, antialias=True),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.4,
                saturation=0.2,
                hue=0.05
            ),
            # from [0, 1] to [-1, 1]
            T.Lambda(lambda x: x * 2 - 1),
        ])

        # Use deterministic transforms for evaluation and inference.
        self.eval_transforms = T.Compose([
            # from [-1, 1] to [0, 1]
            T.Lambda(lambda x: (x + 1) / 2),
            T.Resize(size=image_size, antialias=True),
            # from [0, 1] to [-1, 1]
            T.Lambda(lambda x: x * 2 - 1),
        ])


        if model_name in feature_dim_dict:
            feature_dim = feature_dim_dict[model_name]
        else:
            # Read the backbone feature dimension when it is not listed above.
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
                dummy_output = self.vit_head(dummy_input)
                feature_dim = dummy_output.shape[-1]
                print(f"Auto-detected feature_dim for {model_name}: {feature_dim}")


        self.shape_meta = shape_meta
        self.feature_dim = feature_dim

        # Derive the joint-state dimension from shape_meta.
        # Read joint-state keys from shape_meta.
        obs_shape_meta = shape_meta['obs']
        self.state_keys = [
            k for k, v in obs_shape_meta.items()
            if v.get('type') == 'low_dim'
        ]


        state_dim = 0
        for key in self.state_keys:
            state_dim += obs_shape_meta[key]['shape'][0]


        raw_output_dim = feature_dim * len(self.rgb_camera_keys) + state_dim

        self.output_dim = 400  # Project features to a dimension divisible by the attention head count.
        self.output_proj = nn.Linear(raw_output_dim, self.output_dim)

        print(f"Feature dim: {feature_dim}, State dim: {state_dim}, Raw output: {raw_output_dim}, Projected output: {self.output_dim}")


        logger.info(
            "Number of parameters in obs encoder: %e", sum(p.numel() for p in self.parameters())
        )

        print(f"==================================== Using {model_name} with feature_dim={feature_dim} ====================================")

    def forward_head(self, rgb_data, training=True):
        B, T = rgb_data.shape[:2]
        rgb_data = rgb_data[:, :, :3]

        # B*T,3,H,W
        rgb_data = rgb_data.reshape(B*T, *rgb_data.shape[2:])


        if self.use_augmentation and training:

            rgb_data = self.train_transforms(rgb_data)
        else:

            rgb_data = self.eval_transforms(rgb_data)

        # B*T,num_patches,feature_dim
        rgb_feature = self.vit_head(rgb_data)[:, :1]

        # The backbone returns (B*T, num_patches, feature_dim).
        # Keep the first feature token.

        # (B, T, feature_dim)
        rgb_feature = rgb_feature.reshape(B, T*rgb_feature.shape[1], rgb_feature.shape[-1])
        return rgb_feature

    def forward(self, obs_dict, training=True):
        embeddings = list()
        for k in self.rgb_camera_keys:
            embeddings.append(self.forward_head(obs_dict[k], training))  # (B,T,feature_dim)
        state = torch.cat([obs_dict[k] for k in self.state_keys], dim=2) # (B, T, D_state)
        embeddings.append(state)  # (B,T,feature_dim)

        # Concatenate camera and joint-state features.
        features = torch.cat(embeddings, dim=2)  # (B,T,n_cams*feature_dim)
        features = self.output_proj(features)     # (B, T, 400)

        return features

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        T = None
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            T = attr['horizon']
            this_obs = torch.zeros(
                (1, T) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs

        rgb_feats = [self.forward_head(example_obs_dict[k], training=False) for k in self.rgb_camera_keys]

        state = torch.cat([example_obs_dict[k] for k in self.state_keys], dim=2)

        obs_part_length = [rgb_feats[0].shape[1]]


        example_embeddings = torch.cat([*rgb_feats, state], dim=2)
        example_embeddings = self.output_proj(example_embeddings)
        assert len(example_embeddings.shape) == 3
        assert example_embeddings.shape[0] == 1


        return example_embeddings.shape, obs_part_length
