import pickle
import inspect
import numpy as np
from typing import Dict, Tuple
from einops import rearrange, reduce
from scipy.optimize import linear_sum_assignment
import time
import torch
import torch.nn.functional as F

from petal.policy.base_image_policy import BaseImagePolicy
from petal.model.common.normalizer import LinearNormalizer
from petal.model.diffusion.transformer_for_action import TransformerForAction
from petal.model.vision.timm_obs_encoder_wuji_multirgb_joint import TimmObsEncoder


def noise_assignment(data, noise):
    device = data.device
    data, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (data, noise))
    dist = torch.cdist(data, noise)
    _, assign = linear_sum_assignment(dist.cpu())
    return torch.from_numpy(assign).to(device)

class PetalPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            obs_encoder: TimmObsEncoder,
            # arch
            n_layer=7,
            n_head=8,
            p_drop_attn=0.1,
            use_attn_mask=True,
            # delta-action: predict a_t - q_t instead of absolute joint target
            action_relative=False,
            relative_obs_keys=None,
            # Perturb joint states before encoding observations and computing relative targets.
            q_noise=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']

        obs_shape, obs_part_length = obs_encoder.output_shape()
        n_emb = obs_shape[-1]
        obs_tokens = obs_shape[-2]

        model = TransformerForAction(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens, # Number of observation conditioning tokens.
            p_drop_attn=p_drop_attn,
            obs_part_length=obs_part_length,
            use_attn_mask=use_attn_mask
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.kwargs = kwargs

        # ----- delta-action config -----
        # When action_relative=True, the model target is the increment relative
        # to the current joint state: delta = a_t - q_t, where q_t is the
        # proprio at the (single) obs step, concatenated in ACTION layout order.
        # Add the reference joint state back when returning absolute predictions.
        self.action_relative = bool(action_relative)
        if relative_obs_keys is None:
            relative_obs_keys = [
                "left_arm_states", "right_arm_states",
                "left_hand_states", "right_hand_states",
            ]
        self.relative_obs_keys = list(relative_obs_keys)

        # ----- Joint-state noise configuration -----
        # Scale joint noise by the standard deviation cached from the first batch.
        # Sample a per-example multiplier uniformly in [0, sigma_max].
        self.q_noise = dict(q_noise) if q_noise else {"enable": False}
        self._qn_std = None
        if self.q_noise.get("enable"):
            assert self.action_relative, "q_noise requires action_relative=True"
            print(f"[PetalPolicy] q_noise ENABLED: {self.q_noise}")
        if self.action_relative:
            ref_dim = sum(
                int(shape_meta['obs'][k]['shape'][0]) for k in self.relative_obs_keys
            )
            assert ref_dim == action_dim, (
                f"[PetalPolicy] action_relative: relative_obs_keys dims sum to "
                f"{ref_dim} but action_dim={action_dim}. Keys (must be in action "
                f"layout order [L_arm, R_arm, L_hand, R_hand]): {self.relative_obs_keys}"
            )
            print(
                f"[PetalPolicy] action_relative=True; ref q = concat{self.relative_obs_keys} "
                f"({ref_dim}-d), target = a_t - q_t"
            )

    # ========= delta-action helpers ============
    def _ref_q(self, obs_dict):
        """Reference joint state q_t in ACTION layout, shape (B, action_dim).

        Uses the last obs step (n_obs_steps=1 → the only step). Reads RAW obs
        (must be called on the un-normalized obs dict), same at train and infer.
        """
        parts = [obs_dict[k][:, -1] for k in self.relative_obs_keys]  # each (B, d)
        return torch.cat(parts, dim=-1)  # (B, action_dim)

    def _maybe_inject_q_noise(self, obs, training):
        """Joint-state noise: add ε to the RAW ref-proprio keys in-place (train only).

        Because both the encoder input (normalize(obs)) and the delta target
        (_normalize_target -> _ref_q) read these same obs keys, a single addition
        makes q̃=q+ε enter both paths, so the relative target becomes Δ̃=a-q̃.
        Per-dim σ = sigma_max · std_dim · U[0,1] (std lazily cached from 1st batch).
        """
        if not training or not self.q_noise.get("enable"):
            return obs
        smax = float(self.q_noise.get("sigma_max", 0.5))
        if self._qn_std is None:  # cache per-dim proprio std from this batch
            self._qn_std = {
                k: obs[k].reshape(-1, obs[k].shape[-1]).std(0).clamp_min(1e-3)
                for k in self.relative_obs_keys
            }
        for k in self.relative_obs_keys:
            x = obs[k]                                   # (B, T, d)
            s = torch.rand(x.shape[0], 1, 1, device=x.device)   # per-sample mix U[0,1]
            eps = torch.randn_like(x) * (s * smax * self._qn_std[k].view(1, 1, -1))
            obs[k] = x + eps
        return obs

    def _normalize_target(self, action, obs_dict):
        """Absolute action (B, T, A) + raw obs -> normalized model target."""
        if self.action_relative:
            q = self._ref_q(obs_dict)                 # (B, A)
            action = action - q.unsqueeze(1)          # broadcast over horizon
        return self.normalizer['action'].normalize(action)

    def _denormalize_pred(self, npred, obs_dict):
        """Normalized model output (B, T, A) + raw obs -> (absolute_action, npred).

        Returns the un-normalized absolute action (delta + q_t if relative) and
        the normalized output unchanged (for RTC feedback, which lives in the
        model's normalized space).
        """
        pred = self.normalizer['action'].unnormalize(npred)
        if self.action_relative:
            q = self._ref_q(obs_dict)
            pred = pred + q.unsqueeze(1)
        return pred, npred

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: named observation tensors matching the task schema
        result: action and action_pred tensors
        """
        assert 'past_action' not in obs_dict # Past-action conditioning is disabled.
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        B = next(iter(nobs.values())).shape[0]

        # process input
        obs_tokens = self.obs_encoder(nobs, training=False)
        # (B, N, n_emb)

        noise = torch.randn(B, self.action_horizon, self.action_dim, device=self.device)


        pred_actions, _ = self.model(
            noise,
            cond=obs_tokens,
            training=False,
            gen_attn_map=False
        )

        # unnormalize prediction (+ add q_t if delta-action)
        assert pred_actions.shape == (B, self.action_horizon, self.action_dim)
        abs_actions, npred = self._denormalize_pred(pred_actions, obs_dict)

        result = {
            'action': abs_actions,
            'action_pred': abs_actions,
            'action_normalized': npred.detach(),
        }
        return result

    def predict_action_all(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: str = None,
        prev_traj: torch.Tensor = None,
        latency_step: int = None,
        num_candidates: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate candidate action chunks from named observation tensors.

Return normalized and unnormalized candidates and their alignment scores
(negative distances). action and action_pred contain the selected
unnormalized candidate."""
        self._validate_alignment_inputs(obs_dict, prev_traj, latency_step)
        assert 'past_action' not in obs_dict  # Past-action conditioning is disabled.
        nobs = self.normalizer.normalize(obs_dict)
        obs_tokens = self.obs_encoder(nobs, training=False).repeat_interleave(32, dim=0)
        noise = torch.randn(32, self.action_horizon, self.action_dim, device=self.device)

        pred_actions_normalized, _ = self.model(
            noise,
            cond=obs_tokens,
            training=False,
            gen_attn_map=False
        )

        prev_traj_end = prev_traj[:, 8 + latency_step:].reshape(1, -1)
        gen_traj_start = pred_actions_normalized[:, :self.action_horizon - (8 + latency_step)].reshape(32, -1)

        distances = torch.cdist(gen_traj_start, prev_traj_end).reshape(-1)
        heuristic_scores = -distances

        # Keep the top candidates ranked by trajectory alignment.
        if num_candidates is not None and int(num_candidates) > 0:
            k = min(int(num_candidates), int(pred_actions_normalized.shape[0]))
            topk_idx = torch.topk(heuristic_scores, k=k, largest=True).indices
            pred_actions_normalized = pred_actions_normalized[topk_idx]
            heuristic_scores = heuristic_scores[topk_idx]
            distances = distances[topk_idx]

        min_idx = torch.argmin(distances)

        # Return the selected action with shape (1, T, A).
        selected_action_normalized = pred_actions_normalized[min_idx:min_idx + 1]
        # delta-action: unnormalize (+ add q_t) via helper; all_actions gets q added too
        all_actions_unnormalized, _ = self._denormalize_pred(pred_actions_normalized, obs_dict)
        selected_action, _ = self._denormalize_pred(selected_action_normalized, obs_dict)

        result = {
            'action': selected_action,
            'action_pred': selected_action,
            'action_normalized': selected_action_normalized.detach(),
            'all_actions_normalized': pred_actions_normalized,
            'all_actions_unnormalized': all_actions_unnormalized,
            'heuristic_scores': heuristic_scores,
            'chosen_idx': min_idx
        }
        return result

    def bidirection_predict_action(self, obs_dict: Dict[str, torch.Tensor], output_path: str = None, prev_traj: torch.Tensor = None, latency_step: int = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: named observation tensors matching the task schema
        result: action and action_pred tensors
        """
        self._validate_alignment_inputs(obs_dict, prev_traj, latency_step)
        assert 'past_action' not in obs_dict # Past-action conditioning is disabled.
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        B = next(iter(nobs.values())).shape[0]

        # process input

        obs_tokens = self.obs_encoder(nobs, training=False).repeat_interleave(32, dim=0)
        # (B, N, n_emb)

        noise = torch.randn(32, self.action_horizon, self.action_dim, device=self.device)


        pred_actions, _ = self.model(
            noise,
            cond=obs_tokens,
            training=False,
            gen_attn_map=False
        )

        prev_traj_end = prev_traj[:,8+latency_step:].reshape(1,-1)
        gen_traj_start = pred_actions[:,:self.action_horizon-(8+latency_step)].reshape(32,-1)

        distances = torch.cdist(gen_traj_start,prev_traj_end)
        min_dist, min_idx = distances.min(dim=0)
        pred_actions = pred_actions[min_idx]

        # unnormalize prediction (+ add q_t if delta-action); keep the selected
        # normalized trajectory for RTC feedback (lives in normalized model space)
        selected_normalized = pred_actions.detach()
        abs_actions, _ = self._denormalize_pred(pred_actions, obs_dict)

        result = {
            'action': abs_actions,
            'action_pred': abs_actions,
            'action_normalized': selected_normalized,
        }
        return result

    def _validate_alignment_inputs(self, obs_dict, prev_traj, latency_step):
        """The optional trajectory-alignment helpers operate on one stream."""
        if not obs_dict or any(value.shape[0] != 1 for value in obs_dict.values()):
            raise ValueError('Trajectory-alignment prediction requires batch size 1')
        if not isinstance(latency_step, int) or not 0 <= latency_step < self.action_horizon - 8:
            raise ValueError(f'latency_step must be an integer in [0, {self.action_horizon - 9}]')
        if prev_traj is None or tuple(prev_traj.shape) != (1, self.action_horizon, self.action_dim):
            raise ValueError(
                f'prev_traj must be a normalized tensor of shape '
                f'(1, {self.action_horizon}, {self.action_dim})'
            )
        if prev_traj.device != self.device:
            raise ValueError('prev_traj and policy must be on the same device')

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            betas: Tuple[float, float],
        ) -> torch.optim.Optimizer:

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=fused_available
        )
        return optimizer

    def rs_imle_loss(self, real_samples, fake_samples, epsilon=0.03, weight=None):
        B, T, D = real_samples.shape
        n_samples = fake_samples.shape[1]

        real_flat = real_samples.reshape(B, 1, -1)
        fake_flat = fake_samples.reshape(B, n_samples, -1)

        # Scale coordinates by sqrt(weight) to compute weighted Euclidean distances.
        if weight is not None:
            sqrt_w = weight.reshape(B, 1, -1).sqrt()
            real_flat = real_flat * sqrt_w
            fake_flat = fake_flat * sqrt_w

        distances = torch.cdist(real_flat, fake_flat).squeeze(1)

        valid_samples = (distances > epsilon).float()
        min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
        valid_real_samples = (min_distances < distances.max()).float()
        if valid_real_samples.sum() > 0:
            loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
        else:
            # Keep a graph-connected zero when every candidate is rejected.
            loss = fake_samples.sum() * 0.0

        return loss

    def compute_loss(self, batch, training=True):
        # normalize input
        assert 'valid_mask' not in batch

        # Joint-state noise: perturb ref-proprio BEFORE normalize+target so q̃ hits both
        batch['obs'] = self._maybe_inject_q_noise(batch['obs'], training)

        nobs = self.normalizer.normalize(batch['obs'])


        # delta-action: target = normalize(a_t - q_t) when action_relative
        nactions = self._normalize_target(batch['action'], batch['obs'])

        B = nactions.shape[0]

        n_samples_per_condition = 20

        # process input
        obs_tokens = self.obs_encoder(nobs, training)
        # (B, N, n_emb)

        noise = torch.randn(B * n_samples_per_condition, *nactions.shape[1:], device=self.device)
        repeated_obs_cond = obs_tokens.repeat_interleave(n_samples_per_condition, dim=0)


        pred_actions, _ = self.model(
            noise,
            cond=repeated_obs_cond,
            training=training,
            gen_attn_map=False
        )

        pred_actions = pred_actions.reshape(B, n_samples_per_condition, *nactions.shape[1:])

        loss = self.rs_imle_loss(nactions, pred_actions, epsilon=0.03)

        return loss

    def forward(self, batch, training=True):
        return self.compute_loss(batch, training)
