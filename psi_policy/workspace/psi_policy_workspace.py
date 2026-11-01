if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf

import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
import math
from contextlib import nullcontext

from accelerate import Accelerator, DistributedDataParallelKwargs

from psi_policy.common.pytorch_util import dict_apply
from psi_policy.workspace.base_workspace import BaseWorkspace
from psi_policy.dataset.base_dataset import BaseImageDataset
from psi_policy.policy.psi_policy import PsiPolicy
from psi_policy.common.checkpoint_util import TopKCheckpointManager
from psi_policy.common.json_logger import JsonLogger
from psi_policy.model.diffusion.ema_model import EMAModel
from psi_policy.model.common.lr_scheduler import get_scheduler
from psi_policy.common.workspace_utils import log_action_mse, create_logger, start_epoch_index_from_checkpoint_path

OmegaConf.register_new_resolver("eval", eval, replace=True)

class PsiPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'optimizer_step', 'epoch']

    def __init__(self, cfg: OmegaConf, normalizer_path: str=None):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: PsiPolicy
        model_cfg = copy.deepcopy(cfg)
        if cfg.optimizer.start_ckpt_path is not None:
            # The checkpoint supplies encoder weights; avoid a redundant download.
            OmegaConf.update(model_cfg, 'policy.obs_encoder.pretrained', False, force_add=True)
        self.model = hydra.utils.instantiate(model_cfg.policy)

        self.ema_model: PsiPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self._pending_optimizer_state = None
        if cfg.optimizer.start_ckpt_path is not None:
            print("\n"+"="*100)
            print(f"Load checkpoint from: {cfg.optimizer.start_ckpt_path}")
            print("="*100+"\n")

            payload = self.load_checkpoint(
                path=cfg.optimizer.start_ckpt_path,
                exclude_keys=('optimizer',) if cfg.training.use_ema else ('optimizer', 'ema_model'),
                include_keys=self.include_keys,
                map_location='cpu',
            )
            if cfg.training.use_ema and 'ema_model' not in payload['state_dicts']:
                self.ema_model.load_state_dict(self.model.state_dict())
            if cfg.training.resume:
                self._pending_optimizer_state = payload['state_dicts'].get('optimizer')
                if self._pending_optimizer_state is None:
                    raise ValueError(
                        'This checkpoint has no optimizer state. Use training.resume=False '
                        'for weight initialization, or resume from a full training checkpoint.'
                    )
                if 'optimizer_step' not in payload['pickles']:
                    self.optimizer_step = self.global_step // int(cfg.training.gradient_accumulate_every)
            # Advance the saved epoch before resuming to avoid duplicate epoch log entries.
            pe = start_epoch_index_from_checkpoint_path(cfg.optimizer.start_ckpt_path)
            if pe is not None:
                self.epoch = pe + 1
            else:
                self.epoch = (getattr(self, 'epoch', 0) or 0) + 1
            print(
                f"Resume from checkpoint: next training epoch = {self.epoch}, "
                f"global_step = {getattr(self, 'global_step', 0)}"
            )
            # Reset the scheduler step when initializing a fresh optimizer.
            if not cfg.training.resume:
                self.global_step = 0
                self.optimizer_step = 0
                print("training.resume=False -> reset global_step to 0 for fresh optimizer schedule.")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        accumulate_every = int(cfg.training.gradient_accumulate_every)
        if accumulate_every < 1:
            raise ValueError('training.gradient_accumulate_every must be positive')

        # Set GPU device before initializing accelerator
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        logger = create_logger()


        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            log_with='wandb',

            device_placement=True,
            kwargs_handlers=[ddp_kwargs]
        )

        device = accelerator.device

        if accelerator.is_main_process:
            logger.info('==============================================')
            logger.info(f"Using mixed precision: {accelerator.mixed_precision}")
            logger.info(f"Using device: {accelerator.device}")
            logger.info('==============================================')

        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        if not wandb_cfg.get('entity'):
            wandb_cfg.pop('entity', None)

        # Configure W&B from cfg.logging and environment variables.
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        self.optimizer = self.model.get_optimizer(
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
        )
        if self._pending_optimizer_state is not None:
            self.optimizer.load_state_dict(self._pending_optimizer_state)
            self._pending_optimizer_state = None
        # Initialize learning-rate metadata for checkpoints containing model weights.
        # The scheduler uses initial_lr when resuming from a nonzero global step.
        for pg in self.optimizer.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']

        # configure dataset
        dataset: BaseImageDataset

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        if cfg.optimizer.start_normalizer_path is not None:
            print("\n"+"="*100)
            print(f"Load normalizer from: {cfg.optimizer.start_normalizer_path}")
            print("="*100+"\n")
            normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
            if accelerator.is_local_main_process:
                normalizer = pickle.load(open(cfg.optimizer.start_normalizer_path, 'rb'))
                pickle.dump(normalizer, open(normalizer_path, 'wb'))

        elif cfg.training.resume and cfg.optimizer.start_ckpt_path is not None:
            # Full resume must keep the normalization used by the saved weights.
            normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
            if accelerator.is_local_main_process:
                pickle.dump(copy.deepcopy(self.model.normalizer).cpu(), open(normalizer_path, 'wb'))
        else:
            # compute normalizer on the main process and save to disk
            normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
            # Compute and save the normalizer on the main process.
            if accelerator.is_local_main_process:
                normalizer = dataset.get_normalizer()
                pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Prepare before counting local batches, so each rank uses the actual
        # number of optimizer updates for its learning-rate schedule.
        self.model.to(device)
        train_dataloader, val_dataloader, self.model, self.optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer
        )
        num_train_batches = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            num_train_batches = min(num_train_batches, int(cfg.training.max_train_steps))
        if num_train_batches < 1:
            raise ValueError('Training requires at least one batch')

        # Step the scheduler once per actual optimizer update, including a
        # shorter accumulation window at the end of an epoch.
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=math.ceil(num_train_batches / accumulate_every) * cfg.training.num_epochs,
            last_epoch=self.optimizer_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            if cfg.training.resume:
                ema.optimization_step = self.optimizer_step

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        log_context = JsonLogger(log_path) if accelerator.is_main_process else nullcontext()
        with log_context as json_logger:

            # the number of epochs to train
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()

                train_losses = list()
                self.optimizer.zero_grad()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch, training=True)
                        window_start = (batch_idx // accumulate_every) * accumulate_every
                        window_size = min(accumulate_every, num_train_batches - window_start)
                        loss = raw_loss / window_size
                        accelerator.backward(loss)

                        # step optimizer
                        is_last_batch = batch_idx + 1 == num_train_batches
                        if (batch_idx + 1) % accumulate_every == 0 or is_last_batch:
                            accelerator.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            if not accelerator.optimizer_step_was_skipped:
                                lr_scheduler.step()
                                self.optimizer_step += 1
                                if cfg.training.use_ema:
                                    ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            if json_logger is not None:
                                json_logger.log(step_log)
                            self.global_step += 1

                        if is_last_batch:
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec,
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = policy(batch, training=False)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                        if len(val_losses) > 0:
                            # Gather validation losses across processes.
                            val_losses = torch.stack(val_losses)
                            val_losses = accelerator.gather(val_losses)

                            # Compute the global mean on the main process.
                            if accelerator.is_main_process:
                                val_loss = torch.mean(val_losses).item()
                                step_log['val_loss'] = val_loss

                # Predict action chunks for a training batch.
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference

                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        output_path = None
                        pred_action = policy.predict_action(batch['obs'], output_path)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']

                            if cfg.training.gen_attn_map:
                                val_sample_attn_maps_dir = os.path.join(self.output_dir, 'val_sample_attn_maps')
                                os.makedirs(val_sample_attn_maps_dir, exist_ok=True)
                                output_path = os.path.join(val_sample_attn_maps_dir, f'{self.epoch}.pkl')
                            else:
                                output_path = None

                            pred_action = policy.predict_action(batch['obs'], output_path)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action

                # checkpoint
                save_epoch = (self.epoch % cfg.training.checkpoint_every) == 0
                save_epoch = save_epoch or local_epoch_idx + 1 == cfg.training.num_epochs
                if save_epoch and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # Write the selected checkpoint separately from the asynchronous latest save.
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                if json_logger is not None:
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = PsiPolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
