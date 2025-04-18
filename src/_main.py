from datetime import datetime, timezone
import json
import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.data_module import DataModule
from src.env import DEBUG
from src.global_cfg import set_cfg
from src.misc.LocalLogger import LocalLogger
from src.misc.wandb_tools import update_checkpoint_path
from src.model.wrapper import Wrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    # Print the composed configuration
    # import pprint; pprint.pprint(OmegaConf.to_container(cfg_dict, resolve=True))
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    # Set torch variables
    if cfg.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)
    torch.backends.cudnn.benchmark = cfg.torch.cudnn_benchmark
    
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg.wandb.activated:
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=f"{output_dir.parent.name} ({output_dir.name})",
            id=f"{output_dir.parent.name}_{output_dir.name}",
            tags=cfg.wandb.tags,
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            entity=cfg.wandb.entity
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    checkpoint_dir = output_dir / "checkpoints"
    if cfg.mode == "train" and cfg.checkpointing.save:
        # Always checkpoint and continue from last state
        callbacks.append(
            ModelCheckpoint(
                checkpoint_dir,
                save_last=True,
                save_top_k=1,
                every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                save_on_train_epoch_end=False
            )
        )
        if cfg.checkpointing.every_n_train_steps_persistently is not None:
            # For safety checkpoint top-k w.r.t. total loss
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_dir,
                    save_last=False,
                    save_top_k=-1,
                    every_n_train_steps=cfg.checkpointing.every_n_train_steps_persistently,
                    save_on_train_epoch_end=False
                )
            )
        if cfg.checkpointing.save_top_k is not None:
            # For safety checkpoint top-k w.r.t. total loss
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_dir,
                    filename="epoch={epoch}-step={step}-loss={loss/total:.4f}",
                    monitor="loss/total",
                    save_last=False,
                    save_top_k=cfg.checkpointing.save_top_k,
                    auto_insert_metric_name=False,
                    every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                    save_on_train_epoch_end=False
                )
            )

    # Prepare the checkpoint for loading.
    checkpoint_path = checkpoint_dir / "last.ckpt"
    if os.path.exists(checkpoint_path):
        resume = True
    else:
        # Sets checkpoint_path to None if cfg.checkpointing.load is None
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb.project)
        resume = cfg.checkpointing.resume
    
    d_data = 1 if cfg.dataset.grayscale else 3
    num_classes = get_dataset(cfg=cfg.dataset, conditioning_cfg=cfg.conditioning, stage="train").num_classes

    model = None
    step = 0
    lightning_checkpoint_available = checkpoint_path is not None and checkpoint_path.suffix == ".ckpt"
    if lightning_checkpoint_available:
        step = torch.load(checkpoint_path)["global_step"]
        if (cfg.mode == "train" and not resume) or cfg.mode != "train":
            # Just load model weights but no optimizer state
            model = Wrapper.load_from_checkpoint(
                checkpoint_path, cfg=cfg, d_data=d_data, image_shape=cfg.dataset.image_shape, num_classes=num_classes, strict=False
            )
    
    if model is None:
        model = Wrapper(cfg, d_data, cfg.dataset.image_shape, num_classes)
        if checkpoint_path is not None and checkpoint_path.suffix != ".ckpt":
            if cfg.mode == "train":
                assert not resume, "Cannot resume training from state_dict only"
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=False)

    max_steps = cfg.trainer.max_steps
    if cfg.trainer.task_steps is not None:
        # Compute max steps in case of task arrays
        max_task_steps = step + cfg.trainer.task_steps
        max_steps = max_task_steps if max_steps == -1 else min(max_task_steps, cfg.trainer.max_steps)

    # step_tracker allows the current step to be shared with the data loader processes.
    data_module = DataModule(
        cfg.dataset, 
        cfg.data_loader, 
        cfg.conditioning,
        cfg.patch_size,
        model.patch_grid_size, 
        cfg.validation, 
        cfg.test, model.step_tracker
    )

    trainer = Trainer(
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        limit_val_batches=None if cfg.trainer.validate else 0,
        val_check_interval=cfg.trainer.val_check_interval if cfg.trainer.validate else None,
        check_val_every_n_epoch=None,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.checkpointing.save,
        enable_progress_bar=DEBUG or cfg.mode != "train",
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm=cfg.optimizer.gradient_clip_algorithm,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=max_steps,
        profiler=AdvancedProfiler(dirpath=output_dir, filename="profile") if cfg.trainer.profile else None,
        detect_anomaly=cfg.trainer.detect_anomaly
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path if resume else None)
    elif cfg.mode == "val":
        trainer.validate(model, datamodule=data_module)
    elif cfg.mode == "test":
        metrics = trainer.test(model, datamodule=data_module)
        metrics = {k: v for d in metrics for k, v in d.items()}     # merge list of dicts
        if metrics:
            metric_fname = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
            metrics_path = output_dir / "test" / f"{metric_fname}.json"
            metrics_path.parent.mkdir(exist_ok=True, parents=True)
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=4, sort_keys=True)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")
