from dataclasses import dataclass, field
import inspect
from math import log
from typing import Any, Literal, get_args, Sequence, TYPE_CHECKING

from jaxtyping import Bool, Float, Int64
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from torch import Tensor, optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.nn.functional import (
    avg_pool2d, 
    interpolate, 
    mse_loss, 
    sigmoid
)

if TYPE_CHECKING:
    # NOTE to avoid circular import wrapper -> datamodule -> evaluation -> wrapper
    from src.dataset.data_module import DataModule

from ..misc.LocalLogger import LocalLogger
from ..misc.nn_module_tools import freeze, requires_grad
from ..misc.step_tracker import StepTracker

from .denoiser import Denoiser, DenoiserCfg, get_denoiser
from .diagonal_gaussian import DiagonalGaussian
from .flow import Flow, FlowCfg, get_flow
from .time_sampler import TimeSampler, TimeSamplerCfg, get_time_sampler
from src.env import DEBUG
from src.type_extensions import (
    BatchedExample,
    BatchedEvaluationExample,
    ConditioningCfg,
    FullPrecision,
    Parameterization
)


@dataclass
class LRSchedulerCfg:
    name: str
    interval: Literal["epoch", "step"] = "step"
    frequency: int = 1
    monitor: str | None = None
    kwargs: dict[str, Any] | None = None


@dataclass
class OptimizerCfg:
    name: str
    lr: float
    scale_lr: bool = False
    kwargs: dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
    gradient_clip_val: float | int | None = None 
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"


@dataclass
class TrainCfg:
    step_offset: int
    log_flow_loss_per_time_split: bool = True
    log_sigma_loss_per_time_split: bool = True
    num_time_logging_splits: int = 10
    num_time_samples: int = 1
    ema_decay_rate: float = 0.9999


@dataclass
class PerLossCfg:
    weight: float | int = 1
    apply_after_step: int = 0


@dataclass
class VLBLossCfg(PerLossCfg):
    time_step_size: float = 1.e-3
    

@dataclass
class LossCfg:
    vlb: VLBLossCfg = field(default_factory=VLBLossCfg)
    sigma: PerLossCfg = field(default_factory=PerLossCfg)


@dataclass
class ModelCfg:
    denoiser: DenoiserCfg
    flow: FlowCfg
    time_sampler: TimeSamplerCfg
    learn_sigma: bool
    time_interval: list[float] = field(default_factory=lambda: [0, 1])
    time_eps: float = 1.e-5
    denoiser_parameterization: Parameterization = "ut"
    parameterization: Parameterization = "ut"
    ema: bool = False


@dataclass
class WrapperCfg:
    conditioning: ConditioningCfg
    model: ModelCfg
    loss: LossCfg
    optimizer: OptimizerCfg
    # int: varying noise levels / None: image level baseline
    patch_size: int | None
    train: TrainCfg
    # test: TestCfg


class Wrapper(LightningModule):
    cfg: WrapperCfg
    logger: LocalLogger | WandbLogger | None
    denoiser: Denoiser
    flow: Flow
    time_sampler : TimeSampler
    step_tracker: StepTracker

    def __init__(
        self,
        cfg: WrapperCfg,
        d_data: int,
        image_shape: Sequence[int],
        num_classes: int | None = None
    ) -> None:
        if cfg.conditioning.label:
            assert num_classes is not None
        super().__init__()
        self.cfg = cfg
        self.step_tracker = StepTracker(cfg.train.step_offset)
            
        self.d_data = d_data
        d_in = self.d_data
        if self.cfg.patch_size is None and self.cfg.conditioning.mask:
            d_in += self.d_data + 1
        d_out = self.d_data
        if cfg.model.flow.variance == "learned_range":
            d_out += self.d_data
        if self.cfg.model.learn_sigma:
            d_out += 1
        self.denoiser = get_denoiser(
            cfg.model.denoiser, 
            d_in, 
            d_out,
            image_shape,
            num_classes=num_classes if cfg.conditioning.label else None
        )
        if self.cfg.model.ema:
            self.ema_denoiser = AveragedModel(
                self.denoiser, 
                multi_avg_fn=get_ema_multi_avg_fn(self.cfg.train.ema_decay_rate)
            )
            requires_grad(self.ema_denoiser, False)
        else:
            self.ema_denoiser = None

        self.flow = get_flow(cfg.model.flow, cfg.model.parameterization)
        self.patch_grid_size = (1, 1) if self.cfg.patch_size is None else tuple(s // self.cfg.patch_size for s in image_shape)
        self.time_sampler = get_time_sampler(cfg.model.time_sampler, resolution=self.patch_grid_size)
        
        if self.cfg.patch_size is not None:
            # auxiliary kernels for upsampling
            self.register_buffer("float_kernel", torch.ones(2 * (self.cfg.patch_size,)), persistent=False)
            self.register_buffer("bool_kernel", torch.ones(2 * (self.cfg.patch_size,), dtype=torch.bool), persistent=False)

    def log_time_split_loss(
        self,
        key: str,
        loss: Float[Tensor, "*batch channel height width"], 
        t: Float[Tensor, "*batch 1 height width"]
    ) -> None:
        if self.cfg.train.num_time_logging_splits > 1:
            # Log mean loss for every equal-size time interval
            interval_size = 1 / self.cfg.train.num_time_logging_splits
            loss_log = loss.detach().mean(dim=-3).flatten()
            t_split_idx = torch.floor_divide(t.flatten(), interval_size).long()
            t_split_loss = torch.full(
                (self.cfg.train.num_time_logging_splits,), 
                fill_value=torch.nan, dtype=loss_log.dtype, device=loss_log.device)
            t_split_loss.scatter_reduce_(0, t_split_idx, loss_log, reduce="mean", include_self=False)
            for i in range(self.cfg.train.num_time_logging_splits):
                if not torch.isnan(t_split_loss[i]):
                    start = i * interval_size
                    self.log(f"loss/{key}_{start:.1f}-{start+interval_size:.1f}", t_split_loss[i])

    def forward(
        self,
        z_t: Float[Tensor, "batch time d_data height width"], 
        t: Float[Tensor, "batch time 1 height width"],
        label: Int64[Tensor, "batch"] | None = None,
        c_cat: Float[Tensor, "batch time d_c height width"] | None = None,
        sample: bool = False,
        use_ema: bool = True    # NOTE ignored if sample == False or model has no EMA
    ) -> tuple[
        Float[Tensor, "batch time d_data height width"],
        Float[Tensor, "batch time d_data height width"] | None,
        Float[Tensor, "batch time #d_data #height #width"] | None,
        Float[Tensor, "batch time #d_data #height #width"] | None
    ]:
        in_t = z_t if c_cat is None else torch.cat((z_t, c_cat), dim=-3)
        if sample:
            # NOTE Do not use compile for sampling because of varying batch sizes
            pred = (self.ema_denoiser if self.cfg.model.ema and use_ema else self.denoiser).forward(
                in_t, t, label
            )
        else:
            pred = self.denoiser.forward_compiled(
                in_t, t, label
            )
        mean_theta = pred[..., :self.d_data, :, :]
        # Convert prediction from denoiser parameterization to model parameterization
        mean_theta = getattr(self.flow, f"get_{self.cfg.model.parameterization}")(
            t, zt=z_t, **{self.cfg.model.denoiser_parameterization: mean_theta}
        )

        if self.cfg.model.flow.variance == "learned_range":
            v_theta = (pred[..., self.d_data:2*self.d_data, :, :] + 1) / 2
        else:
            v_theta = None

        # NOTE sigma parameterization is always the same as the model parameterization!
        pixel_sigma_theta = None
        if self.cfg.model.learn_sigma:
            logvar_theta = pred[..., -1:, :, :]
            # Create per-pixel uncertainty before any pooling
            # Clip logvar to prevent extreme values when exp is applied
            # clipped_logvar = torch.clamp(logvar_theta, -20, 10)  # Reasonable range for log variance
            pixel_sigma_theta = torch.exp(0.5 * logvar_theta)
            
            # Debug print for pixel sigma
            if sample:  # Only print during sampling, not training
                with torch.no_grad():
                    print(f"Raw pixel sigma stats in model.forward: "
                          f"min={pixel_sigma_theta.min().item()}, "
                          f"max={pixel_sigma_theta.max().item()}, "
                          f"mean={pixel_sigma_theta.mean().item()}, "
                          f"shape={pixel_sigma_theta.shape}")
            
            if self.cfg.patch_size is None:
                logvar_theta = logvar_theta.mean(dim=(-2, -1), keepdim=True)
            else:
                if self.cfg.patch_size > 1:
                    shape = logvar_theta.shape
                    logvar_theta = logvar_theta.flatten(0, 1)
                    logvar_theta: Tensor = interpolate(
                        avg_pool2d(
                            logvar_theta, 
                            kernel_size=self.cfg.patch_size, 
                            count_include_pad=False
                        ),
                        size=shape[-2:],
                        mode="nearest-exact"
                    )
                    logvar_theta = logvar_theta.reshape(shape)
            sigma_theta = torch.exp(0.5 * logvar_theta)
            
            # Debug print for patch sigma
            if sample:  # Only print during sampling, not training
                with torch.no_grad():
                    print(f"Processed patch sigma stats in model.forward: "
                          f"min={sigma_theta.min().item()}, "
                          f"max={sigma_theta.max().item()}, "
                          f"mean={sigma_theta.mean().item()}, "
                          f"shape={sigma_theta.shape}")
        else:
            sigma_theta = None

        return mean_theta, v_theta, sigma_theta, pixel_sigma_theta
    
    def training_step(self, batch: BatchedExample, batch_idx) -> Float[Tensor, ""]:
        if self.cfg.conditioning.mask:
            assert "mask" in batch
        # Tell the data loader processes about the current step.
        self.step_tracker.set_step(self.global_step)
        step = self.step_tracker.get_step()
        self.log(f"step_tracker/step", step)
        batch_size, device = batch["index"].size(0), batch["index"].device

        x = batch["image"]
        # Sample timestep map and create optional concatenation conditioning for inpainting baseline
        c_cat = None
        t, loss_weight = self.time_sampler(batch_size, self.cfg.train.num_time_samples, device)
                
        if self.cfg.patch_size is None:
            # t = t.expand_as(x[..., :1, :, :])
            if self.cfg.conditioning.mask:
                # Image level time with standard conditioning on masked (and mask)
                c_cat = torch.cat((batch["mask"], x * (1 - batch["mask"])), dim=1)
        else:            
            t = torch.kron(t, self.float_kernel)
            loss_weight = torch.kron(loss_weight, self.float_kernel)
            if self.cfg.conditioning.mask:
                t.mul_(batch["mask"])
                loss_weight.mul_(batch["mask"])

        # Create remaining conditionings by modulation
        label = batch["label"] if self.cfg.conditioning.label else None

        # Unify to same shape: [batch sample channel height width]
        x = x.unsqueeze(1).expand(batch_size, self.cfg.train.num_time_samples, *x.shape[-3:])
        t = t.unsqueeze(2).expand_as(x[..., :1, :, :])
        loss_weight = loss_weight.unsqueeze(2)
        if c_cat is not None:
            c_cat = c_cat.unsqueeze(1).expand(batch_size, self.cfg.train.num_time_samples, *c_cat.shape[-3:])

        eps = self.flow.sample_eps(x)
        z_t = self.flow.get_zt(t, eps=eps, x=x)

        mean_theta, v_theta, sigma_theta, pixel_sigma_theta = self.forward(z_t, t, label, c_cat=c_cat)

        if self.cfg.model.parameterization == "eps":
            target = eps
        elif self.cfg.model.parameterization == "ut":
            target = self.flow.get_ut(t, eps=eps, x=x)
        else:
            raise ValueError(f"Unknown parameterization {self.cfg.model.parameterization}")

        loss_unweighted = mse_loss(mean_theta, target, reduction="none")
        loss_weighted = loss_weight * loss_unweighted
        if self.cfg.train.log_flow_loss_per_time_split:
            self.log_time_split_loss("unweighted/mu", loss_unweighted, t)
            self.log_time_split_loss("weighted/mu", loss_weighted, t)
        self.log("loss/unweighted/mu", loss_unweighted.detach().mean())
        loss = loss_weighted.mean()
        self.log("loss/weighted/mu", loss)
        
        if self.cfg.model.flow.variance == "learned_range" and self.cfg.loss.vlb.weight > 0 \
            and step > self.cfg.loss.vlb.apply_after_step:
            t_next = t - self.cfg.loss.vlb.time_step_size
            nll_mask = t_next < self.cfg.model.time_eps
            t_next[nll_mask] = 0
            p_theta = self.flow.conditional_p(mean_theta.detach(), z_t, t, t_next, alpha=1, v_theta=v_theta)
            q = self.flow.conditional_q(x, eps, t, t_next, alpha=1)
            kl = q.kl(p_theta)
            nll = -p_theta.discretized_log_likelihood(x)
            vlb_unweighted = torch.where(nll_mask, nll, kl) / log(2.0)
            vlb_weighted = loss_weight * vlb_unweighted
            if self.cfg.train.log_sigma_loss_per_time_split:
                self.log_time_split_loss("unweighted/vlb", vlb_unweighted, t)
                self.log_time_split_loss("weighted/vlb", vlb_weighted, t)
            self.log("loss/unweighted/vlb", vlb_unweighted.detach().mean())
            vlb_loss = vlb_weighted.mean()
            self.log(f"loss/weighted/vlb", vlb_loss)
            loss = loss + self.cfg.loss.vlb.weight * vlb_loss
        
        if self.cfg.model.learn_sigma and self.cfg.loss.sigma.weight > 0 \
            and step > self.cfg.loss.sigma.apply_after_step:
            pred_theta = DiagonalGaussian(mean_theta.detach(), std=sigma_theta)
            sigma_loss_unweighted = pred_theta.nll(target)
            sigma_loss_weighted = loss_weight * sigma_loss_unweighted
            if self.cfg.train.log_sigma_loss_per_time_split:
                self.log_time_split_loss("unweighted/sigma", sigma_loss_unweighted, t)
                self.log_time_split_loss("weighted/sigma", sigma_loss_weighted, t)
            self.log("loss/unweighted/sigma", sigma_loss_unweighted.detach().mean())
            sigma_loss = sigma_loss_weighted.mean()
            self.log(f"loss/weighted/sigma", sigma_loss)
            loss = loss + self.cfg.loss.sigma.weight * sigma_loss

        self.log(f"loss/total", loss)
        if self.global_rank == 0 and not DEBUG and step % self.trainer.log_every_n_steps == 0 \
            and (self.trainer.fit_loop.total_batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Print progress
            print(f"train step = {step}; loss = {loss:.6f};")

        return loss
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if self.ema_denoiser is not None:
            self.ema_denoiser.update_parameters(self.denoiser)
        return super(Wrapper, self).on_train_batch_end(outputs, batch, batch_idx)
    

    def test_step(
        self, 
        batch: BatchedEvaluationExample, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        # Call test method of corresponding evaluation
        datamodule: "DataModule" = self.trainer.datamodule
        datamodule.test_data[batch["id"]].test(self, batch["data"], batch_idx)


    def validation_step(
        self, 
        batch: BatchedEvaluationExample, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        # Call validation method of corresponding evaluation
        datamodule: "DataModule" = self.trainer.datamodule
        datamodule.val_data[batch["id"]].validate(self, batch["data"], batch_idx)
    

    @property
    def effective_batch_size(self) -> int:
        datamodule: "DataModule" = self.trainer.datamodule
        # assumes one fixed batch_size for all train dataloaders!
        return self.trainer.accumulate_grad_batches \
            * self.trainer.num_devices \
            * self.trainer.num_nodes \
            * datamodule.data_loader_cfg.train.batch_size

    @property
    def num_training_steps_per_epoch(self) -> int:
        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size // self.trainer.accumulate_grad_batches
        return num_steps


    @property
    def num_training_steps(self) -> int:
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps
        return self.trainer.max_epochs * self.num_training_steps_per_epoch


    def configure_optimizers(self):
        cfg = self.cfg.optimizer
        kwargs = {} if cfg.kwargs is None else cfg.kwargs
        opt_class = getattr(optim, cfg.name)
        opt_signature = inspect.signature(opt_class)
        if "eps" in opt_signature.parameters.keys():
            # Too small epsilon in optimizer can cause training instabilities with half precision
            kwargs |= dict(eps=1.e-8 if self.trainer.precision in get_args(FullPrecision) else 1.e-7)
        if "weight_decay" in opt_signature.parameters.keys():
            wd, no_wd = self.denoiser.get_weight_decay_parameter_groups()
            params = [{"params": wd}, {"params": no_wd, "weight_decay": 0.}]
        else:
            params = [p for p in self.denoiser.parameters() if p.requires_grad]
        lr = self.cfg.optimizer.lr
        if self.cfg.optimizer.scale_lr:
            lr *= self.effective_batch_size
        opt = opt_class(params, lr=lr, **kwargs)
        res = {"optimizer": opt}
        # Generator scheduler
        if self.cfg.optimizer.scheduler is not None:
            cfg = self.cfg.optimizer.scheduler
            res["lr_scheduler"] = {
                "scheduler": getattr(optim.lr_scheduler, cfg.name)(
                    opt, **(cfg.kwargs if cfg.kwargs is not None else {})
                ),
                "interval": cfg.interval,
                "frequency": cfg.frequency
            }
            if cfg.monitor is not None:
                res["lr_scheduler"]["monitor"] = cfg.monitor
        return res
