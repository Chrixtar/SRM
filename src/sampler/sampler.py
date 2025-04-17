from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Generic, Sequence, TypeVar

from jaxtyping import Float, Int64
import torch
from torch import Tensor

from src.type_extensions import SamplingOutput
from ..model import Wrapper


@dataclass
class SamplerCfg:
    name: str
    max_steps: int = 100
    alpha: float | int = 0
    temperature: float | int = 1
    use_ema: bool = True    # NOTE ignored if model does not have EMA


T = TypeVar("T", bound=SamplerCfg)


class Sampler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        patch_size: int | None = None,
        patch_grid_shape: Sequence[int] | None = None,
        dependency_matrix: Float[Tensor, "num_patches num_patches"] | None = None
    ) -> None:
        super(Sampler, self).__init__()
        self.cfg = cfg
        self.patch_size = patch_size
        self.patch_grid_shape = patch_grid_shape
        self.num_patches = prod(self.patch_grid_shape)
    
    @property
    def num_timesteps(self) -> int:
        return self.cfg.max_steps

    
    def sampling_step(
        self,
        model: Wrapper,
        z_t: Float[Tensor, "batch dim height width"], 
        t: Float[Tensor, "batch 1 height width"],
        t_next: Float[Tensor, "batch 1 height width"],
        label: Int64[Tensor, "batch"] | None = None,
        c_cat: Float[Tensor, "batch d_c height width"] | None = None,
        return_sigma: bool = True,
        return_x: bool = True
    ) -> tuple[
        Float[Tensor, "batch dim height width"],
        Float[Tensor, "batch 1 height width"],
        Float[Tensor, "batch 1 height width"] | None,
        Float[Tensor, "batch dim height width"] | None,
        Float[Tensor, "batch 1 height width"] | None
    ]:
        z_t = z_t.unsqueeze(1)
        t = t.unsqueeze(1)
        if c_cat is not None:
            c_cat = c_cat.unsqueeze(1)

        mean_theta, v_theta, sigma_theta, pixel_sigma_theta = model.forward(
            z_t, t, label, c_cat, sample=True, use_ema=self.cfg.use_ema
        )

        if sigma_theta is not None:
            sigma_theta.squeeze_(1)
        
        if pixel_sigma_theta is not None:
            pixel_sigma_theta.squeeze_(1)
            # Debug pixel sigma
            with torch.no_grad():
                print(f"pixel_sigma in sampling_step before masking: "
                      f"min={pixel_sigma_theta.min().item()}, "
                      f"max={pixel_sigma_theta.max().item()}, "
                      f"mean={pixel_sigma_theta.mean().item()}, "
                      f"shape={pixel_sigma_theta.shape}, "
                      f"has_zeros={torch.any(pixel_sigma_theta == 0).item()}")
        
        # Apply masking to both patch and pixel level uncertainties
        sigma_theta_masked = sigma_theta.masked_fill_(t.squeeze(1) == 0, 0) if return_sigma else None
        pixel_sigma_theta_masked = pixel_sigma_theta.masked_fill_(t.squeeze(1) == 0, 0) if return_sigma else None
        
        # Debug masked pixel sigma
        if return_sigma and pixel_sigma_theta_masked is not None:
            with torch.no_grad():
                print(f"pixel_sigma in sampling_step after masking: "
                      f"min={pixel_sigma_theta_masked.min().item()}, "
                      f"max={pixel_sigma_theta_masked.max().item()}, "
                      f"mean={pixel_sigma_theta_masked.mean().item()}, "
                      f"shape={pixel_sigma_theta_masked.shape}, "
                      f"zeros_ratio={torch.sum(pixel_sigma_theta_masked == 0).item() / pixel_sigma_theta_masked.numel()}")
        
        conditional_p = model.flow.conditional_p(
            mean_theta, z_t, t, t_next.unsqueeze(1), self.cfg.alpha, self.cfg.temperature, v_theta=v_theta
        )
        x = model.flow.get_x(t, zt=z_t, **{model.cfg.model.parameterization: mean_theta}).squeeze(1) if return_x else None
        # no noise when t_next == 0
        z_t_next = torch.where(t_next.unsqueeze(1) > 0, conditional_p.sample(), conditional_p.mean)
        z_t_next = z_t_next.squeeze(1)
        return z_t_next, t_next, sigma_theta_masked, x, pixel_sigma_theta_masked

    def get_defaults(
        self,
        model: Wrapper,
        batch_size: int | None = None, 
        image_shape: Sequence[int] | None = None,
        z_t: Float[Tensor, "batch dim height width"] | None = None,
        t: Float[Tensor, "batch 1 height width"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
        masked: Float[Tensor, "batch dim height width"] | None = None
    ) -> tuple[
        Float[Tensor, "batch dim height width"],        # z_t
        Float[Tensor, "batch 1 height width"],          # t
        Int64[Tensor, "batch"] | None,                  # label
        Float[Tensor, "batch d_c height width"] | None, # c_cat
        Float[Tensor, "batch dim height width"] | None  # eps
    ]:
        assert z_t is not None or (batch_size is not None and image_shape is not None)
        if z_t is None:
            z_t = torch.randn((batch_size, model.d_data, *image_shape), device=model.device)
        if not model.cfg.conditioning.label:
            # Ignore given labels if no conditioning on label 
            label = None
        # Handle inpainting depending on approach
        c_cat = None
        eps = None
        if mask is not None:
            if model.cfg.patch_size is None:
                if model.cfg.conditioning.mask:     # use mask only as conditioning if trained so
                    c_cat = torch.cat((mask, masked), dim=1)
                eps = z_t if t is None else model.flow.get_eps(t, x=masked, zt=z_t)
            else:
                z_t = masked + mask * z_t
                t = mask if t is None else torch.minimum(mask, t)
        else:
            t = torch.full_like(z_t[:, :1], fill_value=model.cfg.model.time_interval[1])
        return z_t, t, label, c_cat, eps
    
    @abstractmethod
    def sample(
        self, 
        model: Wrapper,
        batch_size: int | None = None, 
        image_shape: Sequence[int] | None = None,
        z_t: Float[Tensor, "batch dim height width"] | None = None,
        t: Float[Tensor, "batch 1 height width"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
        masked: Float[Tensor, "batch dim height width"] | None = None,
        return_intermediate: bool = False,
        return_time: bool = False,
        return_sigma: bool = False,
        return_x: bool = False
    ) -> SamplingOutput:
        pass
        
    def __call__(
        self, 
        model: Wrapper,
        batch_size: int | None = None, 
        image_shape: Sequence[int] | None = None,
        z_t: Float[Tensor, "batch dim height width"] | None = None,
        t: Float[Tensor, "batch 1 height width"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
        masked: Float[Tensor, "batch dim height width"] | None = None,
        return_intermediate: bool = False,
        return_time: bool = False,
        return_sigma: bool = False,
        return_x: bool = False
    ) -> SamplingOutput:
        assert (mask is None) == (masked is None), "mask and masked must be both given or None"
        res = self.sample(
            model, 
            batch_size, 
            image_shape, 
            z_t, 
            t, 
            label, 
            mask, 
            masked, 
            return_intermediate, 
            return_time,
            return_sigma,
            return_x
        )
        return res
