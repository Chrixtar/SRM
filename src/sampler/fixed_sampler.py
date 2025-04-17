from dataclasses import dataclass
from typing import Literal, Sequence

from jaxtyping import Float, Int64
import torch
from torch import Tensor

from src.misc.tensor import unsqueeze_multi_dims
from src.type_extensions import SamplingOutput
from ..model import Wrapper
from .sampler import Sampler, SamplerCfg


@dataclass
class FixedSamplerCfg(SamplerCfg):
    name: Literal["fixed"]


class FixedSampler(Sampler[FixedSamplerCfg]):

    @torch.no_grad()
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
        z_t, t, label, c_cat, eps = self.get_defaults(
            model, batch_size, image_shape, z_t, t, label, mask, masked
        )
        all_z_t, all_sigma, all_pixel_sigma, all_x = [], [], [], []
        all_t: Tensor = torch.linspace(
            *model.cfg.model.time_interval[::-1], self.cfg.max_steps+1, device=model.device
        )
        all_t = unsqueeze_multi_dims(all_t, z_t.ndim).expand(-1, *z_t[:, :1].shape)
        if t is not None:
            all_t = torch.minimum(all_t, t)
        if return_intermediate:
            all_z_t.append(z_t)
        for i, t in enumerate(all_t[:-1]):
            t_next = all_t[i+1]
            z_t, _, sigma_theta, x, pixel_sigma_theta = self.sampling_step(
                model,
                z_t=z_t, 
                t=t, 
                t_next=t_next,
                label=label,
                c_cat=c_cat,
                return_sigma=return_sigma,
                return_x=return_x
            )
            if mask is not None:
                # Repaint
                if model.cfg.patch_size is None:
                    z_t = (1-mask) * model.flow.get_zt(t_next, x=masked, eps=eps) + mask * z_t
                else:
                    z_t = masked + mask * z_t
            if return_intermediate:
                all_z_t.append(z_t)
                if return_sigma:
                    all_sigma.append(sigma_theta)
                    all_pixel_sigma.append(pixel_sigma_theta)
                if return_x:
                    all_x.append(x)
        
        z_t = all_z_t[-1] if all_z_t else z_t
        res: SamplingOutput = {"sample": z_t}

        if return_intermediate:
            # Ensure correct output format
            batch_size = z_t.size(0)
            res["all_z_t"] = [torch.stack([z_t[i] for z_t in all_z_t]) \
                              for i in range(batch_size)]
            if return_time:
                res["all_t"] = list(all_t.transpose(0, 1))

            if return_sigma:
                all_sigma = torch.stack((*all_sigma, all_sigma[-1]), dim=0)
                res["all_sigma"] = list(all_sigma.transpose(0, 1))
                
                # Add per-pixel uncertainty
                all_pixel_sigma = torch.stack((*all_pixel_sigma, all_pixel_sigma[-1]), dim=0)
                res["all_pixel_sigma"] = list(all_pixel_sigma.transpose(0, 1))
            
            if return_x:
                all_x = torch.stack([*all_x, all_x[-1]], dim=0)
                res["all_x"] = list(all_x.transpose(0, 1))
        return res
