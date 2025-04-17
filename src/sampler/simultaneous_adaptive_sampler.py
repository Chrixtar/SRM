from dataclasses import dataclass
from math import prod
from typing import Literal, Sequence

from jaxtyping import Float, Int64, Bool, Int32
import torch
from torch import Tensor
from torch.nn.functional import avg_pool2d, interpolate

from src.type_extensions import SamplingOutput
from ..model import Wrapper
from .sampler import Sampler, SamplerCfg


@dataclass
class SimultaneousAdaptiveSamplerCfg(SamplerCfg):
    name: Literal["simultaneous_adaptive"]
    epsilon: float = 1e-6
    reverse_certainty: bool = False  # If True, patches with higher sigma_theta get longer schedules
    min_steps_ratio: float = 0.3  # Minimum schedule length as a fraction of max_steps


class SimultaneousAdaptiveSampler(Sampler[SimultaneousAdaptiveSamplerCfg]):

    def full_mask_to_sequence_mask(
        self,
        full_mask: Float[Tensor, "batch_size channels height width"] | None,
    ) -> Float[Tensor, "batch_size num_patches"]:
        batch_size = full_mask.shape[0]
        sequence_mask = full_mask[:, 0, :: self.patch_size, :: self.patch_size]
        sequence_mask = sequence_mask.reshape(batch_size, -1)

        return sequence_mask
    
    def get_schedule_from_uncertainty(
        self,
        sigma_theta: Float[Tensor, "batch d_data height width"],
        is_unknown_map: Bool[Tensor, "batch num_patches"],
    ) -> Float[Tensor, "batch num_patches"]:
        """
        Computes schedule lengths for each patch based on uncertainty.
        Lower uncertainty (sigma_theta) leads to longer schedules.
        """
        total_patches = prod(self.patch_grid_shape)
        batch_size = sigma_theta.shape[0]
        device = sigma_theta.device

        # Get patch-level sigma_theta by average pooling
        patch_sigma_theta = avg_pool2d(
            sigma_theta, kernel_size=self.patch_size, count_include_pad=False
        ).reshape(batch_size, total_patches)
        
        # Set known patches (from mask) to have 0 steps
        patch_sigma_theta = patch_sigma_theta * is_unknown_map
        
        # Normalize sigma values to [0, 1] within each batch
        max_sigma, _ = patch_sigma_theta.max(dim=1, keepdim=True)
        min_sigma, _ = torch.min(
            patch_sigma_theta + (~is_unknown_map) * max_sigma, 
            dim=1, 
            keepdim=True
        )
        
        # Avoid division by zero
        sigma_range = max_sigma - min_sigma
        sigma_range = torch.where(sigma_range < self.cfg.epsilon, 
                                torch.ones_like(sigma_range), 
                                sigma_range)
        
        normalized_sigma = (patch_sigma_theta - min_sigma) / sigma_range
        
        if self.cfg.reverse_certainty:
            # Higher uncertainty -> longer schedule
            schedule_ratios = normalized_sigma
        else:
            # Lower uncertainty -> longer schedule
            schedule_ratios = 1.0 - normalized_sigma
        
        # Ensure minimum schedule length for unknown patches
        min_steps = self.cfg.min_steps_ratio
        schedule_ratios = schedule_ratios * (1.0 - min_steps) + min_steps
        
        # Multiply by max_steps to get actual schedule lengths
        schedule_lengths = schedule_ratios * self.cfg.max_steps
        
        # Ensure known patches have 0 length
        schedule_lengths = schedule_lengths * is_unknown_map
        
        return schedule_lengths
    
    def build_scheduling_matrix(
        self,
        schedule_lengths: Float[Tensor, "batch num_patches"],
        is_unknown_map: Bool[Tensor, "batch num_patches"],
    ) -> Float[Tensor, "max_steps batch_size total_patches"]:
        """
        Builds a scheduling matrix where each entry indicates the timestep for each patch at each step.
        """
        batch_size, total_patches = schedule_lengths.shape
        device = schedule_lengths.device
        
        # Create step indices for each timestep
        steps = torch.arange(self.cfg.max_steps + 1, device=device).unsqueeze(-1).unsqueeze(-1)
        # [max_steps+1, 1, 1]
        
        # Expand to match batch and patches
        steps = steps.expand(-1, batch_size, total_patches)
        # [max_steps+1, batch_size, total_patches]
        
        # For each patch, calculate the timestep at each step
        # Start from 1.0 (fully noised) and decrease linearly to 0.0 (fully denoised)
        scheduling_matrix = 1.0 - steps / schedule_lengths.unsqueeze(0).clamp(min=1.0)
        
        # Clamp values to [0, 1]
        scheduling_matrix.clamp_(0.0, 1.0)
        
        # Ensure known patches are always 0 in the schedule
        scheduling_matrix *= is_unknown_map.unsqueeze(0)
        
        return scheduling_matrix
    
    def get_timestep_from_schedule(
        self,
        scheduling_matrix: Float[Tensor, "total_steps batch_size total_patches"],
        step_id: int,
        image_shape: Sequence[int],
    ) -> Float[Tensor, "batch_size 1 height width"]:
        assert step_id < scheduling_matrix.shape[0]
        batch_size = scheduling_matrix.shape[1]

        t_patch = scheduling_matrix[step_id].reshape(batch_size, *self.patch_grid_shape)
        return interpolate(t_patch.unsqueeze(1), size=image_shape, mode="nearest-exact")

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
        image_shape = z_t.shape[-2:] if image_shape is None else image_shape
        total_patches = prod(self.patch_grid_shape)
        batch_size = z_t.size(0)
        device = z_t.device

        z_t, t, label, c_cat, eps = self.get_defaults(
            model, batch_size, image_shape, z_t, t, label, mask, masked
        )

        is_unknown_map = (
            self.full_mask_to_sequence_mask(mask)
            if mask is not None
            else torch.ones(batch_size, total_patches, device=device)
        ) > 0.5  # [batch_size, total_patches]

        # Initial evaluation to get uncertainty
        with torch.no_grad():
            z_t_expanded = z_t.unsqueeze(1)
            t_expanded = t.unsqueeze(1)
            
            if c_cat is not None:
                c_cat = c_cat.unsqueeze(1)
                
            _, _, sigma_theta, _ = model.forward(
                z_t=z_t_expanded,
                t=t_expanded,
                label=label,
                c_cat=c_cat,
                sample=True,
                use_ema=self.cfg.use_ema
            )
            
            sigma_theta.squeeze_(1)
        
        # Determine schedule lengths based on uncertainty
        schedule_lengths = self.get_schedule_from_uncertainty(sigma_theta, is_unknown_map)
        
        # Build scheduling matrix for the entire sampling process
        scheduling_matrix = self.build_scheduling_matrix(schedule_lengths, is_unknown_map)

        breakpoint()

        all_z_t = []
        if return_intermediate:
            all_z_t.append(z_t)

        all_t = []
        all_sigma = []
        all_pixel_sigma = []
        all_x = []
        last_next_t = None

        if c_cat is not None:
            c_cat = c_cat.unsqueeze(1)

        for step_id in range(self.cfg.max_steps):
            t = self.get_timestep_from_schedule(scheduling_matrix, step_id, image_shape)
            
            z_t = z_t.unsqueeze(1)
            t = t.unsqueeze(1)
            
            mean_theta, v_theta, sigma_theta, pixel_sigma_theta = model.forward(
                z_t=z_t,
                t=t,
                label=label,
                c_cat=c_cat,
                sample=True,
                use_ema=self.cfg.use_ema
            )
            
            sigma_theta.squeeze_(1)
            pixel_sigma_theta.squeeze_(1)
            
            t_next = self.get_timestep_from_schedule(
                scheduling_matrix, step_id + 1, image_shape
            )

            conditional_p = model.flow.conditional_p(
                mean_theta, z_t, t, t_next.unsqueeze(1), self.cfg.alpha, self.cfg.temperature, v_theta=v_theta
            )
            # no noise when t_next == 0
            z_t = torch.where(t_next.unsqueeze(1) > 0, conditional_p.sample(), conditional_p.mean)
            z_t.squeeze_(1)
            t = t.squeeze(1)

            if mask is not None:
                # Repaint
                if self.patch_size is None:
                    z_t = (1 - mask) * model.flow.get_zt(
                        t_next, x=masked, eps=eps
                    ) + mask * z_t
                else:
                    z_t = masked + mask * z_t
                    
            if return_intermediate:
                all_z_t.append(z_t)
            if return_time:
                all_t.append(t)
                last_next_t = t_next
            if return_sigma:
                all_sigma.append(sigma_theta.masked_fill_(t == 0, 0))
                all_pixel_sigma.append(pixel_sigma_theta.masked_fill_(t == 0, 0))
                
                # Debug for pixel sigma collection
                with torch.no_grad():
                    last_pixel_sigma = all_pixel_sigma[-1]
                    print(f"Collecting pixel sigma (step {step_id}): "
                          f"min={last_pixel_sigma.min().item()}, "
                          f"max={last_pixel_sigma.max().item()}, "
                          f"mean={last_pixel_sigma.mean().item()}, "
                          f"zeros={torch.sum(last_pixel_sigma == 0).item() / last_pixel_sigma.numel():.2f}")
                          
            if return_x:
                all_x.append(model.flow.get_x(t, zt=z_t, **{model.cfg.model.parameterization: mean_theta.squeeze(1)}))

            if t_next.max() <= self.cfg.epsilon:
                break  # No more patches to predict

        res: SamplingOutput = {"sample": z_t}

        if return_intermediate:
            batch_size = z_t.size(0)
            res["all_z_t"] = [
                torch.stack([z_t[i] for z_t in all_z_t]) for i in range(batch_size)
            ]

            if return_time:
                all_t = torch.stack([*all_t, last_next_t], dim=0)
                res["all_t"] = list(all_t.transpose(0, 1))
            
            if return_sigma:
                all_sigma = torch.stack((*all_sigma, all_sigma[-1]), dim=0)
                res["all_sigma"] = list(all_sigma.transpose(0, 1))
                
                # Process per-pixel sigmas the same way as patch-level sigmas
                all_pixel_sigma = torch.stack((*all_pixel_sigma, all_pixel_sigma[-1]), dim=0)
                
                # Debug final pixel sigma stack
                with torch.no_grad():
                    print(f"Final pixel sigma stack: shape={all_pixel_sigma.shape}")
                    non_zero_mask = all_pixel_sigma > 0
                    if torch.any(non_zero_mask):
                        print(f"Non-zero pixel sigma values: "
                              f"min={all_pixel_sigma[non_zero_mask].min().item()}, "
                              f"max={all_pixel_sigma[non_zero_mask].max().item()}")
                    else:
                        print("WARNING: No non-zero pixel sigma values found in final stack!")
                
                res["all_pixel_sigma"] = list(all_pixel_sigma.transpose(0, 1))
            
            if return_x:
                all_x = torch.stack((*all_x, all_x[-1]), dim=0)
                res["all_x"] = list(all_x.transpose(0, 1))

        return res
