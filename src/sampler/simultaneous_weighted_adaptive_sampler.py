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
class SimultaneousWeightedAdaptiveSamplerCfg(SamplerCfg):
    name: Literal["simultaneous_weighted_adaptive"]
    epsilon: float = 1e-6
    uncertainty_power: float = 1.0 # Power to raise uncertainty (sigma_theta) for weighting


class SimultaneousWeightedAdaptiveSampler(Sampler[SimultaneousWeightedAdaptiveSamplerCfg]):

    def get_inference_lengths(
        self, num_inference_blocks: Int32[Tensor, "batch_size"]
    ) -> Float[Tensor, "batch_size"]:
        ideal_lengths = self.cfg.max_steps / (
            (num_inference_blocks - 1) * (1 - self.cfg.overlap) + 1
        )

        return ideal_lengths

    def get_schedule_prototypes(
        self, prototype_lengths: Int32[Tensor, "batch_size"]
    ) -> Float[Tensor, "max_prototype_length batch_size"]:
        batch_size = prototype_lengths.size(0)
        device = prototype_lengths.device

        max_prototype_length = prototype_lengths.max()
        assert prototype_lengths.min() > 0

        # We add one to the prototype length to include the timestep with just zeros
        prototype_base = torch.linspace(
            max_prototype_length, 0, max_prototype_length + 1, device=device
        )

        prototypes = prototype_base.unsqueeze(0).expand(batch_size, -1)

        # shift down based on the prototype lengths
        prototypes = prototypes - (max_prototype_length - prototype_lengths).unsqueeze(1)
        # [batch_size, max_prototype_length + 1]

        # scale to batch_max = 1
        prototypes = prototypes / prototype_lengths.unsqueeze(1)
        assert prototypes.max() <= 1 + self.cfg.epsilon

        prototypes.clamp_(0, 1)  # [batch_size, max_prototype_length + 1]
        prototypes = prototypes.T  # [max_prototype_length + 1, batch_size]
        return prototypes[:-1]  # [max_prototype_length, batch_size] skip trailing zeros

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

    def full_mask_to_sequence_mask(
        self,
        full_mask: Float[Tensor, "batch_size channels height width"] | None,
    ) -> Float[Tensor, "batch_size num_patches"]:
        batch_size = full_mask.shape[0]
        sequence_mask = full_mask[:, 0, :: self.patch_size, :: self.patch_size]
        sequence_mask = sequence_mask.reshape(batch_size, -1)

        return sequence_mask

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

        # Create the unknown map to track which patches need denoising
        is_unknown_map = (
            self.full_mask_to_sequence_mask(mask)
            if mask is not None
            else torch.ones(batch_size, total_patches, device=device)
        ) > 0.5  # [batch_size, total_patches]

        # Initialize scheduling matrix with all ones (fully noisy)
        scheduling_matrix = torch.ones(
            [self.cfg.max_steps + 1, batch_size, total_patches], device=device
        )

        # Zero out known regions initially
        scheduling_matrix *= is_unknown_map.unsqueeze(0)
        
        # Calculate the total denoising budget per batch item to distribute per step
        initial_unknown_patches_per_batch = is_unknown_map.sum(dim=1, keepdim=True).float()
        # print(f"DEBUG: Initial unknown patches per batch: {initial_unknown_patches_per_batch}")
        # Budget per step = total initial unknown noise / num steps
        step_denoising_budget_per_batch = initial_unknown_patches_per_batch / self.cfg.max_steps # [batch, 1]

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
            # Get current timestep from scheduling matrix
            t = self.get_timestep_from_schedule(scheduling_matrix, step_id, image_shape)
            
            z_t = z_t.unsqueeze(1)
            t = t.unsqueeze(1)
            
            # Run model forward pass
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
            
            # --- Continuous Denoising Logic --- 
            current_t = scheduling_matrix[step_id]  # [batch, num_patches]
            
            # Pool sigma_theta to patch level for uncertainty weighting
            patch_sigma_theta = avg_pool2d(
                sigma_theta, kernel_size=self.patch_size, count_include_pad=False
            ).reshape(batch_size, total_patches)

            # Identify patches that still need denoising
            noisy_mask = current_t > self.cfg.epsilon  # [batch, num_patches]

            # Initialize delta_t for this step
            delta_t = torch.zeros_like(current_t) # [batch, num_patches]

            # Calculate weights for noisy patches (inverse uncertainty)
            weights = torch.zeros_like(patch_sigma_theta)
            safe_sigma = patch_sigma_theta + self.cfg.epsilon # Add epsilon for stability
            
            # Apply weights only where the mask is true, using uncertainty power
            weights_masked = torch.where(
                noisy_mask, 
                1.0 / (safe_sigma ** self.cfg.uncertainty_power), 
                torch.zeros_like(weights)
            )

            # Normalize weights per batch item ONLY across noisy patches
            sum_weights_per_batch = torch.sum(weights_masked, dim=1, keepdim=True) # [batch, 1]

            # Avoid division by zero if a batch item has no noisy patches left
            safe_sum_weights = torch.where(
                sum_weights_per_batch > self.cfg.epsilon, 
                sum_weights_per_batch, 
                torch.ones_like(sum_weights_per_batch)
            )

            normalized_weights = weights_masked / safe_sum_weights # [batch, num_patches]
            
            # Calculate the denoising amount for each patch for this step, using per-batch budget
            allocated_delta = step_denoising_budget_per_batch * normalized_weights # [batch, num_patches]

            # Update schedule: apply the calculated delta only to noisy patches
            delta_t = torch.where(noisy_mask, allocated_delta, torch.zeros_like(delta_t))
            next_t_candidate = torch.clamp(current_t - delta_t, min=0.0) # [batch, num_patches]
            
            # # -- Debugging Prints --
            # if step_id == 0 or (step_id + 1) % 20 == 0 or step_id == self.cfg.max_steps - 1:
            #     num_noisy_patches = noisy_mask.sum(dim=1).float()
            #     safe_num_noisy = torch.where(num_noisy_patches > 0, num_noisy_patches, torch.ones_like(num_noisy_patches))
                
            #     avg_current_t = (current_t * noisy_mask).sum(dim=1) / safe_num_noisy
            #     avg_sigma = (patch_sigma_theta * noisy_mask).sum(dim=1) / safe_num_noisy
            #     avg_delta_t = (delta_t * noisy_mask).sum(dim=1) / safe_num_noisy
            #     avg_next_t = (next_t_candidate * noisy_mask).sum(dim=1) / safe_num_noisy
                
            #     print(f"-- Step {step_id+1}/{self.cfg.max_steps} Debug --")
            #     print(f"  Shape current_t: {current_t.shape}, patch_sigma_theta: {patch_sigma_theta.shape}, noisy_mask: {noisy_mask.shape}")
            #     print(f"  Shape delta_t: {delta_t.shape}, next_t_candidate: {next_t_candidate.shape}, step_budget/batch: {step_denoising_budget_per_batch.shape}")
            #     # Print stats for the first batch item for simplicity
            #     print(f"  Batch 0: Noisy patches: {num_noisy_patches[0].item():.0f}/{total_patches}")
            #     print(f"  Batch 0: Avg Current t (noisy): {avg_current_t[0].item():.4f}")
            #     print(f"  Batch 0: Avg Sigma (noisy): {avg_sigma[0].item():.4f}")
            #     print(f"  Batch 0: Avg Delta t (noisy): {avg_delta_t[0].item():.4f}")
            #     print(f"  Batch 0: Avg Next t (noisy): {avg_next_t[0].item():.4f}")
            #     print(f"-------------------------")
            # # -- End Debugging --

            # Update matrix for the next step
            if step_id + 1 <= self.cfg.max_steps:
                scheduling_matrix[step_id + 1] = next_t_candidate

            # Propagate minimum forward (ensure monotonicity)
            if step_id + 2 <= self.cfg.max_steps:
                # Create a view for broadcasting comparison [1, batch, num_patches]
                next_t_candidate_expanded = next_t_candidate.unsqueeze(0)
                # Compare and update future steps
                future_steps_slice = scheduling_matrix[step_id + 2:] # [remaining_steps, batch, num_patches]
                updated_future_steps = torch.minimum(future_steps_slice, next_t_candidate_expanded)
                scheduling_matrix[step_id + 2:] = updated_future_steps
            
            # --- End Continuous Denoising Logic ---
            
            # Get next timestep from updated scheduling matrix
            t_next = self.get_timestep_from_schedule(
                scheduling_matrix, step_id + 1, image_shape
            )

            # Sample from conditional distribution
            conditional_p = model.flow.conditional_p(
                mean_theta, z_t, t, t_next.unsqueeze(1), self.cfg.alpha, self.cfg.temperature, v_theta=v_theta
            )
            # No noise when t_next == 0
            z_t = torch.where(t_next.unsqueeze(1) > 0, conditional_p.sample(), conditional_p.mean)
            z_t.squeeze_(1)
            t = t.squeeze(1)

            # Handle masking if needed
            if mask is not None:
                # Repaint
                if self.patch_size is None:
                    z_t = (1 - mask) * model.flow.get_zt(
                        t_next, x=masked, eps=eps
                    ) + mask * z_t
                else:
                    z_t = masked + mask * z_t
                    
            # Collect intermediate results if requested
            if return_intermediate:
                all_z_t.append(z_t)
            if return_time:
                all_t.append(t)
                last_next_t = t_next
            if return_sigma:
                all_sigma.append(sigma_theta.masked_fill_(t == 0, 0))
                all_pixel_sigma.append(pixel_sigma_theta.masked_fill_(t == 0, 0))
                
            if return_x:
                all_x.append(model.flow.get_x(t, zt=z_t, **{model.cfg.model.parameterization: mean_theta.squeeze(1)}))

            # Break early if all patches are fully denoised
            if t_next.max() <= self.cfg.epsilon:
                print(f"DEBUG: Early stopping at step {step_id+1}/{self.cfg.max_steps} - all patches denoised")
                break
        
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
                
                res["all_pixel_sigma"] = list(all_pixel_sigma.transpose(0, 1))
            
            if return_x:
                all_x = torch.stack((*all_x, all_x[-1]), dim=0)
                res["all_x"] = list(all_x.transpose(0, 1))

        return res
