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
    top_k: int = 1
    epsilon: float = 1e-6
    reverse_certainty: bool = False # If True, the top_k patches with the highest sigma_theta are selected


class SimultaneousAdaptiveSampler(Sampler[SimultaneousAdaptiveSamplerCfg]):

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

    def get_next_patch_ids(
        self,
        sigma_theta: Float[Tensor, "batch d_data height width"],
        is_unknown_map: Bool[Tensor, "batch num_patches"],
    ) -> Int64[Tensor, "batch top_k"]:
        total_patches = prod(self.patch_grid_shape)

        patch_sigma_theta = avg_pool2d(
            sigma_theta, kernel_size=self.patch_size, count_include_pad=False
        ).reshape(-1, total_patches)

        if self.cfg.reverse_certainty:
            patch_sigma_theta = patch_sigma_theta * is_unknown_map

            smallest_sigma_indices = torch.topk(
                patch_sigma_theta, self.cfg.top_k, largest=True
            ).indices 
        else:
            # Get K non-masked regions with lowest sigma_theta for each batch element
            known_shift = patch_sigma_theta.max() + 1  # to avoid known regions
            patch_sigma_theta = patch_sigma_theta + ~is_unknown_map * known_shift

            smallest_sigma_indices = torch.topk(
                patch_sigma_theta, self.cfg.top_k, largest=False
            ).indices

        return smallest_sigma_indices

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
        
        # Calculate the decrement per operation 
        decrement_per_step = 1.0 / (self.cfg.max_steps * self.cfg.top_k / total_patches)
        
        # Debug info
        print(f"DEBUG: Decrement per operation = {decrement_per_step:.6f}")
        print(f"DEBUG: Total patches = {total_patches}, Max steps = {self.cfg.max_steps}, Top K = {self.cfg.top_k}")
        
        # Track which patches have been selected at each step
        selected_patches = torch.zeros_like(is_unknown_map, dtype=torch.int)
        # Track fully denoised patches
        fully_denoised_patches = torch.zeros_like(is_unknown_map, dtype=torch.bool)

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
            
            # Compute patches to denoise at every step as long as we have unknown patches
            if is_unknown_map.sum() > self.cfg.epsilon:
                # Get batch elements that have unknown patches
                should_predict_batch_ids = is_unknown_map.any(dim=1).nonzero(as_tuple=True)[0]
                
                if should_predict_batch_ids.numel() > 0:
                    # Get uncertainty values for relevant batch elements
                    sigma_theta_relevant = sigma_theta[should_predict_batch_ids]
                    is_unknown_map_relevant = is_unknown_map[should_predict_batch_ids]
                    
                    # Select patches with lowest uncertainty
                    next_ids = self.get_next_patch_ids(
                        sigma_theta_relevant, is_unknown_map_relevant
                    )
                    
                    # Prepare batch and patch indices
                    repeat_batch_ids = torch.repeat_interleave(
                        should_predict_batch_ids, repeats=next_ids.shape[1]
                    )
                    
                    flat_next_ids = next_ids.flatten()
                    
                    # Track which patches were selected in this step
                    selected_patches[repeat_batch_ids, flat_next_ids] += 1
                    
                    # Update scheduling matrix for next step
                    if step_id + 1 < self.cfg.max_steps:
                        # Get current noise values for selected patches
                        current_values = scheduling_matrix[step_id, repeat_batch_ids, flat_next_ids]
                        
                        # Compute new noise values with the fixed decrement
                        new_values = torch.clamp(current_values - decrement_per_step, min=0.0)
                        
                        # Update the scheduling matrix for next step
                        scheduling_matrix[step_id + 1, repeat_batch_ids, flat_next_ids] = new_values
                        
                        # Set all future steps to be at most the new value
                        # This is the critical fix - ensures denoising is monotonic
                        if step_id + 2 < self.cfg.max_steps:
                            for future_step in range(step_id + 2, self.cfg.max_steps + 1):
                                scheduling_matrix[future_step, repeat_batch_ids, flat_next_ids] = torch.minimum(
                                    scheduling_matrix[future_step, repeat_batch_ids, flat_next_ids],
                                    new_values
                                )
                        
                        # Check if any patches are now fully denoised
                        newly_denoised = new_values <= self.cfg.epsilon
                        if newly_denoised.any():
                            # Get indices of newly denoised patches
                            denoised_batch_ids = repeat_batch_ids[newly_denoised]
                            denoised_patch_ids = flat_next_ids[newly_denoised]
                            
                            # Mark these patches as fully denoised
                            fully_denoised_patches[denoised_batch_ids, denoised_patch_ids] = True
                            
                            # Remove them from the unknown map so they won't be selected again
                            is_unknown_map[denoised_batch_ids, denoised_patch_ids] = False
                            
                            # Set all future steps to 0 for fully denoised patches
                            if step_id + 2 < self.cfg.max_steps:
                                scheduling_matrix[step_id + 2:, denoised_batch_ids, denoised_patch_ids] = 0
            
            # Ensure the final step has all zeros (fully denoised)
            scheduling_matrix[-1] = 0
            
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

            # Periodic debug info
            if step_id % 20 == 0 or step_id == self.cfg.max_steps - 1:
                avg_selections = selected_patches.sum().item() / max(1, is_unknown_map.sum().item() + fully_denoised_patches.sum().item())
                print(f"DEBUG: Step {step_id+1}/{self.cfg.max_steps}, "
                      f"Unknown patches: {is_unknown_map.sum().item()}/{total_patches*batch_size}, "
                      f"Selection count: {selected_patches.sum().item()} (avg: {avg_selections:.1f}/patch), "
                      f"Fully denoised: {fully_denoised_patches.sum().item()}/{total_patches*batch_size}")

            # Break early if all patches are fully denoised
            if t_next.max() <= self.cfg.epsilon:
                print(f"DEBUG: Early stopping at step {step_id+1}/{self.cfg.max_steps} - all patches denoised")
                break

        # Final debug info
        avg_selections = selected_patches.sum().item() / max(1, is_unknown_map.sum().item() + fully_denoised_patches.sum().item())
        print(f"DEBUG: Sampling completed - Selected {selected_patches.sum().item()} patches total (avg: {avg_selections:.1f}/patch)")
        print(f"DEBUG: Fully denoised patches: {fully_denoised_patches.sum().item()}/{total_patches*batch_size}")
        
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
