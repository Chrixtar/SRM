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
class SequentialAdaptiveSamplerCfg(SamplerCfg):
    name: Literal["sequential_adaptive"]
    top_k: int = 1
    overlap: float = 0.1
    epsilon: float = 1e-6
    reverse_certainty: bool = False # If True, the top_k patches with the highest sigma_theta are selected


class SequentialAdaptiveSampler(Sampler[SequentialAdaptiveSamplerCfg]):

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

        is_unknown_map = (
            self.full_mask_to_sequence_mask(mask)
            if mask is not None
            else torch.ones(batch_size, total_patches, device=device)
        ) > 0.5  # [batch_size, total_patches]

        scheduling_matrix = torch.ones(
            [self.cfg.max_steps + 1 , batch_size, total_patches], device=device
        )

        # Zero out known regions
        scheduling_matrix *= is_unknown_map.unsqueeze(0)
        # [max_steps, batch_size, total_patches]

        num_unknown_patches = is_unknown_map.sum(dim=1).long()
        # [batch_size]

        num_inference_blocks = torch.ceil(num_unknown_patches / self.cfg.top_k).int()
        ideal_block_lengths = self.get_inference_lengths(num_inference_blocks)
        block_lengths = ideal_block_lengths.ceil().int()  # [batch_size]
        block_starts = (
            torch.arange(num_inference_blocks.max() + 1, device=device).unsqueeze(0)
            * ideal_block_lengths.unsqueeze(1) * (1 - self.cfg.overlap)
        ).floor_()
        block_starts[:,-1] = -1 # This extra block should never be used! 
        block_counters = torch.zeros(batch_size, device=device, dtype=torch.int64)
        step_targets = torch.zeros(batch_size, device=device, dtype=torch.int64)
        
        prototypes = self.get_schedule_prototypes(block_lengths)

        all_z_t = []
        if return_intermediate:
            all_z_t.append(z_t)

        all_t = []
        all_sigma = []
        all_pixel_sigma = []  # Track per-pixel sigma values
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
            should_predict = step_targets == step_id

            if is_unknown_map.sum() > self.cfg.epsilon and should_predict.any():
                block_counters += should_predict.int()
                step_targets = block_starts[torch.arange(batch_size, device=device), block_counters]
                
                should_predict_batch_ids = should_predict.nonzero(as_tuple=True)[0]

                sigma_theta_relevant = sigma_theta[should_predict_batch_ids]
                is_unknown_map_relevant = is_unknown_map[should_predict_batch_ids]
                prototypes_relevant = prototypes[:, should_predict_batch_ids]

                next_ids = self.get_next_patch_ids(
                    sigma_theta_relevant, is_unknown_map_relevant
                )  # This might include patches that are already known for K > 1

                repeat_batch_ids = torch.repeat_interleave(
                    should_predict_batch_ids, repeats=next_ids.shape[1]
                )

                repeat_prototypes = torch.repeat_interleave(
                    prototypes_relevant, repeats=next_ids.shape[1], dim=1
                )

                flat_next_ids = next_ids.flatten()
                is_unknown_map[repeat_batch_ids, flat_next_ids] = False
                
                length_to_consider = min(repeat_prototypes.shape[0], self.cfg.max_steps - step_id)

                # Paste the prototype into the scheduling matrix
                # We need the torch.minimum because for K>1, we might have chosen a patch that is already known
                scheduling_matrix[
                    step_id : step_id + length_to_consider,
                    repeat_batch_ids,
                    flat_next_ids,
                ] = torch.minimum(
                    repeat_prototypes[:length_to_consider],
                    scheduling_matrix[
                        step_id : step_id + length_to_consider,
                        repeat_batch_ids,
                        flat_next_ids,
                    ],
                )

                if step_id + length_to_consider < self.cfg.max_steps:
                    scheduling_matrix[
                        step_id + length_to_consider:, repeat_batch_ids, flat_next_ids
                    ] = 0
                    
                scheduling_matrix[-1] = 0

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
                # with torch.no_grad():
                #     last_pixel_sigma = all_pixel_sigma[-1]
                #     print(f"Collecting pixel sigma (step {step_id}): "
                #           f"min={last_pixel_sigma.min().item()}, "
                #           f"max={last_pixel_sigma.max().item()}, "
                #           f"mean={last_pixel_sigma.mean().item()}, "
                #           f"zeros={torch.sum(last_pixel_sigma == 0).item() / last_pixel_sigma.numel():.2f}")
                          
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
                # with torch.no_grad():
                #     print(f"Final pixel sigma stack: shape={all_pixel_sigma.shape}")
                #     non_zero_mask = all_pixel_sigma > 0
                #     if torch.any(non_zero_mask):
                #         print(f"Non-zero pixel sigma values: "
                #               f"min={all_pixel_sigma[non_zero_mask].min().item()}, "
                #               f"max={all_pixel_sigma[non_zero_mask].max().item()}")
                #     else:
                #         print("WARNING: No non-zero pixel sigma values found in final stack!")
                
                res["all_pixel_sigma"] = list(all_pixel_sigma.transpose(0, 1))
            
            if return_x:
                all_x = torch.stack((*all_x, all_x[-1]), dim=0)
                res["all_x"] = list(all_x.transpose(0, 1))

        return res
