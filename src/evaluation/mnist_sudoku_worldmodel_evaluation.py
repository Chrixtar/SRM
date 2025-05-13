import collections
import itertools
import warnings
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
from jaxtyping import Bool, Integer, Shaped
import torch
from torch import Tensor
import torch.nn.functional as F
import tqdm

from ..dataset import DatasetMnistSudoku9x9Eager, DatasetMnistSudoku9x9Lazy
from ..model import Wrapper
from ..misc.mnist_classifier import get_classifier
from ..global_cfg import get_mnist_classifier_path

from .mnist_evaluation import MnistEvaluation, MnistEvaluationCfg
from .types import EvaluationOutput
from .sampling_evaluation import BatchedSamplingExample

@dataclass
class MnistSudokuWorldModelEvaluationCfg(MnistEvaluationCfg):
    name: Literal["mnist_sudoku_worldmodel"] = "mnist_sudoku_worldmodel"
    bp_iterations: int = 30
    bp_damping: float = 0.5
    num_model_samples: int = 100
    results_subdir: str = "world_model_eval_results"

class MnistSudokuWorldModelEvaluation(MnistEvaluation[MnistSudokuWorldModelEvaluationCfg]):
    def __init__(
        self,
        cfg: MnistSudokuWorldModelEvaluationCfg,
        tag: str,
        dataset: DatasetMnistSudoku9x9Eager | DatasetMnistSudoku9x9Lazy,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)

    def classify(
        self,
        pred: Integer[Tensor, "batch grid_size grid_size"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size ** 0.5)
        dtype, device = pred.dtype, pred.device
        pred = pred - 1 # Shift [1, 9] to [0, 8] for indices
        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred)
        dist = torch.zeros((batch_size,), dtype=dtype, device=device)
        for dim in range(1, 3):
            cnt = torch.full_like(pred, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred, src=ones)
            dist.add_(cnt.abs_().sum(dim=(1, 2)))
        # Subgrids
        grids = pred.unfold(1, sub_grid_size, sub_grid_size)\
            .unfold(2, sub_grid_size, sub_grid_size).reshape(-1, grid_size, grid_size)
        cnt = torch.full_like(grids, fill_value=-1)
        cnt.scatter_add_(dim=dim, index=grids, src=ones)
        dist.add_(cnt.abs_().sum(dim=(1, 2)))
        label = dist == 0
        return label, {"distance": dist}

    @staticmethod
    def _get_box_indices(box_r: int, box_c: int) -> list[tuple[int, int]]:
        """Get cell indices (r, c) for a given 3x3 box index (0-2, 0-2)."""
        start_row, start_col = 3 * box_r, 3 * box_c
        return [(r, c) for r in range(start_row, start_row + 3)
                for c in range(start_col, start_col + 3)]

    @staticmethod
    def _build_factor_graph_connections(grid: np.ndarray) -> tuple[
        collections.defaultdict[tuple[int, int], list],
        collections.defaultdict[tuple[str, int] | tuple[str, int, int], list],
        dict[tuple[str, int] | tuple[str, int, int], dict[str, str | int | tuple[int, int] | np.ndarray]]
    ]:
        """
        Builds the connections for the Sudoku factor graph.

        Returns:
            var_to_fac (dict): Maps cell (r, c) -> list of connected factor IDs.
            fac_to_var (dict): Maps factor ID -> list of connected cells (r, c).
            factors (dict): Maps factor ID -> {'type': 'clue'/'row'/'col'/'box', 'id': original_id}
        """
        var_to_fac = collections.defaultdict(list)
        fac_to_var = collections.defaultdict(list)
        factors = {}

        # Clue Factors (Unary)
        for r, c in itertools.product(range(9), range(9)):
            if grid[r, c] != 0:
                fac_id = ('clue', r, c)
                factors[fac_id] = {'type': 'clue', 'val': int(grid[r, c])}
                var_to_fac[(r, c)].append(fac_id)
                fac_to_var[fac_id].append((r, c))

        # Row Factors (AllDifferent)
        for r in range(9):
            fac_id = ('row', r)
            factors[fac_id] = {'type': 'row', 'id': r}
            cells = [(r, c) for c in range(9)]
            fac_to_var[fac_id] = cells
            for cell in cells:
                var_to_fac[cell].append(fac_id)

        # Column Factors (AllDifferent)
        for c in range(9):
            fac_id = ('col', c)
            factors[fac_id] = {'type': 'col', 'id': c}
            cells = [(r, c) for r in range(9)]
            fac_to_var[fac_id] = cells
            for cell in cells:
                var_to_fac[cell].append(fac_id)

        # Box Factors (AllDifferent)
        for box_r, box_c in itertools.product(range(3), range(3)):
            fac_id = ('box', box_r, box_c)
            factors[fac_id] = {'type': 'box', 'id': (box_r, box_c)}
            cells = MnistSudokuWorldModelEvaluation._get_box_indices(box_r, box_c)
            fac_to_var[fac_id] = cells
            for cell in cells:
                var_to_fac[cell].append(fac_id)

        return var_to_fac, fac_to_var, factors

    @staticmethod
    def _normalize_bp_message(msg: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
        """Normalizes a message (probability distribution) to sum to 1."""
        msg = np.maximum(msg, epsilon) # Avoid zero probabilities before normalization
        norm = np.sum(msg)
        if norm < epsilon:
            warnings.warn("Normalization failed, sum too small. Resetting to uniform.", RuntimeWarning)
            return np.ones_like(msg) / msg.size
        return msg / norm

    @staticmethod
    def _compute_sudoku_probabilities_bp(
        grid: np.ndarray, iterations: int = 30, damping: float = 0.5, epsilon: float = 1e-9
    ) -> np.ndarray | None:
        """
        Computes approximate probability distributions for Sudoku cells using
        Belief Propagation on the factor graph.
        Returns a 9x9x9 numpy array or None if the initial grid is invalid.
        """
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        if grid.shape != (9, 9):
            raise ValueError("Input grid must be 9x9.")

        # Basic Grid Validation
        for r_idx in range(9):
            for c_idx in range(9):
                digit = grid[r_idx, c_idx]
                if not (0 <= digit <= 9):
                    raise ValueError(f"Invalid value {digit} in grid at ({r_idx},{c_idx}).")
                if digit != 0:
                    # Check row
                    if np.count_nonzero(grid[r_idx, :] == digit) > 1: return None
                    # Check col
                    if np.count_nonzero(grid[:, c_idx] == digit) > 1: return None
                    # Check box
                    br, bc = r_idx // 3, c_idx // 3
                    box_indices = MnistSudokuWorldModelEvaluation._get_box_indices(br, bc)
                    box_vals = [grid[rr, cc] for rr, cc in box_indices if grid[rr, cc] != 0]
                    if box_vals.count(digit) > 1: return None
        
        var_to_fac, fac_to_var, factors = MnistSudokuWorldModelEvaluation._build_factor_graph_connections(grid)
        variables = list(itertools.product(range(9), range(9)))

        msg_v_to_f = {}
        msg_f_to_v = {}
        uniform_msg = np.ones(9) / 9.0

        for var in variables:
            for fac_id in var_to_fac[var]:
                msg_v_to_f[(var, fac_id)] = uniform_msg.copy()
                msg_f_to_v[(fac_id, var)] = uniform_msg.copy()

        for _iteration in range(iterations):
            new_msg_v_to_f = {}
            for var in variables:
                connected_factors = var_to_fac[var]
                for fac_id in connected_factors:
                    prod_msg = np.ones(9)
                    for other_fac_id in connected_factors:
                        if other_fac_id != fac_id:
                            prod_msg *= msg_f_to_v[(other_fac_id, var)]
                    
                    computed_msg = MnistSudokuWorldModelEvaluation._normalize_bp_message(prod_msg, epsilon)
                    damped_msg = (1 - damping) * msg_v_to_f[(var, fac_id)] + damping * computed_msg
                    new_msg_v_to_f[(var, fac_id)] = MnistSudokuWorldModelEvaluation._normalize_bp_message(damped_msg, epsilon)
            msg_v_to_f = new_msg_v_to_f

            new_msg_f_to_v = {}
            for fac_id, factor_info in factors.items():
                connected_vars = fac_to_var[fac_id]
                if factor_info['type'] == 'clue':
                    var = connected_vars[0]
                    clue_val = factor_info['val']
                    computed_msg = np.full(9, epsilon / (8.0 + epsilon))
                    computed_msg[clue_val - 1] = 1.0 / (1.0 + epsilon) # Ensure positive
                    computed_msg = MnistSudokuWorldModelEvaluation._normalize_bp_message(computed_msg, epsilon)
                    new_msg_f_to_v[(fac_id, var)] = computed_msg
                elif factor_info['type'] in ['row', 'col', 'box']:
                    for var in connected_vars:
                        prod_msg = np.ones(9)
                        for k_val_idx in range(9): # For each possible value k+1 for 'var'
                            prob_val_k_possible = 1.0
                            for other_var in connected_vars:
                                if other_var != var:
                                    m_other_to_f = msg_v_to_f[(other_var, fac_id)]
                                    prob_other_not_k = 1.0 - m_other_to_f[k_val_idx]
                                    prob_other_not_k = max(0.0, prob_other_not_k)
                                    prob_val_k_possible *= prob_other_not_k
                                    if prob_val_k_possible < epsilon:
                                        prob_val_k_possible = 0.0
                                        break
                            prod_msg[k_val_idx] = prob_val_k_possible
                        
                        computed_msg = MnistSudokuWorldModelEvaluation._normalize_bp_message(prod_msg, epsilon)
                        damped_msg = (1 - damping) * msg_f_to_v[(fac_id, var)] + damping * computed_msg
                        new_msg_f_to_v[(fac_id, var)] = MnistSudokuWorldModelEvaluation._normalize_bp_message(damped_msg, epsilon)
            msg_f_to_v = new_msg_f_to_v
        
        beliefs = np.zeros((9, 9, 9))
        for var in variables:
            r, c = var
            belief = np.ones(9)
            for fac_id in var_to_fac[var]:
                belief *= msg_f_to_v[(fac_id, var)]
            
            beliefs[r, c, :] = MnistSudokuWorldModelEvaluation._normalize_bp_message(belief, epsilon)
            if grid[r,c] != 0:
                clue_val = grid[r,c]
                beliefs[r,c,:] = 0.0
                beliefs[r,c, clue_val - 1] = 1.0
        return beliefs

    @torch.no_grad()
    def evaluate(
        self, 
        model: Wrapper,
        batch: BatchedSamplingExample,
        return_sample: bool = True 
    ) -> Iterator[EvaluationOutput]:
        if not self.samplers:
            warnings.warn("No samplers initialized for MnistSudokuWorldModelEvaluation, skipping evaluation.")
            return
        
        if self.cfg.num_model_samples <= 0:
            warnings.warn("num_model_samples must be positive. Skipping evaluation.")
            return

        batch_image = batch.get("image")
        if batch_image is None:
            warnings.warn("'image' not found in batch for MnistSudokuWorldModelEvaluation. Skipping batch.")
            return
        batch_size = batch_image.shape[0]
        
        batch_mask = batch.get("mask")
        batch_label = batch.get("label")
        batch_index = batch.get("index")
        if batch_index is None:
            warnings.warn("'index' not found in batch for MnistSudokuWorldModelEvaluation. Skipping batch.")
            return
        batch_target_image = batch.get("target_image")

        device = model.device
        image_shape = self.dataset.cfg.image_shape
        d_data = model.d_data
        
        # Get the MNIST classifier instance
        mnist_classifier = get_classifier(get_mnist_classifier_path(), device)

        for sampler_key, sampler_obj in self.samplers.items():
            # --- 3. Inner Loop over Puzzles for Analysis ---
            for i in tqdm.trange(batch_size):
                # --- 1. Prepare Sampler Inputs for CURRENT PUZZLE i ---
                current_z_t: Tensor
                if self.deterministic:
                    generator = torch.Generator(device=device).manual_seed(batch_index[i].item())
                    current_z_t = torch.randn(
                        (self.cfg.num_model_samples, d_data, *image_shape),
                        generator=generator, device=device
                    )
                else:
                    current_z_t = torch.randn(
                        (self.cfg.num_model_samples, d_data, *image_shape),
                        device=device
                    )

                current_label_for_sampler: Tensor | None = None
                if batch_label is not None:
                    current_label_for_sampler = batch_label[i:i+1].repeat_interleave(
                        self.cfg.num_model_samples, dim=0
                    ).to(device)

                current_mask_for_sampler: Tensor | None = None
                current_masked_img_for_sampler: Tensor | None = None
                if batch_mask is not None and batch_image is not None:
                    # Slice and repeat the current puzzle's mask and image
                    current_mask_expanded = batch_mask[i:i+1].repeat_interleave(
                        self.cfg.num_model_samples, dim=0
                    ).to(device)
                    current_image_expanded = batch_image[i:i+1].repeat_interleave(
                        self.cfg.num_model_samples, dim=0
                    ).to(device)
                    current_mask_for_sampler = current_mask_expanded.float()
                    current_masked_img_for_sampler = current_image_expanded * (1.0 - current_mask_expanded.float())
                
                # --- 2. Sampler Call for CURRENT PUZZLE i ---
                sample_output_struct = sampler_obj.__call__(
                    model,
                    z_t=current_z_t,
                    label=current_label_for_sampler,
                    mask=current_mask_for_sampler,
                    masked=current_masked_img_for_sampler,
                    return_intermediate=False
                )
                # sampled_images_tensor_for_puzzle_i now comes directly from the sampler output for the current puzzle
                sampled_images_tensor_for_puzzle_i = sample_output_struct["sample"]

                # 3a. Prepare initial grid for BP for current puzzle i
                image_h, image_w = self.dataset.cfg.image_shape
                # self.grid_size should be (grid_dim_h, grid_dim_w), e.g. (9,9) for Sudoku
                # It is initialized in MnistEvaluation from dataset.grid_size
                grid_h, grid_w = self.grid_size 

                patch_pixel_h = image_h // grid_h
                patch_pixel_w = image_w // grid_w
  
                current_puzzle_image_for_bp = batch_image[i:i+1]
                all_cell_digits_pred = self.discretize(mnist_classifier, current_puzzle_image_for_bp)
                
                # Default to a mask of all False (no clues) if batch_mask is not present for this item
                current_mask_for_bp_grid_item = batch_mask[i] if batch_mask is not None else torch.zeros((1, image_h, image_w), dtype=torch.bool, device=device)
                
                if current_mask_for_bp_grid_item.ndim == 3 and current_mask_for_bp_grid_item.shape[0] == 1:
                    current_mask_for_bp_grid_item_squeezed = current_mask_for_bp_grid_item.squeeze(0)
                elif current_mask_for_bp_grid_item.ndim == 2: # Already squeezed
                    current_mask_for_bp_grid_item_squeezed = current_mask_for_bp_grid_item
                else:
                    # Fallback or error for unexpected mask shape
                    warnings.warn(f"Unexpected mask shape for BP grid: {current_mask_for_bp_grid_item.shape}. Defaulting to no clues.")
                    current_mask_for_bp_grid_item_squeezed = torch.zeros((image_h, image_w), dtype=torch.bool, device=device)

                initial_bp_grid_np = np.zeros((grid_h, grid_w), dtype=int)
                unfilled_cells_coords: list[tuple[int,int]] = []
                for r in range(grid_h):
                    for c_in_grid in range(grid_w):
                        is_clue = False
                        # Only try to determine is_clue from mask if a valid mask was processed
                        if batch_mask is not None: # Check if original batch_mask was present
                            # Calculate center pixel of the patch for cell (r, c_in_grid)
                            pixel_y_to_check = r * patch_pixel_h + (patch_pixel_h // 2)
                            pixel_x_to_check = c_in_grid * patch_pixel_w + (patch_pixel_w // 2)
                            
                            # Ensure indices are within bounds of the pixel mask
                            pixel_y_to_check = min(max(0, pixel_y_to_check), image_h - 1)
                            pixel_x_to_check = min(max(0, pixel_x_to_check), image_w - 1)

                            # Assuming the mask (float or bool) is non-zero (or True) for clues
                            is_clue = current_mask_for_bp_grid_item_squeezed[pixel_y_to_check, pixel_x_to_check].item() == 0.0

                        if is_clue:
                            # all_cell_digits_pred is (1, grid_h, grid_w)
                            digit_val = all_cell_digits_pred[0, r, c_in_grid].item()
                            initial_bp_grid_np[r, c_in_grid] = digit_val if digit_val > 0 else 0
                        else:
                            initial_bp_grid_np[r, c_in_grid] = 0
                            unfilled_cells_coords.append((r, c_in_grid))
                
                # 3b. Run Belief Propagation for puzzle i
                bp_probabilities = self._compute_sudoku_probabilities_bp(
                    initial_bp_grid_np,
                    iterations=self.cfg.bp_iterations,
                    damping=self.cfg.bp_damping
                )

                current_input_image_cpu = batch_image[i].cpu()
                current_target_image_cpu = None
                if batch_target_image is not None and i < batch_target_image.shape[0]:
                    current_target_image_cpu = batch_target_image[i].cpu()

                if bp_probabilities is None:
                    warnings.warn(f"BP failed for puzzle item {i}, sampler {sampler_key}. Skipping this item for this sampler.")
                    yield EvaluationOutput(
                        key=sampler_key,
                        input_image=current_input_image_cpu,
                        prediction_image=None,
                        target_image=current_target_image_cpu,
                        metrics={"error": "BP_failed_for_puzzle"},
                        metadata={
                            "initial_bp_grid": initial_bp_grid_np.tolist(), 
                            "sampler_key": sampler_key,
                            "puzzle_index_in_batch": i,
                            "original_dataset_index": batch_index[i].item()
                        }
                    )
                    continue

                # 3d. Calculate Empirical Model Marginals for puzzle i
                sampled_digit_grids_torch = self.discretize(mnist_classifier, sampled_images_tensor_for_puzzle_i)
                sampled_digit_grids_np = sampled_digit_grids_torch.cpu().numpy()
                model_marginals_np = np.zeros((9, 9, 9), dtype=float)
                for r_idx in range(9):
                    for c_idx in range(9):
                        for digit_value_1_to_9 in range(1, 10):
                            count = np.sum(sampled_digit_grids_np[:, r_idx, c_idx] == digit_value_1_to_9)
                            model_marginals_np[r_idx, c_idx, digit_value_1_to_9 - 1] = count
                        cell_sum = np.sum(model_marginals_np[r_idx, c_idx, :])
                        if cell_sum > 0:
                            model_marginals_np[r_idx, c_idx, :] /= cell_sum
                        else:
                            model_marginals_np[r_idx, c_idx, :] = 1.0 / 9.0

                # 3e. Compare Distributions (KL Divergence) for unfilled cells for puzzle i
                mean_kl = np.nan
                kl_divergences_list = []
                if not unfilled_cells_coords:
                    warnings.warn(f"Puzzle item {i}, sampler {sampler_key}: no unfilled cells. KL not applicable.")
                else:
                    for r_coord, c_coord in unfilled_cells_coords:
                        bp_probs_cell = torch.from_numpy(bp_probabilities[r_coord, c_coord, :]).float().to(device)
                        model_probs_cell = torch.from_numpy(model_marginals_np[r_coord, c_coord, :]).float().to(device)
                        model_probs_cell = torch.clamp(model_probs_cell, min=1e-9)
                        model_probs_cell = model_probs_cell / torch.sum(model_probs_cell)
                        model_log_probs_cell = torch.log(model_probs_cell)
                        kl_div = F.kl_div(
                            model_log_probs_cell.unsqueeze(0),
                            bp_probs_cell.unsqueeze(0),
                            reduction='sum',
                            log_target=False
                        )
                        kl_divergences_list.append(kl_div.item())
                    if kl_divergences_list:
                        mean_kl = np.mean(kl_divergences_list)

                # 3f. Yield EvaluationOutput for puzzle i
                pred_image_to_display = sampled_images_tensor_for_puzzle_i[0].cpu() \
                    if sampled_images_tensor_for_puzzle_i.shape[0] > 0 else None
                
                current_metrics = {}
                if np.isnan(mean_kl) and unfilled_cells_coords:
                    current_metrics = {"error": "KL_divergence_calculation_failed"}
                elif not unfilled_cells_coords:
                    current_metrics = {"mean_kl_divergence_unfilled": 0.0, "info": "no_unfilled_cells"}
                else:
                    current_metrics = {"mean_kl_divergence_unfilled": mean_kl}

                yield EvaluationOutput(
                    key=sampler_key, 
                    input_image=current_input_image_cpu,
                    prediction_image=pred_image_to_display,
                    target_image=current_target_image_cpu,
                    metrics=current_metrics,
                    metadata={
                        "initial_bp_grid": initial_bp_grid_np.tolist(),
                        "unfilled_cells_count": len(unfilled_cells_coords),
                        "num_model_samples_used": self.cfg.num_model_samples,
                        "sampler_key": sampler_key, 
                        "puzzle_index_in_batch": i,
                        "original_dataset_index": batch_index[i].item()
                    }
                )