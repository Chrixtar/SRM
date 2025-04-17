from typing import Sequence, Union

from jaxtyping import Float
from torch import Tensor

from .sampler import Sampler
from .fixed_sampler import FixedSampler, FixedSamplerCfg
from .manual_sampler import ManualSampler, ManualSamplerCfg
from .sequential_adaptive_sampler import SequentialAdaptiveSampler, SequentialAdaptiveSamplerCfg
from .simultaneous_adaptive_sampler import SimultaneousAdaptiveSampler, SimultaneousAdaptiveSamplerCfg

SAMPLER = {
    "fixed": FixedSampler,
    "manual": ManualSampler,
    "sequential_adaptive": SequentialAdaptiveSampler,
    "simultaneous_adaptive": SimultaneousAdaptiveSampler
}


SamplerCfg = Union[
    FixedSamplerCfg,
    ManualSamplerCfg,
    SequentialAdaptiveSamplerCfg,
    SimultaneousAdaptiveSamplerCfg
]


def get_sampler(
    cfg: SamplerCfg,
    patch_size: int | None = None,
    patch_grid_shape: Sequence[int] | None = None,
    dependency_matrix: Float[Tensor, "num_patches num_patches"] | None = None
) -> Sampler:
    return SAMPLER[cfg.name](cfg, patch_size, patch_grid_shape, dependency_matrix)
