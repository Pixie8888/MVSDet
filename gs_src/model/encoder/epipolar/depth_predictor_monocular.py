import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .conversions import relative_disparity_to_depth
from .distribution_sampler import DistributionSampler

from typing import List, Dict, Set, Tuple
from typing import Union

class DepthPredictorMonocular(nn.Module):
    projection: nn.Sequential
    sampler: DistributionSampler
    num_samples: int
    num_surfaces: int

    def __init__(
        self,
        d_in: int, # 128
        num_samples: int, # 32
        num_surfaces: int, # 1
        use_transmittance: bool, # false
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_in, 2 * num_samples * num_surfaces),
        )
        self.sampler = DistributionSampler()
        self.num_samples = num_samples
        self.num_surfaces = num_surfaces
        self.use_transmittance = use_transmittance

        # This exists for hooks to latch onto.
        self.to_pdf = nn.Softmax(dim=-1)
        self.to_offset = nn.Sigmoid()

    def forward(
        self,
        features: Float[Tensor, "batch view ray channel"], # (bs, num_view, h*w, 128)
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        deterministic: bool,
        gaussians_per_pixel: int,
    ) -> Tuple[
        Float[Tensor, "batch view ray surface sample"],  # depth
        Float[Tensor, "batch view ray surface sample"],  # pdf
    ]:
        s = self.num_samples # 32. depth_bucket

        # Convert the features into a depth distribution plus intra-bucket offsets.
        # print(features.shape)
        features = self.projection(features)
        pdf_raw, offset_raw = rearrange(
            features, "... (dpt srf c) -> c ... srf dpt", c=2, srf=self.num_surfaces
        )
        pdf = self.to_pdf(pdf_raw) # (b, v, h*w, num_surface, num_samples)
        offset = self.to_offset(offset_raw) # (b, v, h*w, num_surface, num_samples)

        # Sample from the depth distribution.
        index, pdf_i = self.sampler.sample(pdf, deterministic, gaussians_per_pixel) # (b,v,h*w, num_surface, num_gs)
        offset = self.sampler.gather(index, offset) # (b,v,h*w, num_surface, num_gs)

        # Convert the sampled bucket and offset to a depth.
        relative_disparity = (index + offset) / s
        depth = relative_disparity_to_depth(
            relative_disparity,
            rearrange(near, "b v -> b v () () ()"),
            rearrange(far, "b v -> b v () () ()"),
        )

        # Compute opacity from PDF.
        if self.use_transmittance:
            partial = pdf.cumsum(dim=-1)
            partial = torch.cat(
                (torch.zeros_like(partial[..., :1]), partial[..., :-1]), dim=-1
            )
            opacity = pdf / (1 - partial + 1e-10)
            opacity = self.sampler.gather(index, opacity)
        else:
            opacity = pdf_i

        return depth, opacity
