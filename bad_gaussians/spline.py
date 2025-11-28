"""
SE(3) B-spline trajectory

Created by lzzhao on 2023.09.29
"""

from __future__ import annotations

from typing import Optional, Tuple, Type

import pypose as pp
import torch
from dataclasses import dataclass, field
from jaxtyping import Bool, Float
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never

from bad_gaussians.base_config import InstantiateConfig
from bad_gaussians.spline_functor import linear_interpolation, cubic_bspline_interpolation

DEBUG_PRINT_PRECISION = 10


@dataclass
class SplineConfig(InstantiateConfig):
    """Configuration for spline instantiation."""

    _target: Type = field(default_factory=lambda: Spline)
    """Target class to instantiate."""

    degree: int = 1
    """Degree of the spline. 1 for linear spline, 3 for cubic spline."""

    sampling_interval: float = 0.1
    """Sampling interval of the control knots."""

    start_time: float = 0.0
    """Starting timestamp of the spline."""


class Spline(nn.Module):
    """SE(3) spline trajectory.

    Args:
        config: the SplineConfig used to instantiate class
        max_knots: the maximum number of control knots
        device: cpu or cuda
    """

    config: SplineConfig
    data: Float[LieTensor, "num_knots 7"]
    start_time: float
    end_time: float
    t_lower_bound: float
    t_upper_bound: float

    def __init__(self, config: SplineConfig, max_knots: Optional[int] = 1000, device: Optional[str] = "cpu"):
        super().__init__()
        self.config = config
        self.max_knots = max_knots
        self.device = device
        self.data = pp.identity_SE3(0)
        self.order = self.config.degree + 1
        """Order of the spline, i.e. control knots per segment, 2 for linear, 4 for cubic"""

        self.set_start_time(config.start_time)
        self.update_end_time()

        self.spline_optimizer = self.config.spline_optimizer.setup(num_cameras=max_knots, device=self.device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Float[LieTensor, "1 7"]:
        return self.data[index]

    def forward(self, timestamps: Float[Tensor, "*batch_size"]) -> Float[LieTensor, "*batch_size 7"]:
        """Interpolate the spline at the given timestamps.

        Args:
            timestamps: Timestamps to interpolate the spline at. Range: [t_lower_bound, t_upper_bound].

        Returns:
            poses: The interpolated pose.
        """
        segment, u = self.get_segment(timestamps)
        u = u[..., None]  # (*batch_size) to (*batch_size, interpolations=1)
        if self.config.degree == 1:
            poses = linear_interpolation(segment, u)
        elif self.config.degree == 3:
            poses = cubic_bspline_interpolation(segment, u)
        else:
            assert_never(self.config.degree)
        return poses.squeeze()

    def check_timestamps(
        self, timestamps: Float[Tensor, "*batch_size"]
    ) -> Tuple[bool, Optional[Bool[Tensor, "*batch_size"]]]:
        """Check if the timestamps are within the valid range.

        Args:
            timestamps: Timestamps to check.

        Returns:
            valid: Whether the timestamps are within the valid range.
            out_of_bound: The timestamps that are out of bound.
        """
        out_of_bound = torch.logical_or(timestamps < self.t_lower_bound, timestamps > self.t_upper_bound)
        return not out_of_bound.any(), out_of_bound

    def get_segment(
        self, timestamps: Float[Tensor, "*batch_size"], check_timestamps: bool = True, eps: float = 1e-4
    ) -> Tuple[Float[LieTensor, "*batch_size self.order 7"], Float[Tensor, "*batch_size"]]:
        """Get the spline segment and normalized position on segment at the given timestamp.

        Args:
            timestamps: Timestamps to get the spline segment and normalized position at.
            check_timestamps: Whether to check if the timestamps are within the valid range.
            eps: A small epsilon to avoid possible numerical issues.

        Returns:
            segment: The spline segment.
            u: The normalized position on the segment.
        """
        timestamps = torch.clip(timestamps, self.t_lower_bound + eps, self.t_upper_bound - eps)
        if check_timestamps:
            valid, out_of_bound = self.check_timestamps(timestamps)
            try:
                assert valid
            except AssertionError as e:
                torch.set_printoptions(precision=DEBUG_PRINT_PRECISION)
                raise ValueError(
                    f"Timestamps {timestamps[out_of_bound]} are out of bound: "
                    f"({self.t_lower_bound}, {self.t_upper_bound})"
                ) from e
        batch_size = timestamps.shape
        relative_time = timestamps - self.start_time
        normalized_time = relative_time / self.config.sampling_interval
        start_index = torch.floor(normalized_time).int()

        start_index = start_index.to(self.device)
        normalized_time = normalized_time.to(self.device)

        u = normalized_time - start_index
        if self.config.degree == 3:
            start_index -= 1

        indices = start_index.tile((self.order, 1)).T + torch.arange(self.order).tile((*batch_size, 1)).to(
            start_index.device
        )

        # Check if indices are within the valid range.
        # This should always be true with check_timestamps=True and eps=1e-4.
        assert torch.logical_and(indices >= 0, indices < len(self)).all()
        segment = pp.SE3(torch.gather(self.data.expand(*batch_size, -1, -1), dim=1, index=indices[..., None].tile(7)))

        if self.config.spline_optimizer.mode != "off":
            pose_opt = self.spline_optimizer(indices.flatten()).view(*batch_size, self.order, 7)
            segment = segment @ pose_opt

        return segment, u

    def insert(self, pose: Float[LieTensor, "1 7"]):
        """Insert a control knot"""
        self.data = pp.SE3(torch.cat([self.data, pose]))
        self.update_end_time()

    def set_data(self, data: Float[LieTensor, "num_knots 7"] | pp.Parameter):
        """Set the spline data."""
        self.data = data
        self.update_end_time()

    def set_start_time(self, start_time: float):
        """Set the starting timestamp of the spline."""
        if isinstance(start_time, torch.Tensor):
            start_time = start_time.item()
        self.start_time = start_time
        if self.config.degree == 1:
            self.t_lower_bound = self.start_time
        elif self.config.degree == 3:
            self.t_lower_bound = self.start_time + self.config.sampling_interval
        else:
            assert_never(self.config.degree)

    def update_end_time(self):
        """Update the ending timestamp of the spline."""
        if isinstance(self.config.sampling_interval, torch.Tensor):
            self.config.sampling_interval = self.config.sampling_interval.item()
        self.end_time = self.start_time + self.config.sampling_interval * (len(self) - 1)
        if self.config.degree == 1:
            self.t_upper_bound = self.end_time
        elif self.config.degree == 3:
            self.t_upper_bound = self.end_time - self.config.sampling_interval
        else:
            assert_never(self.config.degree)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        pass

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Add spline metrics"""
        self.spline_optimizer.get_metrics_dict(metrics_dict)

    def get_param_groups(self, param_groups: dict) -> None:
        """Get spline parameters"""
        self.spline_optimizer.get_param_groups(param_groups)
