"""
Bundle Adjusted Deblur Camera Optimizer
"""

from __future__ import annotations

import functools
from typing import Literal, Type, Union

import pypose as pp
import torch
from dataclasses import dataclass, field
from jaxtyping import Float, Int
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never

from bad_gaussians.base_config import InstantiateConfig
from bad_gaussians.spline_functor import (
    bezier_interpolation,
    cubic_bspline_interpolation,
    linear_interpolation,
    linear_interpolation_mid,
)

TrajSamplingMode = Literal["uniform", "start", "mid", "end"]
"""How to sample the camera trajectory"""


@dataclass
class BadCameraOptimizerConfig(InstantiateConfig):
    """Configuration of BAD-Gaussians camera optimizer."""

    _target: Type = field(default_factory=lambda: BadCameraOptimizer)
    """The target class to be instantiated."""

    mode: Literal["off", "linear", "cubic", "bezier"] = "linear"
    """Pose optimization strategy to use.
    linear: linear interpolation on SE(3);
    cubic: cubic b-spline interpolation on SE(3).
    bezier: Bezier curve interpolation on SE(3).
    """

    bezier_degree: int = 9
    """Degree of the Bezier curve. Only used when mode is bezier."""

    trans_l2_penalty: float = 0.0
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 0.0
    """L2 penalty on rotation parameters."""

    num_virtual_views: int = 10
    """The number of samples used to model the motion-blurring."""

    initial_noise_se3_std: float = 4e-3
    """Initial perturbation to pose delta on se(3). Must be non-zero to prevent NaNs."""


class BadCameraOptimizer(nn.Module):
    """Optimization for BAD-Gaussians virtual camera trajectories."""

    config: BadCameraOptimizerConfig
    pose_adjustment: Float[LieTensor, "num_cameras num_control_knots dof"]

    def __init__(
        self,
        config: BadCameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.dof = 6
        """Degrees of freedom of manifold, i.e. number of dimensions of the tangent space"""
        self.dim = 7
        """Dimentions of pose parameterization. Three for translation, 4-tuple for quaternion"""

        # Initialize learnable parameters.
        if self.config.mode == "off":
            return
        elif self.config.mode == "linear":
            self.num_control_knots = 2
        elif self.config.mode == "cubic":
            self.num_control_knots = 4
        elif self.config.mode == "bezier":
            self.num_control_knots = self.config.bezier_degree
        else:
            assert_never(self.config.mode)

        self.pose_adjustment = pp.Parameter(
            pp.randn_se3(
                (num_cameras, self.num_control_knots),
                sigma=self.config.initial_noise_se3_std,
                device=device,
            ),
        )

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
        mode: TrajSamplingMode = "mid",
    ) -> Float[LieTensor, "camera_indices self.num_control_knots self.dof"]:
        """Indexing into camera adjustments.

        Args:
            indices: indices of Cameras to optimize.
            mode: interpolate between start and end, or return start / mid / end.

        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        else:
            indices = indices.int()
            unique_indices, lut = torch.unique(indices, return_inverse=True)
            # assert torch.logical_and(unique_indices >= 0, unique_indices < self.num_cameras).all()
            camera_opt = self.pose_adjustment[unique_indices].Exp()
            outputs.append(self._interpolate(camera_opt, mode)[lut])

        # Return: identity if no transforms are needed, otherwise composite transforms together.
        if len(outputs) == 0:
            return pp.identity_SE3(*indices.shape, device=self.device)
        return functools.reduce(pp.mul, outputs)

    def _interpolate(
        self,
        camera_opt: Float[LieTensor, "*batch_size self.num_control_knots self.dof"],
        sampling_mode: TrajSamplingMode,
    ) -> Float[Tensor, "*batch_size interpolations self.dof"]:
        if sampling_mode == "uniform":
            u = torch.linspace(
                start=0,
                end=1,
                steps=self.config.num_virtual_views,
                device=camera_opt.device,
            )
            if self.config.mode == "linear":
                return linear_interpolation(camera_opt, u)
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(camera_opt, u)
            elif self.config.mode == "bezier":
                return bezier_interpolation(camera_opt, u)
            else:
                assert_never(self.config.mode)
        elif sampling_mode == "mid":
            if self.config.mode == "linear":
                return linear_interpolation_mid(camera_opt)
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(camera_opt, torch.tensor([0.5], device=camera_opt.device)).squeeze(1)
            elif self.config.mode == "bezier":
                return bezier_interpolation(camera_opt, torch.tensor([0.5], device=camera_opt.device)).squeeze(1)
            else:
                assert_never(self.config.mode)
        elif sampling_mode == "start":
            if self.config.mode == "linear":
                return camera_opt[..., 0, :]
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(camera_opt, torch.tensor([0.0], device=camera_opt.device)).squeeze(1)
            elif self.config.mode == "bezier":
                return bezier_interpolation(camera_opt, torch.tensor([0.0], device=camera_opt.device)).squeeze(1)
            else:
                assert_never(self.config.mode)
        elif sampling_mode == "end":
            if self.config.mode == "linear":
                return camera_opt[..., 1, :]
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(camera_opt, torch.tensor([1.0], device=camera_opt.device)).squeeze(1)
            elif self.config.mode == "bezier":
                return bezier_interpolation(camera_opt, torch.tensor([1.0], device=camera_opt.device)).squeeze(1)
            else:
                assert_never(self.config.mode)
        else:
            assert_never(sampling_mode)

    def apply_to_cameras(
        self,
        c2w: Float[Tensor, "batch_size 4 4"],
        camera_ids: Int[Tensor, "batch_size"],
        mode: TrajSamplingMode = "mid",
    ) -> Float[Tensor, "batch_size (num_interpolations) 4 4"]:
        """Apply pose correction to the camera to world matrices."""
        if self.config.mode == "off":
            return c2w

        if c2w.shape[1] == 3:
            c2w = torch.cat([c2w, torch.tensor([0, 0, 0, 1], device=c2w.device).view(1, 1, 4)], dim=1)

        poses_delta = self((torch.tensor(camera_ids)), mode)

        if mode == "uniform":
            # c2w: (..., 4, 4), poses_delta: (..., num_virtual_views, 4, 4)
            c2ws = c2w.unsqueeze(1).expand(-1, self.config.num_virtual_views, -1, -1)
            c2ws_adjusted = c2ws @ poses_delta.matrix().squeeze()
            return c2ws_adjusted  # (..., num_virtual_views, 4, 4)
        else:
            c2w_adjusted = c2w @ poses_delta.matrix()
            return c2w_adjusted  # (..., 4, 4)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        pass

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            metrics_dict["camera_opt_trajectory_translation"] = (
                self.pose_adjustment[:, 1, :3] - self.pose_adjustment[:, 0, :3]
            ).norm()
            metrics_dict["camera_opt_trajectory_rotation"] = (
                self.pose_adjustment[:, 1, 3:] - self.pose_adjustment[:, 0, 3:]
            ).norm()
            metrics_dict["camera_opt_translation"] = 0
            metrics_dict["camera_opt_rotation"] = 0
            for i in range(self.num_control_knots):
                metrics_dict["camera_opt_translation"] += self.pose_adjustment[:, i, :3].norm()
                metrics_dict["camera_opt_rotation"] += self.pose_adjustment[:, i, 3:].norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0
