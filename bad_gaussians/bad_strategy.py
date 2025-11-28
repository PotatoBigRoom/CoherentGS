from dataclasses import dataclass

import torch

from typing import Any, Dict, Union
from gsplat.strategy import DefaultStrategy


@dataclass
class BadDefaultStrategy(DefaultStrategy):
    """
    Modified default strategy in BAD-Gaussians that only use the gradients of the
    middle virtual-sharp view to update the running state.
    """

    num_virtual_views: int = 10

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        mid_view_id = self.num_virtual_views // 2
        for key in ["means2d", "width", "height", "n_cameras", "radii", "gaussian_ids"]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info["means2d"].absgrad[mid_view_id].clone()
        else:
            grads = info["means2d"].grad[mid_view_id].clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            print("BAD-Gaussians: packed=True may decrease performance")
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"][mid_view_id] > 0.0  # [1, N]
            gs_ids = torch.where(sel)[0]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][mid_view_id][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )
