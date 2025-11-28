from typing import Dict, Literal

import numpy as np
import torch
import viser
import viser.transforms as vtf
from nerfview.viewer import Viewer

from datasets.colmap import Dataset

# not used for now
VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


class PoseViewer(Viewer):

    def init_scene(self, train_dataset: Dataset, train_state: Literal["training", "paused", "completed"]) -> None:
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}

        total_num = len(train_dataset)
        # NOTE: not constraining the maximum number of camera frustums shown
        image_indices = np.linspace(0, total_num - 1, total_num, dtype=np.int32).tolist()
        for idx in image_indices:
            image = train_dataset[idx]["image"]
            image_uint8 = image.detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)

            # torchvision can be slow to import, so we do it lazily.
            import torchvision

            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()

            c2w = train_dataset[idx]["camtoworld"].cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            # NOTE: not understand why this is needed in nerfstudio viewer, but comment it out make ours work
            # probably because gsplat uses OpenCV convention, whereas nerfstudio use the Blender / OpenGL convention
            # R = R @ vtf.SO3.from_x_radians(np.pi)

            K = train_dataset[idx]["K"].cpu().numpy()
            fx = K[0, 0]
            cx = K[0, -1]
            cy = K[1, -1]
            camera_handle = self.server.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.05,  # hardcode this scale for now
                aspect=float(cx / cy),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                # NOTE: not multiplied by VISER_NERFSTUDIO_SCALE_RATIO, this should also be used in get_camera_state
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.state.status = train_state
        # self.train_util = 0.9

    def update_camera_poses(self, camtoworlds):
        assert self.camera_handles is not None
        # TODO: update poses of camera

        idxs = list(self.camera_handles.keys())

        with torch.no_grad():
            c2ws = camtoworlds

        for i, key in enumerate(idxs):
            c2w = c2ws[i].detach().cpu().numpy()

            R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
            # R = R @ vtf.SO3.from_x_radians(np.pi)

            self.camera_handles[key].position = c2w[:3, 3]
            self.camera_handles[key].wxyz = R.wxyz