import os
from typing import Any, Dict, List, Optional

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

from .colmap_dataparser import ColmapParser


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Dataset:
    """A simple dataset class.

    支持通过传入的 `train_indices` 或 `parser.train_indices` 指定训练集索引。
    优先级：参数 `train_indices` > 解析器 `parser.train_indices` > 默认规则（排除测试帧）。
    """

    def __init__(
        self,
        parser: ColmapParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        train_indices: Optional[List[int]] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.debug_loaded = False  # 添加调试标志
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            # 优先使用传入的 train_indices，其次使用 parser.train_indices
            cfg_train_indices = train_indices if train_indices is not None else getattr(self.parser, "train_indices", None)
            if cfg_train_indices is not None and len(cfg_train_indices) > 0:
                # 校验并应用配置的训练索引
                idx_arr = np.array(cfg_train_indices, dtype=int)
                if np.any(idx_arr < 0) or np.any(idx_arr >= len(indices)):
                    raise ValueError(f"训练索引越界: {cfg_train_indices}, 有效范围是 [0, {len(indices)-1}]")
                self.indices = idx_arr
                print(f"[Dataset] 使用配置的训练索引: {self.indices.tolist()}")
            else:
                # 默认训练集为非测试帧
                if self.parser.test_every > 1:
                    self.indices = indices[indices % self.parser.test_every != 0]
                else:
                    self.indices = indices
        elif split == "all":
            self.indices = indices
        else:
            if self.parser.test_every > 1:
                self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        
        # 只在第一次加载时显示详细的pose调试信息
        if not self.debug_loaded:
            print(f"\n[DEBUG] ===== 数据加载调试信息 (仅显示一次) =====")
            print(f"[DEBUG] 数据集内部索引 item: {item}")
            print(f"[DEBUG] 映射到COLMAP索引 index: {index}")
            print(f"[DEBUG] 图像路径: {self.parser.image_paths[index]}")
            print(f"[DEBUG] 图像名称: {self.parser.image_names[index]}")
            print(f"[DEBUG] 相机ID: {self.parser.camera_ids[index]}")
            
            # 显示所有训练集索引对应的信息
            print(f"[DEBUG] 训练集所有索引对应的信息:")
            for i, idx in enumerate(self.indices):
                print(f"[DEBUG]   训练集索引{i} -> COLMAP索引{idx} -> 图像名称: {self.parser.image_names[idx]}")
            
            self.debug_loaded = True  # 设置标志，避免重复显示
        
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        
        # 只在第一次加载时打印pose详细信息
        if not hasattr(self, '_pose_debug_printed'):
            print(f"[DEBUG] 加载的pose矩阵 (COLMAP索引{index}):")
            print(f"[DEBUG] 完整pose矩阵:")
            for i, row in enumerate(camtoworlds):
                print(f"[DEBUG]   [{i}] {row}")
            
            # 分析pose的有效性
            R = camtoworlds[:3, :3]
            t = camtoworlds[:3, 3]
            det_R = np.linalg.det(R)
            is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            
            print(f"[DEBUG] 相机位置 (平移向量): {t}")
            print(f"[DEBUG] 旋转矩阵行列式: {det_R:.8f}")
            print(f"[DEBUG] 旋转矩阵是否正交: {is_orthogonal}")
            print(f"[DEBUG] pose是否有效: {abs(det_R - 1.0) < 1e-6 and is_orthogonal}")
            
            # 打印相机内参
            print(f"[DEBUG] 相机内参K矩阵:")
            for i, row in enumerate(K):
                print(f"[DEBUG]   [{i}] {row}")
            print(f"[DEBUG] ================================\n")
            
            self._pose_debug_printed = True  # 设置标志，避免重复显示

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "colmap_image_id": index,  # the index of the image in the colmap data
        }

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = ColmapParser(data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8)
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
