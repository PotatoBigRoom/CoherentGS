

import json
import math
import os
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

# Import SE(3) interpolation modules
import sys
from typing_extensions import assert_never

from hybrid_sampling import generate_camera_trajectory,se3_interpolate_to_target
from scoring_model import VirtualViewQualityScorer
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from dataclasses import dataclass, field
from pytorch_msssim import ssim as pytorch_ssim
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfview.viewer import Viewer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from bad_gaussians.bad_camera_optimizer import BadCameraOptimizer, BadCameraOptimizerConfig
from datasets.blender_dataperser import BlenderParser
from datasets.colmap import Dataset
from datasets.colmap_dataparser import ColmapParser
from datasets.deblur_nerf import DeblurNerfDataset
from pose_viewer import PoseViewer
from simple_trainer import Config, Runner, create_splats_with_optimizers
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from utils import (
    AppearanceOptModule,
    CameraOptModuleSE3,
    set_random_seed,
)

from pection_loss import VGG16PerceptualLoss, VGG16PerceptualLossWithMultipleLayers, VGG16DISTSLoss

from PIL import Image
try:
    from pipeline_difix import DifixPipeline
    DIFIX3D_AVAILABLE = True
except Exception as e:
    DIFIX3D_AVAILABLE = False
    

@dataclass
class DeblurDiFix3DConfig(Config):
    
    # Data settings
    data_dir: str = "path_of_your_data"
    data_factor: int = 1
    # Optional: specify training image IDs; None uses all default training views
    train_indices: Optional[List[int]] = None
    
    # Evaluation settings
    eval_only: bool = False
    eval_steps: List[int] = field(default_factory=lambda: [3_000, 7_000])
    scale_factor: float = 1.0
    result_dir: str = "path_of_your_result"
    test_every: int = 8

    ########### Viewer ###############
    disable_viewer: bool = False
    port: int = 8080
    visualize_cameras: bool = True

    ########### Training ###############
    max_steps: int = 30000
    eval_steps: List[int] = field(default_factory=lambda: [3_000, 7_000])
    save_steps: List[int] = field(default_factory=lambda: [3_000, 7_000])
    
    # Use fused SSIM optimization
    fused_ssim: bool = False
    pin_memory: bool = False
    
    # Save settings
    save_only_recent_train: bool = False
    
    # Batch size for training
    batch_size: int = 1
    steps_scaler: float = 1.0
    
    ########### Gaussian Initialization ###############
    init_type: str = "sfm"  # "sfm" or "random"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0
    global_scale: float = 1.0
    
    ########### Spherical Harmonics ###############
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    
    ########### Loss ###############
    loss_rgb_lambda: float = 0.8  # L1 loss weight
    loss_ssim_lambda: float = 0.2  # SSIM loss weight
    ssim_lambda: float = 0.2  # for compatibility
    
    ########### Rendering ###############
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    
    ########### Strategy ###############
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)

    ########### Background ###############
    random_bkgd: bool = True

    ########### Motion Deblur (BAD-Gaussians) ###############
    camera_optimizer: BadCameraOptimizerConfig = field(
        default_factory=lambda: BadCameraOptimizerConfig(
            mode="linear",
            num_virtual_views=10,  
        )
    )

    ########### DiFix3D Integration ###############
    enable_difix3d: bool = True
    
    difix3d_model_name: str = "nvidia/difix_ref"
    
    difix3d_prompt: str = "remove degradation"

    difix3d_blend_ratio: float = 1.0

    
    difix3d_num_inference_steps: int = 1

    difix3d_guidance_scale: float = 0.0

    difix3d_use_ref_image: bool = True

    difix3d_augment_training_set: bool = True
    
    difix3d_max_augmented_samples: int = 100

    difix3d_save_comparisons: bool = True

    virtual_view_start_step: int = 25000

    virtual_view_interval: int = 250

    
    virtual_view_poses_per_step: int = 2

    
    virtual_view_loss_weight: float = 0.1


    interp_quality_psnr_min: float = 4.5
    interp_quality_psnr_max: float = 14.5
    
    ########### Camera Opt ###############
    pose_opt: bool = True
    pose_opt_lr: float = 5e-3
    pose_opt_reg: float = 1e-6
    pose_opt_lr_decay: float = 1e-2
    pose_noise: float = 1e-2
    pose_gradient_accumulation_steps: int = 10

    ########### Appearance Opt ###############
    app_opt: bool = False
    app_embed_dim: int = 32
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 0.0

    ########### Bilateral Grid ###############
    use_bilateral_grid: bool = False
    bilateral_grid_shape: List[int] = field(default_factory=lambda: [16, 16, 8])
    
    ########### Novel View Eval ###############
    nvs_eval_enable_during_training: bool = True
    nvs_steps: int = 200
    nvs_steps_final: int = 1000
    nvs_pose_lr: float = 1e-3
    nvs_pose_reg: float = 0.0
    nvs_pose_lr_decay: float = 1e-2
    
    ########### Deblurring Eval ###############
    deblur_eval_enable_during_training: bool = False
    deblur_eval_enable_pose_opt: bool = False
    
    ########### Regularizations ###############
    enable_phys_scale_reg: bool = False
    max_gauss_ratio: float = 10.0
    enable_mcmc_opacity_reg: bool = False
    enable_mcmc_scale_reg: bool = True
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    
    ########### Depth Smooth Loss ###############
    enable_depth_smooth_loss: bool = True
    depth_smooth_lambda: float = 0.1
    
    ########### DiFix Enhancement Loss ###############
    enable_difix_enhancement_loss: bool = True
    difix_enhancement_loss_weight: float = 0.05
    difix_enhancement_l1_weight: float = 0.8
    difix_enhancement_perceptual_weight: float = 0.2

    
    # Avoid multiple initialization
    bad_gaussians_post_init_complete: bool = False

    def __post_init__(self):
        if not self.bad_gaussians_post_init_complete:
            self.bad_gaussians_post_init_complete = True
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.result_dir = Path(self.result_dir) / timestr
            if isinstance(self.strategy, DefaultStrategy):
                self.strategy.grow_grad2d = self.strategy.grow_grad2d / self.camera_optimizer.num_virtual_views
                self.strategy.reset_every = 999999999


def depth_smooth_loss_4neighbor(depth_map: torch.Tensor) -> torch.Tensor:
    # Ensure input format is [B, H, W]
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(-1)  # [B, H, W, 1] -> [B, H, W]
    
    if depth_map.dim() != 3:
        raise ValueError(f"Depth map must be 3D [B, H, W], got: {depth_map.shape}")
    
    batch_size, height, width = depth_map.shape
    
    # Compute horizontal and vertical differences
    # Horizontal diff: depth[i, j] - depth[i, j-1] (except left boundary)
    diff_h = depth_map[:, :, 1:] - depth_map[:, :, :-1]  # [B, H, W-1]
    
    # Vertical diff: depth[i, j] - depth[i-1, j] (except top boundary)
    diff_v = depth_map[:, 1:, :] - depth_map[:, :-1, :]  # [B, H-1, W]
    
    # L2 loss
    smooth_loss_h = torch.mean(diff_h ** 2)  # horizontal smoothness
    smooth_loss_v = torch.mean(diff_v ** 2)  # vertical smoothness
    
    # Total smoothness loss
    total_smooth_loss = smooth_loss_h + smooth_loss_v
    
    return total_smooth_loss


class DiFix3DProcessor:
    
    def __init__(self, model_name: str = "nvidia/difix_ref", device: str = "cuda", ref_image_dir: str = None):
        self.device = device
        self.model_name = model_name
        self.pipeline = None
        self.enabled = DIFIX3D_AVAILABLE
        self.ref_image_dir = ref_image_dir  # ref image directory
        
        # Progressive interpolation state
        self.is_initialized = False
        self.quality_scorer = None
        self.available_interpolation_views = []
        self.training_psnr_mean = None
        self.training_psnr_variance = None
        
        # DiFix3D comparison output directory
        self.difix3d_comparison_dir = None
        
        # Store virtual view quality scores
        self.virtual_view_scores = []
        # Store baseline scores (training view PSNR baseline)
        self.baseline_scores = {}
        
        if self.enabled:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        try:            
            self.pipeline = DifixPipeline.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.pipeline.to(self.device)
            
        except Exception as e:
            self.enabled = False
    
    def _ensure_tensor_format(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:

            # Record original size
        if image_tensor.dim() == 4:  # [1, H, W, 3]
            original_height, original_width = image_tensor.shape[1:3]
            tensor = image_tensor.squeeze(0)  # [H, W, 3]
        elif image_tensor.dim() == 3:  # [H, W, 3]
            original_height, original_width = image_tensor.shape[:2]
            tensor = image_tensor
        else:
            raise ValueError(f"Unsupported tensor dims: {image_tensor.shape}")
        
        # Check channel count
        if tensor.shape[-1] != 3:
            raise ValueError(f"Unsupported channel count: {tensor.shape[-1]}, expected 3")
        
        return tensor, (original_height, original_width)
    
    def process_image(
        self, 
        image_tensor: torch.Tensor, 
        prompt: str = "remove degradation",
        num_inference_steps: int = 1,
        timesteps: List[int] = [199],
        guidance_scale: float = 0.0,
        ref_image: Optional[torch.Tensor] = None,
        save_comparison: bool = False,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        if not self.enabled or self.pipeline is None:
            return image_tensor
        
        try:
            with torch.no_grad():
                # Normalize input tensor and get original size
                input_tensor, original_size = self._ensure_tensor_format(image_tensor)
                
                
                # Ensure value range in [0,1]
                if input_tensor.max() > 1.0 or input_tensor.min() < 0.0:
                    input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
                    print(f"   ⚠️ Value range adjusted to [0,1]")
                
                # Convert to PIL image
                image_np = (input_tensor.cpu().numpy() * 255).astype(np.uint8)
                input_image = Image.fromarray(image_np)
                print(f"   PIL image size: {input_image.size}")  # (width, height)
                
                # Handle reference image if provided
                ref_image_pil = None
                if ref_image is not None:
                    ref_tensor, _ = self._ensure_tensor_format(ref_image)
                    # Ensure value range in [0,1]
                    if ref_tensor.max() > 1.0 or ref_tensor.min() < 0.0:
                        ref_tensor = torch.clamp(ref_tensor, 0.0, 1.0)
                    ref_np = (ref_tensor.cpu().numpy() * 255).astype(np.uint8)
                    ref_image_pil = Image.fromarray(ref_np)
                    print(f"   Reference image size: {ref_image_pil.size}")

                if ref_image_pil is not None:

                    
                    # Ensure input and reference sizes match exactly
                    if input_image.size != ref_image_pil.size:
                        print(f"   🔧 Resize reference image to match input: {ref_image_pil.size} -> {input_image.size}")
                        ref_image_pil = ref_image_pil.resize(input_image.size, Image.Resampling.LANCZOS)
                    
                    # Check image size for memory concerns
                    width, height = input_image.size
                    if width * height > 1000000:  # large images may cause OOM
                        print(f"   ⚠️ Large image size ({width}x{height}); may cause memory issues")
                    
                    # Use single image, avoid creating batch
                    try:
                        output_image = self.pipeline(
                            prompt,
                            image=input_image,
                            ref_image=ref_image_pil,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images[0]
                    except Exception as e:
                        # If single-image fails, try batch
                        input_images = [input_image, input_image]
                        ref_images = [ref_image_pil, ref_image_pil]
                        
                        output_images = self.pipeline(
                            prompt,
                            image=input_images,
                            ref_image=ref_images,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images
                        output_image = output_images[0]
                else:
                    print(f"   🚫 No reference image; run DiFix3D directly")
                    
                    try:
                        output_image = self.pipeline(
                            prompt,
                            image=input_image,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images[0]
                    except Exception as e:
                        # If single-image fails, try batch
                        input_images = [input_image, input_image]
                        
                        output_images = self.pipeline(
                            prompt,
                            image=input_images,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images
                        output_image = output_images[0]
                
                print(f"   DiFix3D output PIL size: {output_image.size}")  # (width, height)
                
                # Convert back to tensor
                output_np = np.array(output_image).astype(np.float32) / 255.0
                output_tensor = torch.from_numpy(output_np).to(image_tensor.device)
                
                print(f"   Converted tensor shape: {output_tensor.shape}")
                
                # Restore batch dimension if original had it
                if image_tensor.dim() == 4:
                    output_tensor = output_tensor.unsqueeze(0)  # [1, H, W, 3]
                
                print(f"✅ DiFix3D processing completed:")
                print(f"   Final output shape: {output_tensor.shape}")
                print(f"   Size change: {original_size} -> {output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]}")
                
                # Check if size changed
                final_size = output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]
                if final_size != original_size:
                    print(f"   ⚠️ Size changed: {original_size} -> {final_size}")
                
                # 保存处理前后对比图像
                return output_tensor
                
        except Exception as e:
            print(f"⚠️ DiFix3D processing failed: {e}")
            print(f"   Input tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"   Error details: {str(e)}")
            
            # Check for einops-related errors
            if "einops" in str(e).lower() or "rearrange" in str(e).lower():
                try:
                    # Try single image with batch dim
                    single_output = self.pipeline(
                        prompt,
                        image=input_image,
                        num_inference_steps=num_inference_steps,
                        timesteps=timesteps,
                        guidance_scale=guidance_scale
                    ).images[0]
                    
                    # 转换回张量
                    output_np = np.array(single_output).astype(np.float32) / 255.0
                    output_tensor = torch.from_numpy(output_np).to(image_tensor.device)
                    
                    if image_tensor.dim() == 4:
                        output_tensor = output_tensor.unsqueeze(0)
                    
                    print(f"   ✅ Single-image processing succeeded")
                    return output_tensor
                    
                except Exception as e2:
                    print(f"   ❌ Single-image processing also failed: {e2}")
            
            # Check for tensor size mismatch errors
            elif "size of tensor" in str(e).lower() and "must match" in str(e).lower():
                return image_tensor
            
            import traceback
            traceback.print_exc()
            # Return original input to keep training running
            return image_tensor
    
    def load_ref_image(self, train_idx: int, trainset) -> Optional[torch.Tensor]:
        try:
            # Check ref_image_dir
            if self.ref_image_dir is None:
                return None
            
            # Get COLMAP index from trainset
            try:
                train_data = trainset[train_idx]
                colmap_idx = train_data["colmap_image_id"]
                if isinstance(colmap_idx, torch.Tensor):
                    colmap_idx = colmap_idx.item()
                
                print(f"🔍 Train index {train_idx} -> COLMAP index {colmap_idx}")
                
            except Exception as e:
                return None
            
            # Build reference image path by COLMAP index
            ref_image_path = f"{self.ref_image_dir}/{colmap_idx:03d}.png"
            
            # Check file existence
            if not os.path.exists(ref_image_path):
                return None
            
            # Load image
            from PIL import Image
            import numpy as np
            
            ref_image_pil = Image.open(ref_image_path).convert('RGB')
            ref_image_np = np.array(ref_image_pil) / 255.0  # normalize to [0,1]
            ref_image_tensor = torch.from_numpy(ref_image_np).float().to(self.device)
            
            return ref_image_tensor
            
        except Exception as e:
            return None
    
    def process_virtual_views_batch(
        self, 
        trainset, 
        camera_optimizer, 
        rasterize_splats_fn,
        cfg,
        step: int,
        ref_image: Optional[torch.Tensor] = None,
        save_comparisons: bool = True,
        comparison_dir: Optional[str] = None
    ) -> List[dict]:
        # Set comparison image save directory
        if comparison_dir is not None:
            self.difix3d_comparison_dir = comparison_dir
        
        if not self.enabled or self.pipeline is None:
            print("⚠️ DiFix3D disabled; cannot batch process")
            return []
        
        if not hasattr(trainset, '__len__') or len(trainset) == 0:
            print("⚠️ Training set empty; cannot batch process")
            return []
        
        # Ensure interpolation pool initialized
        if not self.is_initialized:
            self.initialize_interpolation_pool(trainset, rasterize_splats_fn, cfg)
            if not self.is_initialized:
                print("❌ Interpolation pool initialization failed")
                return []
        
        print(f"🎯 Step {step}: Start processing virtual view batch")
        
        enhanced_samples = []
        quality_threshold = 0  # k <= 0 表示质量可接受
        
        try:
            # 1. Choose interpolation strategy
            if len(self.available_interpolation_views) < 1:
                print(f"❌ Not enough views in pool ({len(self.available_interpolation_views)} < 1); cannot interpolate")
                return []
            
            # Randomly choose two training views for forward interpolation
            train_indices = torch.randperm(len(trainset))[:2]
            train_view1 = trainset[train_indices[0]]
            train_view2 = trainset[train_indices[1]]
            
            # Backward interpolation: choose two virtual views
            print(f"   🔍 Virtual view pool: {len(self.available_interpolation_views)} available views")
            if len(self.available_interpolation_views) >= 2:
                virtual_indices = torch.randperm(len(self.available_interpolation_views))[:2]
                virtual_view1 = self.available_interpolation_views[virtual_indices[0]]
                virtual_view2 = self.available_interpolation_views[virtual_indices[1]]
                use_backward_interpolation = True
                print(f"   ✅ Backward interpolation enabled: choose virtual views {virtual_indices[0]} and {virtual_indices[1]}")
            else:
                # If not enough virtual views, use forward interpolation only
                use_backward_interpolation = False
                virtual_view1 = None
                virtual_view2 = None
            print(f"   Forward interpolation bases: training views {train_indices[0]} and {train_indices[1]}")
            if use_backward_interpolation:
                print(f"   Backward interpolation bases: virtual views {virtual_view1['source']} and {virtual_view2['source']}")
            else:
                print(f"   Backward interpolation: skipped (need at least 2 virtual views)")
            
            # 🔍 Debug: check base views
            print(f"   🔍 基础视角调试:")
            train_pos1 = train_view1['camtoworld'][:3, 3].to(self.device)
            train_pos2 = train_view2['camtoworld'][:3, 3].to(self.device)
            if use_backward_interpolation:
                virtual_pos1 = virtual_view1['pose'][:3, 3].to(self.device)
                virtual_pos2 = virtual_view2['pose'][:3, 3].to(self.device)
                
            
            # 2. Generate forward and backward interpolations
            forward_alpha = 0.5  # 前向插值：在训练视角之间
            backward_alpha = 1.5  # 后向插值：在虚拟视角之外（向外探索）
            
            quality_scores = []
            interpolated_poses = []
            
            # Forward interpolation between training views
            print(f"   🎯 Forward interpolation between training views (α={forward_alpha})")
            # 确保训练视角数据在正确的设备上
            train_pose1 = train_view1["camtoworld"].to(self.device)
            train_K1 = train_view1["K"].to(self.device)
            train_pose2 = train_view2["camtoworld"].to(self.device)
            train_K2 = train_view2["K"].to(self.device)
            
            interpolated_pose_forward, _ = se3_interpolate_to_target(
                train_pose1, train_K1, 
                train_pose2, train_K2, 
                t=forward_alpha
            )
            interpolated_poses.append(interpolated_pose_forward)
            
            # Backward interpolation: outside virtual views (exploration)
            interpolated_pose_backward = None
            if use_backward_interpolation:
                print(f"   🎯 Backward interpolation: explore outside virtual views (α={backward_alpha})")
                # 确保虚拟视角数据在正确的设备上
                virtual_pose1 = virtual_view1["pose"].to(self.device)
                virtual_K1 = virtual_view1["K"].to(self.device)
                virtual_pose2 = virtual_view2["pose"].to(self.device)
                virtual_K2 = virtual_view2["K"].to(self.device)
                
                # Backward interpolation exploration: extend from virtual_pose1 towards virtual_pose2
                # t=1.5 means 0.5 beyond virtual_pose2
                interpolated_pose_backward, _ = se3_interpolate_to_target(
                    virtual_pose1, virtual_K1, 
                    virtual_pose2, virtual_K2, 
                    t=backward_alpha
                )
                interpolated_poses.append(interpolated_pose_backward)
                
                # 调试：检查反向插值后的相机位置
                interp_pos_backward = interpolated_pose_backward[:3, 3]
                virtual_pos1 = virtual_pose1[:3, 3]
                virtual_pos2 = virtual_pose2[:3, 3]
            else:
                print(f"   🚫 Backward interpolation: skipped (not enough virtual views)")
            
            # 3. Generate interpolated frames
            print(f"   ✅ Start generating {cfg.virtual_view_poses_per_step} interpolated frames")
            
            for i in range(cfg.virtual_view_poses_per_step):
                # 交替生成前向和后向插值
                if i == 0:
                    # Forward interpolation: between training views
                    interpolated_pose = interpolated_pose_forward
                    direction = "forward"
                    alpha = forward_alpha
                elif i == 1 and use_backward_interpolation:
                    # Backward interpolation: explore outside virtual views
                    interpolated_pose = interpolated_pose_backward
                    direction = "backward"
                    alpha = backward_alpha
                elif i == 1 and not use_backward_interpolation:
                    # 如果后向插值不可用，跳过
                    continue
                else:
                    # For more than 2, randomly choose interpolation strategy
                    if torch.rand(1).item() < 0.5:
                        # Between training views
                        alpha = torch.rand(1).item() * 0.8 + 0.1
                        interpolated_pose, _ = se3_interpolate_to_target(
                            train_pose1, train_K1, 
                            train_pose2, train_K2, 
                            t=alpha
                        )
                        direction = "forward-random"
                    elif use_backward_interpolation:
                        # Explore outside virtual views (random direction)
                        if torch.rand(1).item() < 0.5:
                            # Forward exploration: t > 1.0
                            alpha = torch.rand(1).item() * 0.5 + 1.0  # [1.0, 1.5]
                            interpolated_pose, _ = se3_interpolate_to_target(
                                virtual_pose1, virtual_K1, 
                                virtual_pose2, virtual_K2, 
                                t=alpha
                            )
                            direction = "backward-random-forward"
                        else:
                            # Backward exploration: t < 0.0
                            alpha = torch.rand(1).item() * 0.5 - 0.5  # [-0.5, 0.0]
                            interpolated_pose, _ = se3_interpolate_to_target(
                                virtual_pose2, virtual_K2, 
                                virtual_pose1, virtual_K1, 
                                t=alpha
                            )
                            direction = "backward-random-backward"
                    else:
                        # If backward interpolation unavailable, use forward interpolation
                        alpha = torch.rand(1).item() * 0.8 + 0.1
                        interpolated_pose, _ = se3_interpolate_to_target(
                            train_pose1, train_K1, 
                            train_pose2, train_K2, 
                            t=alpha
                        )
                        direction = "forward-random"
                
                print(f"   🎯 Generate {direction} interpolation frame (α={alpha:.3f})")
                
                # 🔍 Debug: check interpolated camera position
                interp_pos = interpolated_pose[:3, 3]
                
                # Use training intrinsics and image ID; ensure correct device
                interp_K = train_view1["K"].unsqueeze(0).to(self.device)  # [1, 3, 3]
                interp_img_id = train_view1["image_id"].unsqueeze(0).to(self.device)
                
                # Ensure interpolated pose is on correct device
                interpolated_pose = interpolated_pose.to(self.device)  # [4, 4]
                
                # Get image size (from training view)
                
                # Check image shape
                if len(train_view1["image"].shape) == 4:  # [1, H, W, 3]
                    height, width = train_view1["image"].shape[1:3]  # [H, W]
                    print(f"     4D image shape: [1, {height}, {width}, 3]")
                elif len(train_view1["image"].shape) == 3:  # [H, W, 3]
                    height, width = train_view1["image"].shape[:2]  # [H, W]
                    print(f"     3D image shape: [{height}, {width}, 3]")
                else:
                    print(f"     ⚠️ Unexpected image shape: {train_view1['image'].shape}")
                    # 使用默认尺寸
                    height, width = 400, 600  # 假设是400x600
                    print(f"     Use default size: height={height}, width={width}")
                
                # Dataset __getitem__ returns image as [H, W, 3]; use [:2] for size
                if len(train_view1["image"].shape) == 3:
                    height, width = train_view1["image"].shape[:2]  # [H, W]
                    print(f"     ✅ Use 3D image shape: [{height}, {width}, 3]")
                
                print(f"     Final extracted height: {height}, width: {width}")
                
                # Render interpolated view (with depth info)
                renders_interp, depths_interp, _ = rasterize_splats_fn(
                    camtoworlds=interpolated_pose.unsqueeze(0),  # [1, 4, 4]
                    Ks=interp_K,  # [1, 3, 3]
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=interp_img_id,
                    render_mode="RGB+ED" if cfg.enable_depth_smooth_loss else "RGB",
                )
                
                # Ensure render results are on correct device
                renders_interp = renders_interp.to(self.device)
                if depths_interp is not None:
                    depths_interp = depths_interp.to(self.device)
                
                
                # Find nearest training view as reference
                nearest_train_idx = self._find_nearest_training_view(interpolated_pose, trainset)
                nearest_train_data = trainset[nearest_train_idx]
                
                # Render nearest training view as reference image
                train_pose = nearest_train_data["camtoworld"].unsqueeze(0).to(self.device)
                train_K = nearest_train_data["K"].unsqueeze(0).to(self.device)
                
                # Get training view ID
                if isinstance(nearest_train_data["image_id"], int):
                    train_view_id = nearest_train_data["image_id"]
                else:
                    train_view_id = nearest_train_data["image_id"].item()
                
                # Enhance interpolated view with DiFix3D
                # Choose reference image for interpolated view
                ref_image_for_interp = None
                if cfg.difix3d_use_ref_image:
                    # Load reference image from preset directory by train index
                    ref_image_for_interp = self.load_ref_image(nearest_train_idx, trainset)
                    
                    if ref_image_for_interp is not None:
                        print(f"   📷 Loaded reference image from directory (train_idx={nearest_train_idx})")
                        print(f"   🔍 ref_image_for_interp shape: {ref_image_for_interp.shape}")
                        print(f"   🔍 ref_image_for_interp device: {ref_image_for_interp.device}")
                    else:
                        print(f"   ⚠️ Cannot load reference image (train_idx={nearest_train_idx}); will not use reference")
                else:
                    print(f"   🚫 Not using reference image for DiFix3D")
                
                
                # Ensure render format correct; handle RGB+ED 4-channel output
                if renders_interp[0].dim() != 3:
                    print(f"     ⚠️ Render result dims incorrect; skip DiFix3D processing")
                    enhanced_interp = renders_interp[0]  # 直接使用原始渲染结果
                elif renders_interp[0].shape[-1] == 4:
                    # RGB+ED: use first 3 channels (RGB) for DiFix3D
                    print(f"     🔧 RGB+ED: extract RGB channels for DiFix3D")
                    rgb_interp = renders_interp[0][:, :, :3]  # [H, W, 3]
                    enhanced_interp = self.process_image(
                        rgb_interp,  # [H, W, 3]
                        prompt=cfg.difix3d_prompt,
                        num_inference_steps=cfg.difix3d_num_inference_steps,
                        timesteps=[199],
                        guidance_scale=cfg.difix3d_guidance_scale,
                        ref_image=ref_image_for_interp,
                        save_comparison=cfg.difix3d_save_comparisons,  # save based on config
                        save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{i}_rgb_ed"
                    )
                elif renders_interp[0].shape[-1] == 3:
                    # Standard RGB mode
                    enhanced_interp = self.process_image(
                        renders_interp[0],  # [H, W, 3]
                        prompt=cfg.difix3d_prompt,
                        num_inference_steps=cfg.difix3d_num_inference_steps,
                        timesteps=[199],
                        guidance_scale=cfg.difix3d_guidance_scale,
                        ref_image=ref_image_for_interp,
                        save_comparison=cfg.difix3d_save_comparisons,  # save based on config
                        save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{i}_rgb"
                    )
                else:
                    print(f"     ⚠️ Incorrect render channel count; skip DiFix3D processing")
                    enhanced_interp = renders_interp[0]  # 直接使用原始渲染结果
                
                # Ensure all tensors on correct device (before scoring)
                interpolated_pose_device = interpolated_pose.to(self.device)
                interp_K_device = interp_K[0].to(self.device)
                interp_img_id_device = interp_img_id[0].to(self.device)
                
                # Compute quality score (ensure both inputs are RGB)
                try:
                    # Ensure original image used for scoring is RGB
                    if renders_interp[0].shape[-1] == 4:
                        original_rgb_for_score = renders_interp[0][:, :, :3]  # [H, W, 3]
                    else:
                        original_rgb_for_score = renders_interp[0]  # [H, W, 3]
                    
                    _, quality_score = self.quality_scorer.score_pseudo_view(
                        original_rgb_for_score, enhanced_interp
                    )
                    print(f"   📊 Interpolated frame quality score: k={quality_score:.4f}")
                    
                    # Save virtual view quality score data
                    score_data = {
                        "step": step,
                        "view_idx": i,
                        "direction": direction,
                        "alpha": alpha,
                        "quality_score": float(quality_score),
                        "nearest_train_idx": nearest_train_idx,
                        "interpolated_pose": interpolated_pose_device.cpu().numpy().tolist(),
                        "timestamp": time.time()
                    }
                    self.virtual_view_scores.append(score_data)
                    print(f"   💾 Saved virtual view quality score data")
                    
                except Exception as e:
                    print(f"   ⚠️ Quality scoring failed: {e}")
                    quality_score = 0.0  # 默认质量评分
                    
                    # Save data even if scoring fails
                    score_data = {
                        "step": step,
                        "view_idx": i,
                        "direction": direction,
                        "alpha": alpha,
                        "quality_score": 0.0,
                        "nearest_train_idx": nearest_train_idx,
                        "interpolated_pose": interpolated_pose_device.cpu().numpy().tolist(),
                        "timestamp": time.time(),
                        "error": str(e)
                    }
                    self.virtual_view_scores.append(score_data)
                    print(f"   💾 Saved virtual view quality score data (scoring failed)")
                
                # Ensure tensors on correct device
                enhanced_interp_device = enhanced_interp.to(self.device)
                
                # Add to enhanced samples (for re-rendering and loss)
                # ✅ Do not store original_image; store render params and re-render each step
                sample = {
                    "enhanced_image": enhanced_interp_device.detach().clone(),  # [H, W, 3] - DiFix增强后的图像（作为监督信号）
                    "pose": interpolated_pose_device.detach().clone(),  # [4, 4] - 用于重新渲染
                    "K": interp_K_device.detach().clone(),  # [3, 3] - 用于重新渲染
                    "image_id": interp_img_id_device.detach().clone(),  # 用于重新渲染
                    "width": width,  # 图像宽度
                    "height": height,  # 图像高度
                    "view_idx": i,
                    "interpolated": True,
                    "alpha": alpha,
                    "nearest_train_idx": nearest_train_idx,
                    "quality_score": quality_score,  # 添加质量评分
                }
                
                # Debug: check device of tensors in sample
                print(f"   🔍 样本{i}张量设备检查:")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: 设备={value.device}, 形状={value.shape}")
                    else:
                        print(f"     {key}: 非张量={type(value)}")
                should_add_to_pool = False
                # Forward interpolation: quality check (PSNR delta)
                # Accept if PSNR delta within configured range
                if (quality_score < cfg.interp_quality_psnr_max) and (quality_score > cfg.interp_quality_psnr_min):
                    should_add_to_pool = True
                    enhanced_samples.append(sample)
                    print(f"   ✅ Forward interpolation accepted (PSNR Δ={quality_score:.4f}, range {cfg.interp_quality_psnr_min}~{cfg.interp_quality_psnr_max}); add to pool")
                else:
                    print(f"   ❌ Forward interpolation rejected (PSNR Δ={quality_score:.4f}, out of range {cfg.interp_quality_psnr_min}~{cfg.interp_quality_psnr_max}); not added")
                
                # Only accepted interpolations go to available pool
                if should_add_to_pool:
                    self.available_interpolation_views.append({
                        "pose": interpolated_pose_device,  # [4, 4] - 确保在正确设备上
                        "K": interp_K_device,  # [3, 3] - 使用插值视角的内参
                        "image_id": interp_img_id_device,  # 使用插值视角的图像ID
                        "enhanced_image": enhanced_interp,
                        "source": f"interpolated_step_{step}_view_{i}",
                        "quality_score": quality_score,  # 记录质量评分
                        "direction": direction  # 记录插帧方向
                    })
                    print(f"   🎯 Interpolation added to pool; current size: {len(self.available_interpolation_views)}")
                else:
                    print(f"   🚫 Interpolation not added; current pool size: {len(self.available_interpolation_views)}")
                
                print(f"   ✅ Interpolated frame {i+1}/{cfg.virtual_view_poses_per_step} processed (α={alpha:.3f}, reference train view={nearest_train_idx})")
                print(f"   🔍 Virtual view pool updated: now {len(self.available_interpolation_views)} views")
            
        except Exception as e:
            print(f"❌ Virtual view batch processing failed: {e}")
            return []
        
        # Print processing results
        if enhanced_samples:
            print(f"🎯 Step {step} virtual view batch completed!")
            print(f"   Generated {len(enhanced_samples)} enhanced views")
            print(f"   Interpolation pool now has {len(self.available_interpolation_views)} views")
        else:
            print(f"⚠️ Step {step} virtual view batch failed; no successful views generated")
        
        return enhanced_samples
    
    def _find_nearest_training_view(self, target_pose: torch.Tensor, trainset) -> int:
        """
        Find the nearest training view to the target pose.
        
        Args:
            target_pose: target pose [4, 4]
            trainset: training dataset
            
        Returns:
            Index of the nearest training view
        """
        min_distance = float('inf')
        nearest_idx = 0
        
        target_position = target_pose[:3, 3]  # [3]
        
        for i in range(len(trainset)):
            train_data = trainset[i]
            train_pose = train_data["camtoworld"].to(self.device)  # [4, 4]
            train_position = train_pose[:3, 3]  # [3]
            
            # Euclidean distance
            distance = torch.norm(target_position - train_position).item()
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        return nearest_idx
    def initialize_interpolation_pool(
        self,
        trainset,
        rasterize_splats_fn,
        cfg
    ):
        """
        One-time initialization of interpolation pool and PSNR baseline.
        
        Args:
            trainset: training dataset
            rasterize_splats_fn: 3DGS rendering function
            cfg: config object
        """
        if self.is_initialized:
            print("🔄 Interpolation pool already initialized; skipping")
            return
        
        print("🚀 Initializing interpolation pool and PSNR baseline...")
        
        # 1. Initialize VirtualViewQualityScorer
        self.quality_scorer = VirtualViewQualityScorer()
        print("✅ VirtualViewQualityScorer initialized")
        
        # 2. Process training views and compute fixed PSNR baseline
        print("📊 Processing training views and computing PSNR baseline...")
        
        # Use first 3 training views as baseline
        num_training_views = min(3, len(trainset))
        all_original_views = []
        all_difix_views = []
        
        for i in range(num_training_views):
            try:
                # Get training view data
                train_data = trainset[i]
                train_pose = train_data["camtoworld"].unsqueeze(0).to(self.device)  # [1, 4, 4]
                train_K = train_data["K"].unsqueeze(0).to(self.device)  # [1, 3, 3]
                train_image = train_data["image"].unsqueeze(0).to(self.device) / 255.0  # [1, H, W, 3]
                
                # Ensure image_id is tensor
                if isinstance(train_data["image_id"], int):
                    train_img_id = torch.tensor([train_data["image_id"]], device=self.device)
                else:
                    train_img_id = train_data["image_id"].unsqueeze(0).to(self.device)
                
                height, width = train_image.shape[1:3]
                
                # Render training view
                renders_train, _, _ = rasterize_splats_fn(
                    camtoworlds=train_pose,
                    Ks=train_K,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=train_img_id,
                    render_mode="RGB",
                )
                
                # Enhance training view using DiFix3D
                print(f"   🎨 Start DiFix3D processing for training view {i+1}...")
                
                # Choose reference image: use another training view's render
                ref_image_for_training = None
                if cfg.difix3d_use_ref_image:
                    # Choose another training view's raw render as reference
                    ref_idx = (i + 1) % num_training_views
                    if ref_idx != i:  # 确保不是同一个视角
                        # Render reference view
                        ref_train_data = trainset[ref_idx]
                        ref_train_pose = ref_train_data["camtoworld"].unsqueeze(0).to(self.device)
                        ref_train_K = ref_train_data["K"].unsqueeze(0).to(self.device)
                        
                        # Ensure image_id is tensor
                        if isinstance(ref_train_data["image_id"], int):
                            ref_train_img_id = torch.tensor([ref_train_data["image_id"]], device=self.device)
                        else:
                            ref_train_img_id = ref_train_data["image_id"].unsqueeze(0).to(self.device)
                        
                        # Render reference view
                        ref_renders, _, _ = rasterize_splats_fn(
                            camtoworlds=ref_train_pose,
                            Ks=ref_train_K,
                            width=width,  # 使用当前视角的尺寸
                            height=height,
                            sh_degree=cfg.sh_degree,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            image_ids=ref_train_img_id,
                            render_mode="RGB",
                        )
                        
                        ref_image_for_training = ref_renders[0].to(self.device)  # [H, W, 3] - correct device
                        print(f"   📷 Use training view {ref_idx+1} raw render as reference image")
                    else:
                        print(f"   🚫 Cannot select a different training view; skip reference image")
                else:
                    print(f"   🚫 Not using reference image for DiFix3D")
                
                enhanced_train = self.process_image(
                    renders_train[0],  # [H, W, 3]
                    prompt=cfg.difix3d_prompt,
                    num_inference_steps=cfg.difix3d_num_inference_steps,
                    timesteps=[199],
                    guidance_scale=cfg.difix3d_guidance_scale,
                    ref_image=ref_image_for_training,  # 使用选择的参考图像
                    save_comparison=False
                )
                print(f"   🎨 DiFix3D processing completed")
                
                # Collect data for PSNR computation
                all_original_views.append(renders_train[0])
                all_difix_views.append(enhanced_train)
                
                # 🔍 Debug: check image equality
                print(f"   🔍 训练视角 {i+1} 调试信息:")
                print(f"     原始图像形状: {renders_train[0].shape}, 范围: [{renders_train[0].min():.4f}, {renders_train[0].max():.4f}]")
                print(f"     DiFix图像形状: {enhanced_train.shape}, 范围: [{enhanced_train.min():.4f}, {enhanced_train.max():.4f}]")
                
                # 计算MSE来检查图像差异
                mse = torch.mean((renders_train[0] - enhanced_train) ** 2)
                print(f"     Image MSE: {mse.item():.8f}")
                
                if mse < 1e-8:
                    print(f"     ⚠️ Warning: original and DiFix images nearly identical!")
                    print(f"       DiFix3D processing might not be effective")
                else:
                    print(f"     ✅ Images differ; DiFix3D processing effective")
                
                # Directly add to available interpolation pool (training views don't need scoring)
                self.available_interpolation_views.append({
                    "pose": train_pose[0],  # [4, 4]
                    "K": train_K[0],  # [3, 3]
                    "image_id": train_img_id[0],
                    "enhanced_image": enhanced_train,
                    "source": f"training_view_{i}"
                })
                
                print(f"   ✅ Training view {i+1}/{num_training_views} processed")
                
            except Exception as e:
                print(f"   ❌ Training view {i} processing failed: {e}")
                continue
        
        if len(all_original_views) == 0:
            print("❌ No successfully processed training views; cannot compute PSNR baseline")
            return
        
        # 3. Compute fixed PSNR baseline (not updated later)
        try:
            self.training_psnr_mean, self.training_psnr_variance = self.quality_scorer.evaluate_training_views(
                all_original_views, all_difix_views
            )
            
            # Validate PSNR values
            if np.isinf(self.training_psnr_mean) or np.isnan(self.training_psnr_mean):
                print(f"❌ Invalid PSNR mean: {self.training_psnr_mean}")
                print(f"   This often means DiFix3D output equals original image")
                print(f"   Please verify DiFix3D works properly")
                raise ValueError("PSNR mean is inf; DiFix3D may not be effective")
            
            if np.isinf(self.training_psnr_variance) or np.isnan(self.training_psnr_variance):
                print(f"❌ Invalid PSNR variance: {self.training_psnr_variance}")
                print(f"   This often means all PSNR values are identical (inf)")
                raise ValueError("PSNR variance is nan; images may be identical")
            
            print(f"📊 Fixed PSNR baseline computed: mean={self.training_psnr_mean:.4f}, var={self.training_psnr_variance:.4f}")
            
            # Save baseline scores
            self.baseline_scores = {
                "training_psnr_mean": float(self.training_psnr_mean),
                "training_psnr_variance": float(self.training_psnr_variance),
                "training_views_count": len(all_original_views),
                "timestamp": time.time()
            }
            print(f"💾 Saved baseline scores: mean={self.training_psnr_mean:.4f}, var={self.training_psnr_variance:.4f}")
            
        except Exception as e:
            print(f"❌ PSNR baseline computation failed: {e}")
            print(f"   Possible causes:")
            print(f"   1. DiFix3D may be ineffective and returned the original image")
            print(f"   2. Image data may be problematic")
            print(f"   3. Please check if DiFix3D model is loaded correctly")
            raise e
        print(f"🔄 Available interpolation view pool initialized with {len(self.available_interpolation_views)} training views")
        
        # Mark as initialized
        self.is_initialized = True
        print("✅ Interpolation pool and PSNR baseline initialization done")
    

class DeblurDiFix3DRunner(Runner):
    """BAD-Gaussians deblurring + DiFix3D training engine"""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: DeblurDiFix3DConfig) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Set output directories
        self.result_dir = cfg.result_dir  # preserve original path
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        # DiFix3D comparison directory
        self.difix3d_comparison_dir = f"{cfg.result_dir}/difix3d_comparisons"
        os.makedirs(self.difix3d_comparison_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Extract scene_name from data_dir to build ref_image path
        self.scene_name = Path(cfg.data_dir).name
        self.ref_image_dir = f"{cfg.data_dir}/ref_image"
        print(f"🔍 Scene name: {self.scene_name}")
        print(f"🔍 Reference image directory: {self.ref_image_dir}")

        # 加载数据
        self.parser = ColmapParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,  # force using original images, no downsampling
            normalize=True,
            scale_factor=cfg.scale_factor,
            # Force disabling automatic downsampling
            downscale_rounding_mode="round",  # use round instead of floor
        )
        # Training indices config: pass CLI/config train_indices to parser
        self.parser.train_indices = cfg.train_indices
        if cfg.train_indices is not None:
            print(f"[Dataset] Using configured training indices: {cfg.train_indices}")
        
        # Debug: check ColmapParser config
        print(f"🔍 ColmapParser config check:")
        print(f"   data_factor: {cfg.data_factor}")
        print(f"   scale_factor: {cfg.scale_factor}")
        print(f"   downscale_rounding_mode: {self.parser.downscale_rounding_mode}")
        print(f"   parser.factor: {self.parser.factor}")
        if hasattr(self.parser, '_downscale_factor'):
            print(f"   parser._downscale_factor: {self.parser._downscale_factor}")
        
        # Check image paths
        print(f"🔍 Image path check:")
        if hasattr(self.parser, 'image_paths') and len(self.parser.image_paths) > 0:
            sample_path = Path(self.parser.image_paths[0])
            print(f"   Sample image path: {sample_path}")
            if sample_path.exists():
                img = Image.open(sample_path)
                print(f"   Sample image size: {img.size}")
            else:
                print(f"   ⚠️ Sample image path does not exist!")
        else:
            print(f"   ⚠️ No image paths found!")

        self.trainset = DeblurNerfDataset(self.parser, split="train")
        
        # Initialize camera trajectory generator
        self.trajectory_generator = generate_camera_trajectory
        self.valset = DeblurNerfDataset(self.parser, split="val")
        self.testset = DeblurNerfDataset(self.parser, split="test")
        self.quality_scorer = VirtualViewQualityScorer(device=self.device)
        print(f"✅ Virtual view quality scoring model initialized")
        # 初始化DiFix3D处理器
        if cfg.enable_difix3d:
            print("🎨 Initializing DiFix3D processor...")
            self.difix3d_processor = DiFix3DProcessor(
                model_name=cfg.difix3d_model_name,
                device=self.device,
                ref_image_dir=self.ref_image_dir
            )
            if self.difix3d_processor.enabled:
                print(f"✅ DiFix3D processor initialized")
            else:
                print(f"⚠️ DiFix3D processor initialization failed; disabling DiFix3D features")
                cfg.enable_difix3d = False
        else:
            self.difix3d_processor = None
            print("🚫 DiFix3D features disabled")

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        # Initialize 3D Gaussian points
        feature_dim = None
        if cfg.app_opt:
            feature_dim = cfg.app_embed_dim

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("模型初始化完成. 高斯点数量:", len(self.splats["means"]))

        # Densification strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # BAD-Gaussians camera optimizer
        self.pose_optimizers = []
        # Total camera count across train/val/test
        total_cameras = len(self.trainset) + (len(self.valset) if self.valset else 0) + (len(self.testset) if self.testset else 0)
        self.camera_optimizer: BadCameraOptimizer = self.cfg.camera_optimizer.setup(
            num_cameras=total_cameras,
            device=self.device,
        )
        camera_optimizer_param_groups = {}
        # Handle DDP-wrapped module
        camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
        camera_optimizer.get_param_groups(camera_optimizer_param_groups)
        self.pose_optimizers = [
            torch.optim.Adam(
                camera_optimizer_param_groups["camera_opt"],
                lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.pose_opt_reg,
            )
        ]
        if world_size > 1:
            self.camera_optimizer = DDP(self.camera_optimizer)

        # Appearance optimizer
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # Bilateral grid
        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1], 
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Evaluation metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        
        # Initialize VGG perceptual loss models (pre-created to avoid repetition)
        self.perceptual_loss = VGG16PerceptualLoss(
            feature_layer='relu2_2',
            device=self.device,
            enable_timing=False
        )
        self.dists_loss = VGG16DISTSLoss(
            device=self.device,
            enable_timing=False
        )

        # Initialize viewer
        if not cfg.disable_viewer:
            import nerfview
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        self.cfg_to_save = cfg
        
        # Store virtual camera batches
        self.virtual_camera_batches = []
        # Store all training camera positions
        self.all_train_cameras = None
        
        # Store virtual view quality scores
        self.virtual_view_scores = []
        # Store baseline score data (training PSNR baseline)
        self.baseline_scores = {}
        
        # Initialize hybrid sampling strategy (on-demand interpolation)
        # Track hybrid sampling state
        self.hybrid_sampling_initialized = False
        
    def collect_train_camera_data(self):
        """
        Collect all training camera poses
        """
        if self.all_train_cameras is None:
            train_cameras = []
            for i in range(len(self.trainset)):
                camera_info = self.trainset[i]
                if 'camtoworld' in camera_info:
                    train_cameras.append(camera_info['camtoworld'])
                elif 'pose' in camera_info:
                    train_cameras.append(camera_info['pose'])
            
            if train_cameras:
                self.all_train_cameras = torch.stack(train_cameras).to(self.device)  # [N, 4, 4]
                print(f"📊 Collected {len(train_cameras)} training camera poses")
            else:
                print("⚠️ Unable to get camera poses from training dataset")


    def collect_virtual_camera_data(self, camera_poses: torch.Tensor = None, enhanced_samples: List[dict] = None, step: int = None, source: str = "unknown"):
        """
        Collect virtual camera poses (unified interface)
        
        Args:
            camera_poses: camera poses [N, 4, 4] (BAD-Gaussians)
            enhanced_samples: list of enhanced samples (DiFix3D)
            step: current step (optional)
            source: data source ("BAD-Gaussians" or "DiFix3D")
        """
        if camera_poses is not None and len(camera_poses) > 0:
            # BAD-Gaussians virtual cameras
            self.virtual_camera_batches.append(camera_poses.detach().clone())
            step_info = f"step {step}: " if step is not None else ""
            print(f"📊 {step_info}collected {len(camera_poses)} {source} virtual cameras")
        elif enhanced_samples:
            # DiFix3D enhanced virtual cameras
            virtual_poses = []
            print(f"🔍 Debug enhanced_samples device info:")
            for i, sample in enumerate(enhanced_samples):
                pose = sample["pose"]  # [4, 4] - no batch dim
                print(f"   sample {i}: pose device={pose.device}, shape={pose.shape}, expected device={self.device}")
                # 确保pose在正确的设备上
                if pose.device != self.device:
                    print(f"   🔧 sample {i}: moving pose from {pose.device} to {self.device}")
                    pose = pose.to(self.device)
                virtual_poses.append(pose.unsqueeze(0))  # 添加batch维度 [1, 4, 4]
            
            if virtual_poses:
                print(f"🔍 Debug virtual_poses device info:")
                for i, pose in enumerate(virtual_poses):
                    print(f"   virtual_poses[{i}]: device={pose.device}, shape={pose.shape}")
                
                try:
                    virtual_cameras_batch = torch.cat(virtual_poses, dim=0)  # [N, 4, 4]
                    self.virtual_camera_batches.append(virtual_cameras_batch)
                    print(f"📊 Collected {len(virtual_poses)} {source} virtual cameras")
                    print(f"   📊 Current virtual camera batch count: {len(self.virtual_camera_batches)}")
                    total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
                    print(f"   📊 Current total virtual cameras: {total_virtual_cameras}")
                except Exception as e:
                    print(f"❌ torch.cat failed: {e}")
                    print(f"   Tensor devices: {[pose.device for pose in virtual_poses]}")
                    raise e


    def train(self):
        """Main training loop - based on simple_trainer_deblur.py"""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]
        
        # pose optimization has a learning rate schedule
        pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.pose_optimizers[0], gamma=cfg.pose_opt_lr_decay ** (1.0 / max_steps)
        )
        schedulers.append(pose_scheduler)

        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=cfg.pin_memory,
        )
        trainloader_iter = iter(trainloader)

        if cfg.visualize_cameras:
            self._init_viewer_state()

        # Collect training camera data before training starts
        if world_rank == 0:
            print("📊 Start collecting training camera poses...")
            self.collect_train_camera_data()

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
                
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device, non_blocking=True)  # [1, 4, 4]
            Ks = data["K"].to(device, non_blocking=True)  # [1, 3, 3]
            pixels = data["image"].to(device, non_blocking=True) / 255.0  # [1, H, W, 3]
            
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device, non_blocking=True)
            if cfg.depth_loss:
                points = data["points"].to(device, non_blocking=True)  # [1, M, 2]
                depths_gt = data["depths"].to(device, non_blocking=True)  # [1, M]

            height, width = pixels.shape[1:3]

            assert camtoworlds.shape[0] == 1
            # 处理DDP包装的情况
            camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
            camtoworlds = camera_optimizer.apply_to_cameras(camtoworlds, image_ids, "uniform")[0]
            assert camtoworlds.shape[0] == cfg.camera_optimizer.num_virtual_views
            Ks = Ks.tile((camtoworlds.shape[0], 1, 1))
            
            # 📊 Disabled BAD-Gaussians virtual camera collection; keep DiFix3D only
            # if step % 1000 == 0:  # 每1000步收集一次，避免数据过多
            #     self.collect_virtual_camera_data(camera_poses=camtoworlds, step=step, source="BAD-Gaussians")

            
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if (cfg.depth_loss or cfg.enable_depth_smooth_loss) else "RGB",
            )
            
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            
            # 🎯 Compute depth smooth loss (every step)
            depth_smooth_loss_value = 0.0

            # BAD-Gaussians: average the virtual views
            colors = colors.mean(0)[None]
            
            # 🎯 Virtual view training strategy
            virtual_view_loss_to_add = 0.0  

            # 🆕 Hybrid sampling strategy: unified virtual view training
            print(f"🔍 Check hybrid sampling: step={step}, virtual_view_start_step={cfg.virtual_view_start_step}, enable_difix3d={cfg.enable_difix3d}, difix3d_processor={self.difix3d_processor is not None}, step%interval={step % cfg.virtual_view_interval}")
            if step >= cfg.virtual_view_start_step and cfg.enable_difix3d and self.difix3d_processor is not None and step % cfg.virtual_view_interval == 0:
                if step == cfg.virtual_view_start_step:
                    print(f"🎯 Step {step}: first enable hybrid-sampling virtual view training")
                else:
                    print(f"🎯 Step {step}: continue hybrid-sampling virtual view training")
                
                # 🆕 Use new virtual view batch processing strategy
                enhanced_samples = self.difix3d_processor.process_virtual_views_batch(
                    trainset=self.trainset,
                    camera_optimizer=self.camera_optimizer,
                    rasterize_splats_fn=self.rasterize_splats,
                    cfg=cfg,
                    step=step,
                    save_comparisons=cfg.difix3d_save_comparisons,
                    comparison_dir=self.difix3d_comparison_dir
                )
                
                if enhanced_samples:
                    # 🎯 Append enhanced samples to class attribute
                    if not hasattr(self, 'enhanced_data'):
                        self.enhanced_data = []
                    elif not isinstance(self.enhanced_data, list):
                        self.enhanced_data = []  # reset to list
                    
                    # Add new samples with total limit
                    max_samples = getattr(cfg, 'difix3d_max_augmented_samples', 100)
                    for enhanced_sample in enhanced_samples:
                        # Limit enhanced data count; keep latest samples
                        if len(self.enhanced_data) >= max_samples:
                            self.enhanced_data.pop(0)  # remove oldest sample
                        
                        self.enhanced_data.append(enhanced_sample)
                    
                    # 📊 Collect virtual camera pose data
                    self.collect_virtual_camera_data(enhanced_samples=enhanced_samples, source="DiFix3D-Progressive")
                    
                    print(f"🎯 Progressive interpolation done:")
                    print(f"   Generated samples this round: {len(enhanced_samples)}")
                    print(f"   Total enhanced samples now: {len(self.enhanced_data)}")
                    print(f"   📊 Current virtual camera batch count: {len(self.virtual_camera_batches)}")
                    total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
                    print(f"   📊 Current total virtual cameras: {total_virtual_cameras}")
                    for i, sample in enumerate(enhanced_samples):
                        quality_score = sample.get('quality_score', 'N/A')
                        if isinstance(quality_score, (int, float)):
                            print(f"   Sample {i}: image_id={sample['image_id'].item()}, quality_score={quality_score:.4f}")
                        else:
                            print(f"   Sample {i}: image_id={sample['image_id'].item()}, quality_score={quality_score}")
                else:
                    print("⚠️ Progressive interpolation failed; no new samples")
                    
            # 🎯 After virtual view start step: compute virtual view loss
            if step >= cfg.virtual_view_start_step and hasattr(self, 'enhanced_data'):
                # 🆕 Randomly choose one enhanced sample for loss (reduce cost)
                if isinstance(self.enhanced_data, list) and len(self.enhanced_data) > 0:
                    # Randomly pick a sample
                    import random
                    sample = random.choice(self.enhanced_data)
                    
                    loss_virtual_sample = 0.0
                    
                    # 🎯 Key: re-render virtual view to connect gradients to current 3D GS
                    virtual_pose = sample["pose"].unsqueeze(0).to(device)  # [1, 4, 4]
                    virtual_K = sample["K"].unsqueeze(0).to(device)  # [1, 3, 3]
                    virtual_image_id = sample["image_id"].unsqueeze(0).to(device)
                    virtual_width = sample["width"]
                    virtual_height = sample["height"]
                    
                    # Re-render virtual view (with gradients connected to current 3D GS)
                    renders_virtual, alphas_virtual, info_virtual = self.rasterize_splats(
                        camtoworlds=virtual_pose,
                        Ks=virtual_K,
                        width=virtual_width,
                        height=virtual_height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=virtual_image_id,
                        render_mode="RGB+ED" if cfg.enable_depth_smooth_loss else "RGB",
                    )
                    
                    # Extract RGB and depth
                    if renders_virtual.shape[-1] == 4:
                        colors_virtual, depths_virtual = renders_virtual[..., 0:3], renders_virtual[..., 3:4]
                    else:
                        colors_virtual, depths_virtual = renders_virtual, None
                    
                    # Apply random background (if enabled)
                    if cfg.random_bkgd:
                        colors_virtual = colors_virtual + bkgd * (1.0 - alphas_virtual)
                    
                    # Use DiFix-enhanced image as supervision (no gradient)
                    enhanced_image = sample["enhanced_image"].to(device)  # [H, W, 3]
                    if enhanced_image.dim() == 3:
                        enhanced_image = enhanced_image.unsqueeze(0)  # [1, H, W, 3]
                    
                    # 🎯 Compute DiFix distillation loss: distill enhanced image into render
                    difix_distillation_loss = 0.0
                    if cfg.enable_difix_enhancement_loss:
                        # L1 loss
                        difix_l1_loss = F.l1_loss(colors_virtual, enhanced_image)
                        
                        # SSIM loss
                        difix_ssim_loss = 1.0 - self.ssim(
                            colors_virtual.permute(0, 3, 1, 2), 
                            enhanced_image.permute(0, 3, 1, 2)
                        )
                        
                        # DISTS perceptual loss
                        difix_dists_loss = self.dists_loss(colors_virtual, enhanced_image)
                        difix_perc_loss = self.perceptual_loss(colors_virtual, enhanced_image)
                        # Combined distillation loss
                        difix_distillation_loss = (
                            difix_l1_loss * cfg.difix_enhancement_l1_weight +
                            difix_dists_loss * 0.01
                        )
                        
                        loss_virtual_sample += difix_distillation_loss
                    
                    # Depth smooth loss (if enabled)
                    if cfg.enable_depth_smooth_loss and depths_virtual is not None:
                        depth_smooth_loss_virtual = depth_smooth_loss_4neighbor(depths_virtual)
                        loss_virtual_sample += depth_smooth_loss_virtual * cfg.depth_smooth_lambda
                    
                    # Apply weight
                    virtual_view_loss_to_add = cfg.virtual_view_loss_weight * loss_virtual_sample
                    
                    # Add debug info
                    if step % 100 == 0:  # 每100步打印一次详细信息
                        print(f"🔍 Virtual view loss debug (step {step}):")
                        print(f"   Available sample count: {len(self.enhanced_data)}")
                        print(f"   Current sample ID: {sample.get('image_id', 'unknown')}")
                        print(f"   Current sample quality score: {sample.get('quality_score', 'N/A'):.4f}")
                        print(f"   Virtual loss: {loss_virtual_sample:.6f}")
                        print(f"   Weighted loss: {virtual_view_loss_to_add:.6f}")
                        print(f"   Weight: {cfg.virtual_view_loss_weight}")
                        if cfg.enable_depth_smooth_loss:
                            print(f"   🔍 Depth smooth loss enabled, weight: {cfg.depth_smooth_lambda}")
                    

            else:
                # Before start step, virtual view loss is 0
                virtual_view_loss_to_add = 0.0
                
           

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]


            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            if self.cfg.fused_ssim:
                ssimloss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid")
            else:
                ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            if cfg.enable_mcmc_opacity_reg:
                loss = loss + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()

            if cfg.enable_mcmc_scale_reg:
                loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()

            if cfg.enable_phys_scale_reg and step % 10 == 0:
                scale_exp = torch.exp(self.splats["scales"])
                scale_reg = (
                    torch.maximum(
                        scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                        torch.tensor(cfg.max_gauss_ratio),
                    )
                    - cfg.max_gauss_ratio
                )
                scale_reg = 0.1 * scale_reg.mean()
                loss += scale_reg

            # 🎯 Add depth smooth loss
            if cfg.enable_depth_smooth_loss and step >= 25000 and depths is not None:
                depth_smooth_loss_value = depth_smooth_loss_4neighbor(depths)
                loss += depth_smooth_loss_value * cfg.depth_smooth_lambda
                if step % 100 == 0:  # print every 100 steps
                    print(f"🔍 Depth smooth loss weight: {cfg.depth_smooth_lambda}, weighted: {depth_smooth_loss_value * cfg.depth_smooth_lambda:.6f}")

            # 🎯 Critical: add virtual view loss after all other losses
            loss += virtual_view_loss_to_add
            
            # If virtual view training enabled, print total loss info
            if virtual_view_loss_to_add > 0:
                print(f"🎯 Final Loss: base={loss.item() - virtual_view_loss_to_add:.4f}, virtual={virtual_view_loss_to_add:.4f}, total={loss.item():.4f}")

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.enable_depth_smooth_loss and depth_smooth_loss_value > 0:
                desc += f"depth smooth={depth_smooth_loss_value.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)

                # monitor camera pose optimization
                metrics_dict = {}
                # Handle DDP-wrapped module
                camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
                camera_optimizer.get_metrics_dict(metrics_dict)
                for k, v in metrics_dict.items():
                    self.writer.add_scalar(f"train/{k}", v, step)

                # monitor pose learning rate
                self.writer.add_scalar("train/poseLR", pose_scheduler.get_last_lr()[0], step)

                # monitor ATE
                #     self.visualize_traj(step)

                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.enable_depth_smooth_loss and depth_smooth_loss_value > 0:
                    self.writer.add_scalar("train/depth_smooth_loss", depth_smooth_loss_value.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if world_size > 1:
                    data["camera_opt"] = self.camera_optimizer.module.state_dict()
                else:
                    data["camera_opt"] = self.camera_optimizer.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                if step % cfg.pose_gradient_accumulation_steps == cfg.pose_gradient_accumulation_steps - 1:
                    optimizer.step()
                if step % cfg.pose_gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                if cfg.deblur_eval_enable_during_training and self.testset is not None:
                    if cfg.deblur_eval_enable_pose_opt:
                        self.eval_with_pose_opt(step, "deblur", self.testset)
                    else:
                        self.eval_deblur(step, "deblur", self.testset)
                if cfg.nvs_eval_enable_during_training and self.valset is not None:
                    self.eval_with_pose_opt(step, "nvs", self.valset)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

        print(f"训练完成. 总时间: {time.time() - global_tic:.2f} seconds")
        
        # 📊 训练结束后生成相机分布可视化
        if world_rank == 0:
            print("📊 生成训练相机和DiFix3D虚拟相机分布图...")
            print(f"   训练相机数量: {len(self.all_train_cameras) if self.all_train_cameras is not None else 0}")
            print(f"   DiFix3D虚拟相机批次数量: {len(self.virtual_camera_batches)}")
            total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
            print(f"   DiFix3D虚拟相机总数量: {total_virtual_cameras}")
            total_cameras = (len(self.all_train_cameras) if self.all_train_cameras is not None else 0) + total_virtual_cameras
            print(f"   相机总数: {total_cameras}")
            
            # 详细分析DiFix3D虚拟相机
            if self.virtual_camera_batches:
                print("📊 DiFix3D虚拟相机详细分析:")
                for i, batch in enumerate(self.virtual_camera_batches):
                    print(f"   批次 {i+1}: {len(batch)} 个DiFix3D虚拟相机")
                    print(f"     -> DiFix3D渐进式插值虚拟视角")
                    # 显示每个批次的相机位置范围
                    if len(batch) > 0:
                        positions = batch[:, :3, 3].cpu().numpy()
                        print(f"       位置范围: X=[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], Y=[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], Z=[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
            else:
                print("⚠️ 没有DiFix3D虚拟相机数据")
            
            # 保存最终的质量评分数据
            if self.difix3d_processor is not None:
                self.difix3d_processor.save_quality_scores_to_json(step=max_steps-1, result_dir=self.result_dir)


    @torch.no_grad()
    def eval_deblur(self, step: int, stage: str, dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(testloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            # Apply learned mid-virtual-view pose optimizations
            # 处理DDP包装的情况
            camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
            camtoworlds = camera_optimizer.apply_to_cameras(camtoworlds, image_ids, "mid")

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            if world_rank == 0:
                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(f"{self.render_dir}/{step:04d}_{stage}_{i:04d}.png", (canvas * 255).astype(np.uint8))

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))
                    # write images
                    canvas = torch.cat([pixels, cc_colors], dim=2).squeeze(0).cpu().numpy()
                    imageio.imwrite(
                        f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_corrected.png", (canvas * 255).astype(np.uint8)
                    )

        if world_rank == 0:
            ellipse_time /= len(testloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            
            # 添加最佳结果统计（新增功能）
            best_stats = {}
            for k, v in metrics.items():
                if "psnr" in k or "ssim" in k:
                    best_stats[f"best_{k}"] = torch.stack(v).max().item()
                elif "lpips" in k:
                    best_stats[f"best_{k}"] = torch.stack(v).min().item()
            
            # 添加每个样本的详细结果（新增功能）
            detailed_results = {}
            for k, v in metrics.items():
                detailed_results[f"{k}_per_sample"] = [float(val.item()) for val in v]
            
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            
            # 合并所有统计信息
            final_stats = {**stats, **best_stats, **detailed_results}
            
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # 打印最佳结果（新增功能）
           
            
            # save stats as json（保持原有功能，但增加更多信息）
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(final_stats, f, indent=2)
            
            # save stats to tensorboard（保持原有功能）
            for k, v in final_stats.items():
                if not k.endswith("_per_sample"):  # 只保存汇总指标到tensorboard
                    self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    def eval_with_pose_opt(self, step: int, stage: str, dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # Freeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = False

        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]

            eval_pose_adjust = CameraOptModuleSE3(1).to(self.device)
            eval_pose_adjust.random_init(cfg.pose_noise)
            eval_pose_optimizer = torch.optim.Adam(
                eval_pose_adjust.parameters(),
                lr=cfg.nvs_pose_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.nvs_pose_reg,
                eps=1e-15,
            )

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                eval_pose_optimizer, gamma=cfg.pose_opt_lr_decay ** (1.0 / cfg.max_steps)
            )

            NVS_STEPS = cfg.nvs_steps_final if step == cfg.max_steps - 1 else cfg.nvs_steps
            for j in range(NVS_STEPS):
                camtoworlds_new = eval_pose_adjust(camtoworlds, torch.tensor([0]).to(self.device))
                colors, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds_new,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB",
                )
                # clamping here should be fine since we are only optimizing the camera
                colors = torch.clamp(colors, 0.0, 1.0)
                colors_p = colors.permute(0, 3, 1, 2).detach()  # [1, 3, H, W]

                # loss
                l1loss = F.l1_loss(colors, pixels)
                loss = l1loss

                loss.backward()

                eval_pose_optimizer.step()
                eval_pose_optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                with torch.no_grad():
                    if j % 20 == 0:
                        psnr = self.psnr(colors_p, pixels_p)
                        ssim = self.ssim(colors_p, pixels_p)
                        lpips = self.lpips(colors_p, pixels_p)
                        print(
                            f"Stage {stage} at Step_{step:04d}:"
                            f"NVS_IMG_#{i:04d}_step_{j:04d}:"
                            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                        )
                        if cfg.use_bilateral_grid:
                            cc_colors = color_correct(colors, pixels)
                            cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                            cc_psnr = self.psnr(cc_colors_p, pixels_p)
                            cc_ssim = self.ssim(cc_colors_p, pixels_p)
                            cc_lpips = self.lpips(cc_colors_p, pixels_p)
                            print(
                                f"Corrected PSNR: {cc_psnr.item():.3f}, SSIM: {cc_ssim.item():.4f}, LPIPS: {cc_lpips.item():.3f} "
                            )
                        # # NVS Debugging
                        # stats = {
                        #     "psnr": psnr.item(),
                        #     "ssim": ssim.item(),
                        #     "lpips": lpips.item(),
                        # }
                        # for k, v in stats.items():
                        #     self.writer.add_scalar(f"nvs/{step}/{i}/{k}", v, j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/pose_lr", scheduler.get_last_lr()[0], j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/camera_opt_translation", eval_pose_adjust.poses_opt[:, :3].mean(), j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/camera_opt_rotation", eval_pose_adjust.poses_opt[:, 3:].mean(), j)
                        # self.writer.flush()
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["lpips"].append(lpips)
            if cfg.use_bilateral_grid:
                metrics["cc_psnr"].append(cc_psnr)
                metrics["cc_ssim"].append(cc_ssim)
                metrics["cc_lpips"].append(cc_lpips)
            
            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).detach().cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_{j:04d}.png", (canvas * 255).astype(np.uint8)
            )
            if cfg.use_bilateral_grid:
                canvas = torch.cat([pixels, cc_colors], dim=2).squeeze(0).detach().cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_{j:04d}_corrected.png",
                    (canvas * 255).astype(np.uint8),
                )
        # 计算平均值
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        
        # 计算最佳值（对于PSNR和SSIM是最大值，对于LPIPS是最小值）
        best_stats = {}
        for k, v in metrics.items():
            if "psnr" in k or "ssim" in k:
                best_stats[f"best_{k}"] = torch.stack(v).max().item()
            elif "lpips" in k:
                best_stats[f"best_{k}"] = torch.stack(v).min().item()
        
        # 保存每个样本的详细结果
        detailed_results = {}
        for k, v in metrics.items():
            detailed_results[f"{k}_per_sample"] = [float(val.item()) for val in v]
        
        # 合并所有统计信息
        final_stats = {**stats, **best_stats, **detailed_results}
        
        # 打印最佳结果
        print(f"Best PSNR: {best_stats['best_psnr']:.3f}, Best SSIM: {best_stats['best_ssim']:.4f}, Best LPIPS: {best_stats['best_lpips']:.3f}")
        if cfg.use_bilateral_grid:
            print(f"Best Corrected PSNR: {best_stats['best_cc_psnr']:.3f}, Best Corrected SSIM: {best_stats['best_cc_ssim']:.4f}, Best Corrected LPIPS: {best_stats['best_cc_lpips']:.3f}")
        
        # save stats as json
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(final_stats, f, indent=2)

        # save stats to tensorboard
        for k, v in final_stats.items():
            if not k.endswith("_per_sample"):  # 只保存汇总指标到tensorboard
                self.writer.add_scalar(f"{stage}/{k}", v, step)
        self.writer.flush()

        # Unfreeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = True

    @torch.no_grad()
    def eval_traj(self, step: int):
        # TODO: add gt trajectory

        camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
        camtoworlds = camera_optimizer.get_cameras()

        raise NotImplementedError

    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        if not self.cfg.disable_viewer and isinstance(self.viewer, PoseViewer):
            assert self.viewer and self.trainset
            self.viewer.init_scene(train_dataset=self.trainset, train_state="training")


def main(local_rank: int, world_rank, world_size: int, cfg: DeblurDiFix3DConfig):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_size > 1:
            print("Viewer is disabled in distributed training.")

    runner = DeblurDiFix3DRunner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [torch.load(file, map_location=runner.device, weights_only=False) for file in cfg.ckpt]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k].detach().to(runner.device) for ckpt in ckpts])
        runner.camera_optimizer.load_state_dict(ckpts[0]["camera_opt"])
        step = ckpts[0]["step"]
        if runner.testset is not None:
            if cfg.deblur_eval_enable_pose_opt:
                runner.eval_with_pose_opt(step=step, stage="deblur", dataset=runner.testset)
            else:
                runner.eval_deblur(step=step, stage="deblur", dataset=runner.testset)
        if runner.valset is not None:
            runner.eval_with_pose_opt(step=step, stage="nvs", dataset=runner.valset)

        runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:
    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default
    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25
    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            DeblurDiFix3DConfig(
                strategy=DefaultStrategy(
                    verbose=True,
                    grow_grad2d=3e-3,
                    absgrad=True,
                    refine_start_iter=1000,
                ),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            DeblurDiFix3DConfig(
                init_opa=0.5,
                init_scale=0.1,
                strategy=MCMCStrategy(verbose=True, cap_max=500_000),
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
