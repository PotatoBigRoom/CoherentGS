#!/usr/bin/env python3
"""
BAD-Gaussians去模糊训练器 + DiFix3D集成版本

基于simple_trainer_deblur.py，集成DiFix3D图像增强功能
支持运动模糊去除和DiFix3D联合优化
集成SE(3)混合采样策略
"""

import json
import math
import os
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

# 导入SE(3)插帧模块
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

# Perceptual Loss导入
from pection_loss import VGG16PerceptualLoss, VGG16PerceptualLossWithMultipleLayers, VGG16DISTSLoss

# DiFix3D集成（使用本地模块，不注入绝对路径）
from PIL import Image
try:
    # 直接从同目录下的 pipeline_difix.py 导入
    from pipeline_difix import DifixPipeline
    DIFIX3D_AVAILABLE = True
except Exception as e:
    DIFIX3D_AVAILABLE = False
    print(f"警告: 未找到本地 pipeline_difix 或依赖缺失，将禁用DiFix3D功能: {e}")
    

@dataclass
class DeblurDiFix3DConfig(Config):
    """BAD-Gaussians去模糊 + DiFix3D配置"""
    
    # 数据配置
    data_dir: str = "/remote-home/fcr/Event_proj/DeblurDIFIXZK/BAD-Gaussians-gsplat-only_vgg3/data/bad-nerf-gtK-colmap-nvs/blurpool"
    data_factor: int = 1
    # 指定训练集图像ID列表（可选）；None 表示使用默认的全部训练视角
    train_indices: Optional[List[int]] = None
    
    # 评估配置s
    eval_only: bool = False
    """是否仅执行评估"""
    eval_steps: List[int] = field(default_factory=lambda: [3_000, 7_000])
    """评估步骤列表"""
    scale_factor: float = 1.0
    result_dir: str = "/remote-home/fcr/Event_proj/DeblurDIFIXZK/results"
    test_every: int = 8

    ########### Viewer ###############
    disable_viewer: bool = False
    port: int = 8080
    visualize_cameras: bool = True

    ########### Training ###############
    max_steps: int = 30000
    eval_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 10_000, 15_000, 20_000, 25_001,30_000])
    save_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 10_000, 15_000, 20_000,25_001, 30_000])
    
    # 使用fused SSIM优化
    fused_ssim: bool = False
    pin_memory: bool = False
    
    # 保存配置
    save_only_recent_train: bool = False
    """是否只保存最近的训练检查点"""
    
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
            num_virtual_views=10,  # 恢复原始最佳配置
        )
    )

    ########### DiFix3D Integration ###############
    enable_difix3d: bool = True
    """是否启用DiFix3D处理"""
    
    difix3d_model_name: str = "nvidia/difix_ref"
    """DiFix3D模型名称或路径"""
    
    difix3d_prompt: str = "remove degradation"
    """DiFix3D处理提示词"""
    
    difix3d_blend_ratio: float = 1.0
    """DiFix3D增强图像与原图的混合比例"""
    
    difix3d_num_inference_steps: int = 1
    """DiFix3D推理步数"""
    
    difix3d_guidance_scale: float = 0.0
    """DiFix3D引导尺度"""

    difix3d_use_ref_image: bool = True
    """是否使用参考图像进行DiFix3D处理"""

    difix3d_augment_training_set: bool = True
    """是否将DiFix3D增强的虚拟视角添加到训练集"""
    
    difix3d_max_augmented_samples: int = 100
    """训练集中最多保存的增强样本数量"""
    
    difix3d_save_comparisons: bool = True
    """是否保存DiFix3D处理前后的对比图像"""
    
    ########### 混合采样策略配置（统一DiFix3D和插帧参数）###############
    # 统一的虚拟视角训练配置
    virtual_view_start_step: int = 25000
    """开始使用虚拟视角训练的步数（早期启动以获得更好效果）"""
    
    virtual_view_interval: int = 250
    """虚拟视角生成间隔（步数）"""
    
    virtual_view_poses_per_step: int = 2
    """每步生成的虚拟视角pose数量"""
    
    virtual_view_loss_weight: float = 0.1
    """虚拟视角Loss的权重，用于平衡原始Loss和虚拟视角Loss"""

    # 插帧质量阈值（PSNR差值范围判断）
    interp_quality_psnr_min: float = 4.5
    """插值帧质量评分（PSNR差值）下限，质量需大于该值"""
    interp_quality_psnr_max: float = 14.5
    """插值帧质量评分（PSNR差值）上限，质量需小于该值"""
    
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
    """是否启用深度图平滑损失"""
    depth_smooth_lambda: float = 0.1
    """深度平滑损失的权重"""
    
    ########### DiFix Enhancement Loss ###############
    enable_difix_enhancement_loss: bool = True
    """是否启用DiFix增强前后的损失计算"""
    difix_enhancement_loss_weight: float = 0.05
    """DiFix增强前后损失的权重"""
    difix_enhancement_l1_weight: float = 0.8
    """DiFix增强损失中L1损失的权重"""
    difix_enhancement_perceptual_weight: float = 0.2
    """DiFix增强损失中感知损失的权重"""
    
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
    """
    计算深度图的4邻域差分L2平滑损失
    
    Args:
        depth_map: 深度图张量 [B, H, W] 或 [B, H, W, 1]
        
    Returns:
        平滑损失值
    """
    # 确保输入格式为 [B, H, W]
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(-1)  # [B, H, W, 1] -> [B, H, W]
    
    if depth_map.dim() != 3:
        raise ValueError(f"深度图维度应为3 [B, H, W]，实际为: {depth_map.shape}")
    
    batch_size, height, width = depth_map.shape
    
    # 计算水平和垂直方向的差分
    # 水平差分：depth[i, j] - depth[i, j-1] (除了左边界)
    diff_h = depth_map[:, :, 1:] - depth_map[:, :, :-1]  # [B, H, W-1]
    
    # 垂直差分：depth[i, j] - depth[i-1, j] (除了上边界)
    diff_v = depth_map[:, 1:, :] - depth_map[:, :-1, :]  # [B, H-1, W]
    
    # 计算L2损失
    smooth_loss_h = torch.mean(diff_h ** 2)  # 水平方向平滑损失
    smooth_loss_v = torch.mean(diff_v ** 2)  # 垂直方向平滑损失
    
    # 总平滑损失
    total_smooth_loss = smooth_loss_h + smooth_loss_v
    
    return total_smooth_loss


class DiFix3DProcessor:
    """DiFix3D图像处理器 - 稳定版本，保持原始数据质量"""
    
    def __init__(self, model_name: str = "nvidia/difix_ref", device: str = "cuda", ref_image_dir: str = None):
        self.device = device
        self.model_name = model_name
        self.pipeline = None
        self.enabled = DIFIX3D_AVAILABLE
        self.ref_image_dir = ref_image_dir  # 添加ref_image_dir属性
        
        # 渐进式插值相关属性
        self.is_initialized = False
        self.quality_scorer = None
        self.available_interpolation_views = []
        self.training_psnr_mean = None
        self.training_psnr_variance = None
        
        # DiFix3D对比图像保存目录
        self.difix3d_comparison_dir = None
        
        # 用于存储虚拟视角质量评分数据
        self.virtual_view_scores = []
        # 用于存储基础打分数据（训练视角PSNR基准）
        self.baseline_scores = {}
        
        if self.enabled:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """初始化DiFix3D管道"""
        try:            
            print(f"🔄 加载DiFix3D模型: {self.model_name}")
            self.pipeline = DifixPipeline.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.pipeline.to(self.device)
            print(f"✅ DiFix3D模型加载完成")
            
        except Exception as e:
            print(f"❌ DiFix3D模型加载失败: {e}")
            self.enabled = False
    
    def _ensure_tensor_format(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        确保输入张量格式正确并返回标准化的张量和原始尺寸
        
        Args:
            image_tensor: 输入图像张量
            
        Returns:
            tuple: (标准化张量, (原始高度, 原始宽度))
        """
        # 记录原始尺寸
        if image_tensor.dim() == 4:  # [1, H, W, 3]
            original_height, original_width = image_tensor.shape[1:3]
            tensor = image_tensor.squeeze(0)  # [H, W, 3]
        elif image_tensor.dim() == 3:  # [H, W, 3]
            original_height, original_width = image_tensor.shape[:2]
            tensor = image_tensor
        else:
            raise ValueError(f"不支持的张量维度: {image_tensor.shape}")
        
        # 检查通道数
        if tensor.shape[-1] != 3:
            raise ValueError(f"不支持的通道数: {tensor.shape[-1]}, 期望3")
        
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
        """
        处理图像张量 - 稳定版本，保持原始尺寸和质量
        
        Args:
            image_tensor: 输入图像张量 [H, W, 3] 或 [1, H, W, 3] 范围[0,1]
            prompt: 处理提示词
            num_inference_steps: 推理步数
            timesteps: 时间步列表
            guidance_scale: 引导尺度
            ref_image: 可选的参考图像张量 [H, W, 3] 或 [1, H, W, 3] 范围[0,1]
            save_comparison: 是否保存处理前后对比图像
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            处理后的图像张量，保持原始尺寸和格式
        """
        if not self.enabled or self.pipeline is None:
            return image_tensor
        
        try:
            with torch.no_grad():
                # 标准化输入张量并获取原始尺寸
                input_tensor, original_size = self._ensure_tensor_format(image_tensor)
                
                
                # 确保值范围在[0,1]
                if input_tensor.max() > 1.0 or input_tensor.min() < 0.0:
                    input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
                    print(f"   ⚠️ 数值范围已调整到[0,1]")
                
                # 转换为PIL图像
                image_np = (input_tensor.cpu().numpy() * 255).astype(np.uint8)
                input_image = Image.fromarray(image_np)
                print(f"   PIL图像尺寸: {input_image.size}")  # (width, height)
                
                # 处理参考图像（如果提供）
                ref_image_pil = None
                if ref_image is not None:
                    ref_tensor, _ = self._ensure_tensor_format(ref_image)
                    # 确保值范围在[0,1]
                    if ref_tensor.max() > 1.0 or ref_tensor.min() < 0.0:
                        ref_tensor = torch.clamp(ref_tensor, 0.0, 1.0)
                    ref_np = (ref_tensor.cpu().numpy() * 255).astype(np.uint8)
                    ref_image_pil = Image.fromarray(ref_np)
                    print(f"   参考图像尺寸: {ref_image_pil.size}")
                
                # DiFix3D处理
                print(f"🔄 应用DiFix3D增强: {prompt}")
                
                # 修复：确保输入图像和参考图像尺寸完全匹配
                if ref_image_pil is not None:
                    # 使用参考图像的处理方式
                    print(f"   📷 使用参考图像进行DiFix3D处理")
                    
                    # 确保输入图像和参考图像尺寸完全匹配
                    if input_image.size != ref_image_pil.size:
                        print(f"   🔧 调整参考图像尺寸以匹配输入图像: {input_image.size} -> {ref_image_pil.size}")
                        ref_image_pil = ref_image_pil.resize(input_image.size, Image.Resampling.LANCZOS)
                    
                    # 检查图像尺寸是否合理
                    width, height = input_image.size
                    if width * height > 1000000:  # 如果图像太大，可能导致内存问题
                        print(f"   ⚠️ 图像尺寸较大 ({width}x{height})，可能导致内存问题")
                        # 可以选择缩小图像，但这里先尝试直接处理
                    
                    # 直接使用单张图像，不复制为batch
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
                        print(f"   ⚠️ 单张图像处理失败，尝试batch处理: {e}")
                        # 如果单张图像失败，尝试batch处理
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
                    # 不使用参考图像的处理方式
                    print(f"   🚫 不使用参考图像，直接进行DiFix3D处理")
                    
                    # 直接使用单张图像
                    try:
                        output_image = self.pipeline(
                            prompt,
                            image=input_image,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images[0]
                    except Exception as e:
                        print(f"   ⚠️ 单张图像处理失败，尝试batch处理: {e}")
                        # 如果单张图像失败，尝试batch处理
                        input_images = [input_image, input_image]
                        
                        output_images = self.pipeline(
                            prompt,
                            image=input_images,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images
                        output_image = output_images[0]
                
                print(f"   DiFix3D输出PIL尺寸: {output_image.size}")  # (width, height)
                
                # 转回张量
                output_np = np.array(output_image).astype(np.float32) / 255.0
                output_tensor = torch.from_numpy(output_np).to(image_tensor.device)
                
                print(f"   转换后张量形状: {output_tensor.shape}")
                
                # 如果原始输入有batch维度，添加回来
                if image_tensor.dim() == 4:
                    output_tensor = output_tensor.unsqueeze(0)  # [1, H, W, 3]
                
                print(f"✅ DiFix3D处理完成:")
                print(f"   最终输出形状: {output_tensor.shape}")
                print(f"   尺寸变化: {original_size} -> {output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]}")
                
                # 检查尺寸是否发生变化
                final_size = output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]
                if final_size != original_size:
                    print(f"   ⚠️ 尺寸发生变化: {original_size} -> {final_size}")
                
                # 保存处理前后对比图像
                return output_tensor
                
        except Exception as e:
            print(f"⚠️ DiFix3D处理失败: {e}")
            print(f"   输入张量形状: {image_tensor.shape}, 数据类型: {image_tensor.dtype}")
            print(f"   错误详情: {str(e)}")
            
            # 检查是否是einops相关的错误
            if "einops" in str(e).lower() or "rearrange" in str(e).lower():
                print(f"   🔧 检测到einops张量重排错误，尝试使用单张图像处理...")
                try:
                    # 尝试使用单张图像，但添加batch维度
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
                    
                    print(f"   ✅ 单张图像处理成功")
                    return output_tensor
                    
                except Exception as e2:
                    print(f"   ❌ 单张图像处理也失败: {e2}")
            
            # 检查是否是张量维度不匹配错误
            elif "size of tensor" in str(e).lower() and "must match" in str(e).lower():
                print(f"   🔧 检测到张量维度不匹配错误，尝试跳过DiFix3D处理...")
                print(f"   ⚠️ DiFix3D处理失败，返回原始图像")
                return image_tensor
            
            import traceback
            traceback.print_exc()
            # 返回原始输入，确保训练继续进行
            return image_tensor
    
    def load_ref_image(self, train_idx: int, trainset) -> Optional[torch.Tensor]:
        """
        根据训练集索引从ref_image目录加载参考图像
        
        Args:
            train_idx: 训练集内的索引（0, 1, 2等）
            trainset: 训练数据集，用于获取对应的COLMAP索引
            
        Returns:
            加载的参考图像张量 [H, W, 3]，如果加载失败则返回None
        """
        try:
            # 检查ref_image_dir是否设置
            if self.ref_image_dir is None:
                print(f"⚠️ ref_image_dir未设置，无法加载参考图像")
                return None
            
            # 从训练集获取对应的COLMAP索引
            try:
                train_data = trainset[train_idx]
                colmap_idx = train_data["colmap_image_id"]
                if isinstance(colmap_idx, torch.Tensor):
                    colmap_idx = colmap_idx.item()
                
                print(f"🔍 训练集索引 {train_idx} -> COLMAP索引 {colmap_idx}")
                
            except Exception as e:
                print(f"❌ 无法获取COLMAP索引 (train_idx={train_idx}): {e}")
                return None
            
            # 构建参考图像路径（使用COLMAP索引）
            ref_image_path = f"{self.ref_image_dir}/{colmap_idx:03d}.png"
            
            # 检查文件是否存在
            if not os.path.exists(ref_image_path):
                print(f"⚠️ 参考图像不存在: {ref_image_path}")
                return None
            
            # 加载图像
            from PIL import Image
            import numpy as np
            
            ref_image_pil = Image.open(ref_image_path).convert('RGB')
            ref_image_np = np.array(ref_image_pil) / 255.0  # 归一化到[0,1]
            ref_image_tensor = torch.from_numpy(ref_image_np).float().to(self.device)
            
            print(f"✅ 成功加载参考图像: {ref_image_path}, 尺寸: {ref_image_tensor.shape}")
            return ref_image_tensor
            
        except Exception as e:
            print(f"⚠️ 加载参考图像失败 (train_idx={train_idx}): {e}")
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
        """
        处理虚拟视角批次 - 选择两个视角，生成插值帧，质量评估，DiFix3D增强
        
        Args:
            trainset: 训练数据集
            camera_optimizer: BAD-Gaussians相机优化器
            rasterize_splats_fn: 3DGS渲染函数
            cfg: 配置对象
            step: 当前训练步数
            ref_image: 可选的参考图像（未使用，保持接口兼容性）
            save_comparisons: 是否保存DiFix3D处理前后对比图像
            comparison_dir: 对比图像保存目录，如果为None则使用默认目录
            
        Returns:
            增强样本列表 List[dict]
        """
        # 设置对比图像保存目录
        if comparison_dir is not None:
            self.difix3d_comparison_dir = comparison_dir
        
        if not self.enabled or self.pipeline is None:
            print("⚠️ DiFix3D未启用，无法进行批量处理")
            return []
        
        if not hasattr(trainset, '__len__') or len(trainset) == 0:
            print("⚠️ 训练集为空，无法进行批量处理")
            return []
        
        # 确保插值池已初始化
        if not self.is_initialized:
            self.initialize_interpolation_pool(trainset, rasterize_splats_fn, cfg)
            if not self.is_initialized:
                print("❌ 插值池初始化失败")
                return []
        
        print(f"🎯 步数 {step}: 开始处理虚拟视角批次")
        
        enhanced_samples = []
        quality_threshold = 0  # k <= 0 表示质量可接受
        
        try:
            # 1. 选择插值策略
            if len(self.available_interpolation_views) < 1:
                print(f"❌ 插值池视角不足 ({len(self.available_interpolation_views)} < 1)，无法进行插值")
                return []
            
            # 随机选择两个不同的训练视角作为前向插值的基础
            train_indices = torch.randperm(len(trainset))[:2]
            train_view1 = trainset[train_indices[0]]
            train_view2 = trainset[train_indices[1]]
            
            # 后向插值：选择两个不同的虚拟视角
            print(f"   🔍 虚拟视角池状态: {len(self.available_interpolation_views)} 个可用视角")
            if len(self.available_interpolation_views) >= 2:
                virtual_indices = torch.randperm(len(self.available_interpolation_views))[:2]
                virtual_view1 = self.available_interpolation_views[virtual_indices[0]]
                virtual_view2 = self.available_interpolation_views[virtual_indices[1]]
                use_backward_interpolation = True
                print(f"   ✅ 后向插值可用: 选择虚拟视角 {virtual_indices[0]} 和 {virtual_indices[1]}")
            else:
                # 如果虚拟视角不足，只使用前向插值
                use_backward_interpolation = False
                virtual_view1 = None
                virtual_view2 = None
            print(f"   前向插值基础: 训练视角 {train_indices[0]} 和 {train_indices[1]}")
            if use_backward_interpolation:
                print(f"   后向插值基础: 虚拟视角 {virtual_view1['source']} 和 {virtual_view2['source']}")
            else:
                print(f"   后向插值: 跳过（虚拟视角不足，需要至少2个）")
            
            # 🔍 调试：检查基础视角
            print(f"   🔍 基础视角调试:")
            train_pos1 = train_view1['camtoworld'][:3, 3].to(self.device)
            train_pos2 = train_view2['camtoworld'][:3, 3].to(self.device)
            if use_backward_interpolation:
                virtual_pos1 = virtual_view1['pose'][:3, 3].to(self.device)
                virtual_pos2 = virtual_view2['pose'][:3, 3].to(self.device)
                
            
            # 2. 生成前向和后向插值帧
            forward_alpha = 0.5  # 前向插值：在训练视角之间
            backward_alpha = 1.5  # 后向插值：在虚拟视角之外（向外探索）
            
            quality_scores = []
            interpolated_poses = []
            
            # 前向插值：训练视角之间
            print(f"   🎯 前向插值：训练视角之间 (α={forward_alpha})")
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
            
            # 后向插值：虚拟视角之外（向外探索）
            interpolated_pose_backward = None
            if use_backward_interpolation:
                print(f"   🎯 后向插值：虚拟视角之外向外探索 (α={backward_alpha})")
                # 确保虚拟视角数据在正确的设备上
                virtual_pose1 = virtual_view1["pose"].to(self.device)
                virtual_K1 = virtual_view1["K"].to(self.device)
                virtual_pose2 = virtual_view2["pose"].to(self.device)
                virtual_K2 = virtual_view2["K"].to(self.device)
                
                # 使用反向插值向外探索：从virtual_pose1向virtual_pose2方向延伸
                # t=1.5 意味着在virtual_pose2之外0.5倍距离的位置
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
                print(f"   🚫 后向插值：跳过（虚拟视角不足）")
            
            # 3. 生成插值帧
            print(f"   ✅ 开始生成 {cfg.virtual_view_poses_per_step} 个插值帧")
            
            for i in range(cfg.virtual_view_poses_per_step):
                # 交替生成前向和后向插值
                if i == 0:
                    # 前向插值：训练视角之间
                    interpolated_pose = interpolated_pose_forward
                    direction = "前向"
                    alpha = forward_alpha
                elif i == 1 and use_backward_interpolation:
                    # 后向插值：虚拟视角之外向外探索
                    interpolated_pose = interpolated_pose_backward
                    direction = "后向"
                    alpha = backward_alpha
                elif i == 1 and not use_backward_interpolation:
                    # 如果后向插值不可用，跳过
                    continue
                else:
                    # 如果超过2个，随机选择插值策略
                    if torch.rand(1).item() < 0.5:
                        # 训练视角之间
                        alpha = torch.rand(1).item() * 0.8 + 0.1
                        interpolated_pose, _ = se3_interpolate_to_target(
                            train_pose1, train_K1, 
                            train_pose2, train_K2, 
                            t=alpha
                        )
                        direction = "前向随机"
                    elif use_backward_interpolation:
                        # 虚拟视角之外向外探索（随机选择探索方向）
                        if torch.rand(1).item() < 0.5:
                            # 向前探索：t > 1.0
                            alpha = torch.rand(1).item() * 0.5 + 1.0  # [1.0, 1.5]
                            interpolated_pose, _ = se3_interpolate_to_target(
                                virtual_pose1, virtual_K1, 
                                virtual_pose2, virtual_K2, 
                                t=alpha
                            )
                            direction = "后向随机-向前探索"
                        else:
                            # 向后探索：t < 0.0
                            alpha = torch.rand(1).item() * 0.5 - 0.5  # [-0.5, 0.0]
                            interpolated_pose, _ = se3_interpolate_to_target(
                                virtual_pose2, virtual_K2, 
                                virtual_pose1, virtual_K1, 
                                t=alpha
                            )
                            direction = "后向随机-向后探索"
                    else:
                        # 如果后向插值不可用，使用前向插值
                        alpha = torch.rand(1).item() * 0.8 + 0.1
                        interpolated_pose, _ = se3_interpolate_to_target(
                            train_pose1, train_K1, 
                            train_pose2, train_K2, 
                            t=alpha
                        )
                        direction = "前向随机"
                
                print(f"   🎯 生成{direction}插值帧 (α={alpha:.3f})")
                
                # 🔍 调试：检查插值后的相机位置
                interp_pos = interpolated_pose[:3, 3]
                
                # 使用训练视角的内参和图像ID，确保在正确的设备上
                interp_K = train_view1["K"].unsqueeze(0).to(self.device)  # [1, 3, 3]
                interp_img_id = train_view1["image_id"].unsqueeze(0).to(self.device)
                
                # 确保插值pose在正确的设备上
                interpolated_pose = interpolated_pose.to(self.device)  # [4, 4]
                
                # 获取图像尺寸（使用训练视角的尺寸）
                
                # 检查图像形状是否正确
                if len(train_view1["image"].shape) == 4:  # [1, H, W, 3]
                    height, width = train_view1["image"].shape[1:3]  # [H, W]
                    print(f"     4D图像形状: [1, {height}, {width}, 3]")
                elif len(train_view1["image"].shape) == 3:  # [H, W, 3]
                    height, width = train_view1["image"].shape[:2]  # [H, W]
                    print(f"     3D图像形状: [{height}, {width}, 3]")
                else:
                    print(f"     ⚠️ 意外的图像形状: {train_view1['image'].shape}")
                    # 使用默认尺寸
                    height, width = 400, 600  # 假设是400x600
                    print(f"     使用默认尺寸: height={height}, width={width}")
                
                # 根据Dataset的__getitem__方法，image形状应该是[H, W, 3]
                # 所以我们应该使用[:2]来获取height, width
                if len(train_view1["image"].shape) == 3:
                    height, width = train_view1["image"].shape[:2]  # [H, W]
                    print(f"     ✅ 使用3D图像形状: [{height}, {width}, 3]")
                
                print(f"     最终提取的height: {height}, width: {width}")
                
                # 渲染插值视角（包括深度信息）
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
                
                # 确保渲染结果在正确的设备上
                renders_interp = renders_interp.to(self.device)
                if depths_interp is not None:
                    depths_interp = depths_interp.to(self.device)
                
                
                # 找到最近的训练视角作为参考
                nearest_train_idx = self._find_nearest_training_view(interpolated_pose, trainset)
                nearest_train_data = trainset[nearest_train_idx]
                
                # 渲染最近训练视角作为参考图像
                train_pose = nearest_train_data["camtoworld"].unsqueeze(0).to(self.device)
                train_K = nearest_train_data["K"].unsqueeze(0).to(self.device)
                
                # 获取训练视角ID
                if isinstance(nearest_train_data["image_id"], int):
                    train_view_id = nearest_train_data["image_id"]
                else:
                    train_view_id = nearest_train_data["image_id"].item()
                
                # 使用DiFix3D增强插值视角
                # 为插值视角选择参考图像
                ref_image_for_interp = None
                if cfg.difix3d_use_ref_image:
                    # 从预设目录加载参考图像，基于训练集索引
                    ref_image_for_interp = self.load_ref_image(nearest_train_idx, trainset)
                    
                    if ref_image_for_interp is not None:
                        print(f"   📷 成功从目录加载参考图像 (train_idx={nearest_train_idx})")
                        print(f"   🔍 ref_image_for_interp形状: {ref_image_for_interp.shape}")
                        print(f"   🔍 ref_image_for_interp设备: {ref_image_for_interp.device}")
                    else:
                        print(f"   ⚠️ 无法从目录加载参考图像 (train_idx={nearest_train_idx})，将不使用参考图像")
                else:
                    print(f"   🚫 不使用参考图像进行DiFix3D处理")
                
                
                # 确保渲染结果格式正确，处理RGB+ED模式的4通道输出
                if renders_interp[0].dim() != 3:
                    print(f"     ⚠️ 渲染结果维度不正确，跳过DiFix3D处理")
                    enhanced_interp = renders_interp[0]  # 直接使用原始渲染结果
                elif renders_interp[0].shape[-1] == 4:
                    # RGB+ED模式：只取前3个通道（RGB）用于DiFix3D处理
                    print(f"     🔧 RGB+ED模式：提取RGB通道用于DiFix3D处理")
                    rgb_interp = renders_interp[0][:, :, :3]  # [H, W, 3]
                    enhanced_interp = self.process_image(
                        rgb_interp,  # [H, W, 3]
                        prompt=cfg.difix3d_prompt,
                        num_inference_steps=cfg.difix3d_num_inference_steps,
                        timesteps=[199],
                        guidance_scale=cfg.difix3d_guidance_scale,
                        ref_image=ref_image_for_interp,
                        save_comparison=cfg.difix3d_save_comparisons,  # 根据配置决定是否保存
                        save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{i}_rgb_ed"
                    )
                elif renders_interp[0].shape[-1] == 3:
                    # 标准RGB模式
                    enhanced_interp = self.process_image(
                        renders_interp[0],  # [H, W, 3]
                        prompt=cfg.difix3d_prompt,
                        num_inference_steps=cfg.difix3d_num_inference_steps,
                        timesteps=[199],
                        guidance_scale=cfg.difix3d_guidance_scale,
                        ref_image=ref_image_for_interp,
                        save_comparison=cfg.difix3d_save_comparisons,  # 根据配置决定是否保存
                        save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{i}_rgb"
                    )
                else:
                    print(f"     ⚠️ 渲染结果通道数不正确，跳过DiFix3D处理")
                    enhanced_interp = renders_interp[0]  # 直接使用原始渲染结果
                
                # 确保所有张量都在正确的设备上（在质量评分计算之前）
                interpolated_pose_device = interpolated_pose.to(self.device)
                interp_K_device = interp_K[0].to(self.device)
                interp_img_id_device = interp_img_id[0].to(self.device)
                
                # 计算质量评分（确保两个输入都是RGB格式）
                try:
                    # 确保用于质量评分的原始图像也是RGB格式
                    if renders_interp[0].shape[-1] == 4:
                        original_rgb_for_score = renders_interp[0][:, :, :3]  # [H, W, 3]
                    else:
                        original_rgb_for_score = renders_interp[0]  # [H, W, 3]
                    
                    _, quality_score = self.quality_scorer.score_pseudo_view(
                        original_rgb_for_score, enhanced_interp
                    )
                    print(f"   📊 插值帧质量评分: k={quality_score:.4f}")
                    
                    # 保存虚拟视角质量评分数据
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
                    print(f"   💾 已保存虚拟视角质量评分数据")
                    
                except Exception as e:
                    print(f"   ⚠️ 质量评分计算失败: {e}")
                    quality_score = 0.0  # 默认质量评分
                    
                    # 即使评分失败也保存数据
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
                    print(f"   💾 已保存虚拟视角质量评分数据（评分失败）")
                
                # 确保所有张量都在正确的设备上
                enhanced_interp_device = enhanced_interp.to(self.device)
                
                # 添加到增强样本列表（用于后续重新渲染和损失计算）
                # ✅ 不保存original_image，而是保存渲染参数，在每个训练步重新渲染
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
                
                # 调试：检查sample中所有张量的设备
                print(f"   🔍 样本{i}张量设备检查:")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: 设备={value.device}, 形状={value.shape}")
                    else:
                        print(f"     {key}: 非张量={type(value)}")
                should_add_to_pool = False
                # 前向插帧：质量评分判断 (PSNR差值)
                # 当PSNR差值在配置的范围内时，认为质量合格
                if (quality_score < cfg.interp_quality_psnr_max) and (quality_score > cfg.interp_quality_psnr_min):
                    should_add_to_pool = True
                    enhanced_samples.append(sample)
                    print(f"   ✅ 前向插帧质量合格 (PSNR差值={quality_score:.4f}，范围 {cfg.interp_quality_psnr_min}~{cfg.interp_quality_psnr_max})，加入插值池")
                else:
                    print(f"   ❌ 前向插帧质量不合格 (PSNR差值={quality_score:.4f}，超出范围 {cfg.interp_quality_psnr_min}~{cfg.interp_quality_psnr_max})，不加入插值池")
                
                # 只有质量合格的插帧才添加到可用插值视角池
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
                    print(f"   🎯 插帧已加入池中，当前池大小: {len(self.available_interpolation_views)}")
                else:
                    print(f"   🚫 插帧未加入池中，当前池大小: {len(self.available_interpolation_views)}")
                
                print(f"   ✅ 插值帧 {i+1}/{cfg.virtual_view_poses_per_step} 处理完成 (α={alpha:.3f}, 参考训练视角={nearest_train_idx})")
                print(f"   🔍 虚拟视角池更新: 当前有 {len(self.available_interpolation_views)} 个视角")
            
        except Exception as e:
            print(f"❌ 虚拟视角批次处理失败: {e}")
            return []
        
        # 打印处理结果
        if enhanced_samples:
            print(f"🎯 步数 {step} 虚拟视角批次处理完成！")
            print(f"   成功生成 {len(enhanced_samples)} 个增强视角")
            print(f"   插值视角池现包含 {len(self.available_interpolation_views)} 个视角")
        else:
            print(f"⚠️ 步数 {step} 虚拟视角批次处理失败，没有成功生成的视角")
        
        return enhanced_samples
    
    def _find_nearest_training_view(self, target_pose: torch.Tensor, trainset) -> int:
        """
        找到与目标pose最近的训练视角
        
        Args:
            target_pose: 目标pose [4, 4]
            trainset: 训练数据集
            
        Returns:
            最近训练视角的索引
        """
        min_distance = float('inf')
        nearest_idx = 0
        
        target_position = target_pose[:3, 3]  # [3]
        
        for i in range(len(trainset)):
            train_data = trainset[i]
            train_pose = train_data["camtoworld"].to(self.device)  # [4, 4]
            train_position = train_pose[:3, 3]  # [3]
            
            # 计算欧几里得距离
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
        一次性初始化插值池和PSNR基准
        
        Args:
            trainset: 训练数据集
            rasterize_splats_fn: 3DGS渲染函数
            cfg: 配置对象
        """
        if self.is_initialized:
            print("🔄 插值池已初始化，跳过重复初始化")
            return
        
        print("🚀 开始初始化插值池和PSNR基准...")
        
        # 1. 初始化VirtualViewQualityScorer
        self.quality_scorer = VirtualViewQualityScorer()
        print("✅ VirtualViewQualityScorer初始化完成")
        
        # 2. 处理训练视角，计算固定PSNR基准
        print("📊 处理训练视角并计算PSNR基准...")
        
        # 选择前3个训练视角作为基准
        num_training_views = min(3, len(trainset))
        all_original_views = []
        all_difix_views = []
        
        for i in range(num_training_views):
            try:
                # 获取训练视角数据
                train_data = trainset[i]
                train_pose = train_data["camtoworld"].unsqueeze(0).to(self.device)  # [1, 4, 4]
                train_K = train_data["K"].unsqueeze(0).to(self.device)  # [1, 3, 3]
                train_image = train_data["image"].unsqueeze(0).to(self.device) / 255.0  # [1, H, W, 3]
                
                # 确保image_id是张量格式
                if isinstance(train_data["image_id"], int):
                    train_img_id = torch.tensor([train_data["image_id"]], device=self.device)
                else:
                    train_img_id = train_data["image_id"].unsqueeze(0).to(self.device)
                
                height, width = train_image.shape[1:3]
                
                # 渲染训练视角
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
                
                # 使用DiFix3D增强训练视角
                print(f"   🎨 开始DiFix3D处理训练视角 {i+1}...")
                
                # 为训练视角选择参考图像：使用另一个训练视角的渲染作为参考
                ref_image_for_training = None
                if cfg.difix3d_use_ref_image:
                    # 选择另一个训练视角的原始渲染作为参考
                    ref_idx = (i + 1) % num_training_views
                    if ref_idx != i:  # 确保不是同一个视角
                        # 渲染参考视角
                        ref_train_data = trainset[ref_idx]
                        ref_train_pose = ref_train_data["camtoworld"].unsqueeze(0).to(self.device)
                        ref_train_K = ref_train_data["K"].unsqueeze(0).to(self.device)
                        
                        # 确保image_id是张量格式
                        if isinstance(ref_train_data["image_id"], int):
                            ref_train_img_id = torch.tensor([ref_train_data["image_id"]], device=self.device)
                        else:
                            ref_train_img_id = ref_train_data["image_id"].unsqueeze(0).to(self.device)
                        
                        # 渲染参考视角
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
                        
                        ref_image_for_training = ref_renders[0].to(self.device)  # [H, W, 3] - 确保在正确设备上
                        print(f"   📷 使用训练视角 {ref_idx+1} 的原始渲染作为参考图像")
                    else:
                        print(f"   🚫 无法选择不同的训练视角作为参考，跳过参考图像")
                else:
                    print(f"   🚫 不使用参考图像进行DiFix3D处理")
                
                enhanced_train = self.process_image(
                    renders_train[0],  # [H, W, 3]
                    prompt=cfg.difix3d_prompt,
                    num_inference_steps=cfg.difix3d_num_inference_steps,
                    timesteps=[199],
                    guidance_scale=cfg.difix3d_guidance_scale,
                    ref_image=ref_image_for_training,  # 使用选择的参考图像
                    save_comparison=False
                )
                print(f"   🎨 DiFix3D处理完成")
                
                # 收集用于PSNR计算的数据
                all_original_views.append(renders_train[0])
                all_difix_views.append(enhanced_train)
                
                # 🔍 调试：检查图像是否相同
                print(f"   🔍 训练视角 {i+1} 调试信息:")
                print(f"     原始图像形状: {renders_train[0].shape}, 范围: [{renders_train[0].min():.4f}, {renders_train[0].max():.4f}]")
                print(f"     DiFix图像形状: {enhanced_train.shape}, 范围: [{enhanced_train.min():.4f}, {enhanced_train.max():.4f}]")
                
                # 计算MSE来检查图像差异
                mse = torch.mean((renders_train[0] - enhanced_train) ** 2)
                print(f"     图像MSE: {mse.item():.8f}")
                
                if mse < 1e-8:
                    print(f"     ⚠️ 警告：原始图像和DiFix图像几乎完全相同！")
                    print(f"       这可能意味着DiFix3D处理没有生效")
                else:
                    print(f"     ✅ 图像有差异，DiFix3D处理生效")
                
                # 直接添加到可用插值视角池（训练视角无需评分）
                self.available_interpolation_views.append({
                    "pose": train_pose[0],  # [4, 4]
                    "K": train_K[0],  # [3, 3]
                    "image_id": train_img_id[0],
                    "enhanced_image": enhanced_train,
                    "source": f"training_view_{i}"
                })
                
                print(f"   ✅ 训练视角 {i+1}/{num_training_views} 处理完成")
                
            except Exception as e:
                print(f"   ❌ 训练视角 {i} 处理失败: {e}")
                continue
        
        if len(all_original_views) == 0:
            print("❌ 没有成功处理的训练视角，无法计算PSNR基准")
            return
        
        # 3. 计算固定PSNR基准（不会在后续过程中更新）
        try:
            self.training_psnr_mean, self.training_psnr_variance = self.quality_scorer.evaluate_training_views(
                all_original_views, all_difix_views
            )
            
            # 检查PSNR值的有效性
            if np.isinf(self.training_psnr_mean) or np.isnan(self.training_psnr_mean):
                print(f"❌ PSNR均值为无效值: {self.training_psnr_mean}")
                print(f"   这通常意味着DiFix3D处理后的图像与原始图像完全相同")
                print(f"   请检查DiFix3D处理是否正常工作")
                raise ValueError("PSNR均值为inf，DiFix3D处理可能没有生效")
            
            if np.isinf(self.training_psnr_variance) or np.isnan(self.training_psnr_variance):
                print(f"❌ PSNR方差为无效值: {self.training_psnr_variance}")
                print(f"   这通常意味着所有图像的PSNR值都相同（都是inf）")
                raise ValueError("PSNR方差为nan，所有图像可能完全相同")
            
            print(f"📊 固定PSNR基准计算完成: 均值={self.training_psnr_mean:.4f}, 方差={self.training_psnr_variance:.4f}")
            
            # 保存基础打分数据
            self.baseline_scores = {
                "training_psnr_mean": float(self.training_psnr_mean),
                "training_psnr_variance": float(self.training_psnr_variance),
                "training_views_count": len(all_original_views),
                "timestamp": time.time()
            }
            print(f"💾 已保存基础打分数据: 均值={self.training_psnr_mean:.4f}, 方差={self.training_psnr_variance:.4f}")
            
        except Exception as e:
            print(f"❌ PSNR基准计算失败: {e}")
            print(f"   原因分析：")
            print(f"   1. DiFix3D处理可能没有生效，返回了原始图像")
            print(f"   2. 图像数据可能有问题")
            print(f"   3. 请检查DiFix3D模型是否正确加载")
            raise e
        print(f"🔄 可用插值视角池初始化完成，包含 {len(self.available_interpolation_views)} 个训练视角")
        
        # 标记为已初始化
        self.is_initialized = True
        print("✅ 插值池和PSNR基准初始化完成")
    

class DeblurDiFix3DRunner(Runner):
    """BAD-Gaussians去模糊 + DiFix3D训练引擎"""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: DeblurDiFix3DConfig) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # 设置输出目录
        self.result_dir = cfg.result_dir  # 保存原始路径
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        # DiFix3D对比图像保存目录
        self.difix3d_comparison_dir = f"{cfg.result_dir}/difix3d_comparisons"
        os.makedirs(self.difix3d_comparison_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # 从data_dir中提取scene_name，用于构建ref_image路径
        self.scene_name = Path(cfg.data_dir).name
        self.ref_image_dir = f"{cfg.data_dir}/ref_image"
        print(f"🔍 场景名称: {self.scene_name}")
        print(f"🔍 参考图像目录: {self.ref_image_dir}")

        # 加载数据
        self.parser = ColmapParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,  # 强制使用原始图像，不使用任何下采样
            normalize=True,
            scale_factor=cfg.scale_factor,
            # 强制禁用自动下采样
            downscale_rounding_mode="round",  # 使用round而不是floor
        )
        # 训练索引配置：将 CLI/配置中的 train_indices 传递给解析器
        self.parser.train_indices = cfg.train_indices
        if cfg.train_indices is not None:
            print(f"[Dataset] 使用配置的训练索引: {cfg.train_indices}")
        
        # 调试：检查ColmapParser的配置
        print(f"🔍 ColmapParser配置检查:")
        print(f"   data_factor: {cfg.data_factor}")
        print(f"   scale_factor: {cfg.scale_factor}")
        print(f"   downscale_rounding_mode: {self.parser.downscale_rounding_mode}")
        print(f"   parser.factor: {self.parser.factor}")
        if hasattr(self.parser, '_downscale_factor'):
            print(f"   parser._downscale_factor: {self.parser._downscale_factor}")
        
        # 检查图像路径
        print(f"🔍 图像路径检查:")
        if hasattr(self.parser, 'image_paths') and len(self.parser.image_paths) > 0:
            sample_path = Path(self.parser.image_paths[0])
            print(f"   样本图像路径: {sample_path}")
            if sample_path.exists():
                img = Image.open(sample_path)
                print(f"   样本图像尺寸: {img.size}")
            else:
                print(f"   ⚠️ 样本图像路径不存在!")
        else:
            print(f"   ⚠️ 没有找到图像路径!")

        self.trainset = DeblurNerfDataset(self.parser, split="train")
        
        # 初始化相机轨迹生成器
        self.trajectory_generator = generate_camera_trajectory
        self.valset = DeblurNerfDataset(self.parser, split="val")
        self.testset = DeblurNerfDataset(self.parser, split="test")
        self.quality_scorer = VirtualViewQualityScorer(device=self.device)
        print(f"✅ 虚拟视角质量打分模型已初始化")
        # 初始化DiFix3D处理器
        if cfg.enable_difix3d:
            print("🎨 初始化DiFix3D处理器...")
            self.difix3d_processor = DiFix3DProcessor(
                model_name=cfg.difix3d_model_name,
                device=self.device,
                ref_image_dir=self.ref_image_dir
            )
            if self.difix3d_processor.enabled:
                print(f"✅ DiFix3D处理器初始化成功")
            else:
                print(f"⚠️ DiFix3D处理器初始化失败，将禁用DiFix3D功能")
                cfg.enable_difix3d = False
        else:
            self.difix3d_processor = None
            print("🚫 DiFix3D功能已禁用")

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        # 初始化3D高斯点
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

        # 密集化策略
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # BAD-Gaussians相机优化器
        self.pose_optimizers = []
        # 计算总相机数量，包括训练集、验证集和测试集
        total_cameras = len(self.trainset) + (len(self.valset) if self.valset else 0) + (len(self.testset) if self.testset else 0)
        self.camera_optimizer: BadCameraOptimizer = self.cfg.camera_optimizer.setup(
            num_cameras=total_cameras,
            device=self.device,
        )
        camera_optimizer_param_groups = {}
        # 处理DDP包装的情况
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

        # 外观优化器
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

        # 双边网格
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

        # 评估指标
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        
        # 初始化VGG感知损失模型（预创建，避免重复创建）
        self.perceptual_loss = VGG16PerceptualLoss(
            feature_layer='relu2_2',
            device=self.device,
            enable_timing=False
        )
        self.dists_loss = VGG16DISTSLoss(
            device=self.device,
            enable_timing=False
        )

        # 初始化查看器
        if not cfg.disable_viewer:
            import nerfview
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        self.cfg_to_save = cfg
        
        # 用于存储虚拟相机位置数据
        self.virtual_camera_batches = []
        # 用于存储所有训练相机位置数据
        self.all_train_cameras = None
        
        # 用于存储虚拟视角质量评分数据
        self.virtual_view_scores = []
        # 用于存储基础打分数据（训练视角PSNR基准）
        self.baseline_scores = {}
        
        # 初始化混合采样策略（按需插帧模式）
        # 混合采样策略状态跟踪
        self.hybrid_sampling_initialized = False
        
    def collect_train_camera_data(self):
        """
        收集所有训练相机位置数据
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
                print(f"📊 收集到 {len(train_cameras)} 个训练相机位置")
            else:
                print("⚠️ 无法从训练数据集获取相机位置")


    def collect_virtual_camera_data(self, camera_poses: torch.Tensor = None, enhanced_samples: List[dict] = None, step: int = None, source: str = "unknown"):
        """
        收集虚拟相机位置数据（统一接口）
        
        Args:
            camera_poses: 相机poses [N, 4, 4] (BAD-Gaussians使用)
            enhanced_samples: 增强样本列表 (DiFix3D使用)
            step: 当前步数（可选）
            source: 数据来源（"BAD-Gaussians" 或 "DiFix3D"）
        """
        if camera_poses is not None and len(camera_poses) > 0:
            # BAD-Gaussians虚拟相机
            self.virtual_camera_batches.append(camera_poses.detach().clone())
            step_info = f"步数{step}: " if step is not None else ""
            print(f"📊 {step_info}收集到{source}虚拟相机 {len(camera_poses)} 个")
        elif enhanced_samples:
            # DiFix3D增强虚拟相机
            virtual_poses = []
            print(f"🔍 调试enhanced_samples设备信息:")
            for i, sample in enumerate(enhanced_samples):
                pose = sample["pose"]  # [4, 4] - 不包含batch维度
                print(f"   样本{i}: pose设备={pose.device}, 形状={pose.shape}, 期望设备={self.device}")
                # 确保pose在正确的设备上
                if pose.device != self.device:
                    print(f"   🔧 样本{i}: 将pose从{pose.device}移动到{self.device}")
                    pose = pose.to(self.device)
                virtual_poses.append(pose.unsqueeze(0))  # 添加batch维度 [1, 4, 4]
            
            if virtual_poses:
                print(f"🔍 调试virtual_poses设备信息:")
                for i, pose in enumerate(virtual_poses):
                    print(f"   virtual_poses[{i}]: 设备={pose.device}, 形状={pose.shape}")
                
                try:
                    virtual_cameras_batch = torch.cat(virtual_poses, dim=0)  # [N, 4, 4]
                    self.virtual_camera_batches.append(virtual_cameras_batch)
                    print(f"📊 收集到{source}虚拟相机 {len(virtual_poses)} 个")
                    print(f"   📊 当前总虚拟相机批次数量: {len(self.virtual_camera_batches)}")
                    total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
                    print(f"   📊 当前总虚拟相机数量: {total_virtual_cameras}")
                except Exception as e:
                    print(f"❌ torch.cat失败: {e}")
                    print(f"   所有张量设备: {[pose.device for pose in virtual_poses]}")
                    raise e


    def train(self):
        """主训练循环 - 完全基于simple_trainer_deblur.py"""
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

        # 在训练开始前收集训练相机数据
        if world_rank == 0:
            print("📊 开始收集训练相机位置数据...")
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
            
            # 📊 注释掉BAD-Gaussians虚拟相机收集，只保留DiFix3D的虚拟相机
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
            
            # 🎯 计算深度平滑损失 (每个step都执行)
            depth_smooth_loss_value = 0.0

            # BAD-Gaussians: average the virtual views
            colors = colors.mean(0)[None]
            
            # 🎯 虚拟视角训练策略
            virtual_view_loss_to_add = 0.0  

            # 🆕 混合采样策略：统一的虚拟视角训练
            print(f"🔍 检查混合采样条件: step={step}, virtual_view_start_step={cfg.virtual_view_start_step}, enable_difix3d={cfg.enable_difix3d}, difix3d_processor={self.difix3d_processor is not None}, step%interval={step % cfg.virtual_view_interval}")
            if step >= cfg.virtual_view_start_step and cfg.enable_difix3d and self.difix3d_processor is not None and step % cfg.virtual_view_interval == 0:
                if step == cfg.virtual_view_start_step:
                    print(f"🎯 步数 {step}: 首次启用混合采样策略虚拟视角训练")
                else:
                    print(f"🎯 步数 {step}: 继续混合采样策略虚拟视角训练")
                
                # 🆕 使用新的虚拟视角批次处理策略
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
                    # 🎯 将多个增强样本添加到类属性中
                    if not hasattr(self, 'enhanced_data'):
                        self.enhanced_data = []
                    elif not isinstance(self.enhanced_data, list):
                        self.enhanced_data = []  # 重置为列表格式
                    
                    # 逐个添加新样本，限制总数量
                    max_samples = getattr(cfg, 'difix3d_max_augmented_samples', 100)
                    for enhanced_sample in enhanced_samples:
                        # 限制增强数据数量，保持最新的样本
                        if len(self.enhanced_data) >= max_samples:
                            self.enhanced_data.pop(0)  # 移除最旧的样本
                        
                        self.enhanced_data.append(enhanced_sample)
                    
                    # 📊 收集虚拟相机位置数据
                    self.collect_virtual_camera_data(enhanced_samples=enhanced_samples, source="DiFix3D-Progressive")
                    
                    print(f"🎯 渐进式插值完成:")
                    print(f"   本次生成样本数: {len(enhanced_samples)}")
                    print(f"   当前总增强样本数: {len(self.enhanced_data)}")
                    print(f"   📊 当前虚拟相机批次数量: {len(self.virtual_camera_batches)}")
                    total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
                    print(f"   📊 当前虚拟相机总数量: {total_virtual_cameras}")
                    for i, sample in enumerate(enhanced_samples):
                        quality_score = sample.get('quality_score', 'N/A')
                        if isinstance(quality_score, (int, float)):
                            print(f"   样本 {i}: 图像ID={sample['image_id'].item()}, 质量评分={quality_score:.4f}")
                        else:
                            print(f"   样本 {i}: 图像ID={sample['image_id'].item()}, 质量评分={quality_score}")
                else:
                    print("⚠️ 渐进式插值失败，没有生成新样本")
                    
            # 🎯 虚拟视角训练开始步数及之后：计算虚拟视角Loss
            if step >= cfg.virtual_view_start_step and hasattr(self, 'enhanced_data'):
                # 🆕 随机选择一个增强样本进行Loss计算（减少计算开销）
                if isinstance(self.enhanced_data, list) and len(self.enhanced_data) > 0:
                    # 随机选择一个样本
                    import random
                    sample = random.choice(self.enhanced_data)
                    
                    loss_virtual_sample = 0.0
                    
                    # 🎯 关键：重新渲染虚拟视角，获得连接到当前3D高斯场的梯度
                    virtual_pose = sample["pose"].unsqueeze(0).to(device)  # [1, 4, 4]
                    virtual_K = sample["K"].unsqueeze(0).to(device)  # [1, 3, 3]
                    virtual_image_id = sample["image_id"].unsqueeze(0).to(device)
                    virtual_width = sample["width"]
                    virtual_height = sample["height"]
                    
                    # 重新渲染虚拟视角（包含梯度，连接到当前训练步的3D高斯场）
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
                    
                    # 提取RGB和深度信息
                    if renders_virtual.shape[-1] == 4:
                        colors_virtual, depths_virtual = renders_virtual[..., 0:3], renders_virtual[..., 3:4]
                    else:
                        colors_virtual, depths_virtual = renders_virtual, None
                    
                    # 应用随机背景（如果启用）
                    if cfg.random_bkgd:
                        colors_virtual = colors_virtual + bkgd * (1.0 - alphas_virtual)
                    
                    # 获取DiFix增强后的图像作为监督信号（无梯度）
                    enhanced_image = sample["enhanced_image"].to(device)  # [H, W, 3]
                    if enhanced_image.dim() == 3:
                        enhanced_image = enhanced_image.unsqueeze(0)  # [1, H, W, 3]
                    
                    # 🎯 计算DiFix蒸馏损失：将增强图像的信息蒸馏到渲染结果中
                    difix_distillation_loss = 0.0
                    if cfg.enable_difix_enhancement_loss:
                        # L1损失
                        difix_l1_loss = F.l1_loss(colors_virtual, enhanced_image)
                        
                        # SSIM损失
                        difix_ssim_loss = 1.0 - self.ssim(
                            colors_virtual.permute(0, 3, 1, 2), 
                            enhanced_image.permute(0, 3, 1, 2)
                        )
                        
                        # DISTS感知损失
                        difix_dists_loss = self.dists_loss(colors_virtual, enhanced_image)
                        difix_perc_loss = self.perceptual_loss(colors_virtual, enhanced_image)
                        # 组合蒸馏损失
                        difix_distillation_loss = (
                            difix_l1_loss * cfg.difix_enhancement_l1_weight +
                            difix_dists_loss * 0.01
                        )
                        
                        loss_virtual_sample += difix_distillation_loss
                    
                    # 深度平滑损失（如果启用）
                    if cfg.enable_depth_smooth_loss and depths_virtual is not None:
                        depth_smooth_loss_virtual = depth_smooth_loss_4neighbor(depths_virtual)
                        loss_virtual_sample += depth_smooth_loss_virtual * cfg.depth_smooth_lambda
                    
                    # 应用权重
                    virtual_view_loss_to_add = cfg.virtual_view_loss_weight * loss_virtual_sample
                    
                    # 添加调试信息
                    if step % 100 == 0:  # 每100步打印一次详细信息
                        print(f"🔍 虚拟视角Loss调试 (步数{step}):")
                        print(f"   可用样本数量: {len(self.enhanced_data)}")
                        print(f"   当前选择样本ID: {sample.get('image_id', 'unknown')}")
                        print(f"   当前样本质量评分: {sample.get('quality_score', 'N/A'):.4f}")
                        print(f"   虚拟Loss: {loss_virtual_sample:.6f}")
                        print(f"   加权后Loss: {virtual_view_loss_to_add:.6f}")
                        print(f"   权重: {cfg.virtual_view_loss_weight}")
                        if cfg.enable_depth_smooth_loss:
                            print(f"   🔍 深度平滑损失已启用，权重: {cfg.depth_smooth_lambda}")
                    

            else:
                # 虚拟视角训练开始步数之前，虚拟视角Loss为0
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

            # 🎯 添加深度平滑损失
            if cfg.enable_depth_smooth_loss and step >= 25000 and depths is not None:
                depth_smooth_loss_value = depth_smooth_loss_4neighbor(depths)
                loss += depth_smooth_loss_value * cfg.depth_smooth_lambda
                if step % 100 == 0:  # 每100步打印一次详细信息
                    print(f"🔍 深度平滑损失权重: {cfg.depth_smooth_lambda}, 加权后损失: {depth_smooth_loss_value * cfg.depth_smooth_lambda:.6f}")

            # 🎯 关键：在所有loss计算完成后，添加虚拟视角Loss
            loss += virtual_view_loss_to_add
            
            # 如果启用了虚拟视角训练，打印总Loss信息
            if virtual_view_loss_to_add > 0:
                print(f"🎯 最终Loss: 基础={loss.item() - virtual_view_loss_to_add:.4f}, 虚拟视角={virtual_view_loss_to_add:.4f}, 总计={loss.item():.4f}")

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
                # 处理DDP包装的情况
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

        # Get estimated trajectory
        # 处理DDP包装的情况
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
