#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import math
from typing import Optional, Tuple, List

def _load_vgg16_pretrained() -> nn.Module:
    """兼容新旧 torchvision 的 VGG16 预训练权重加载。"""
    try:
        # torchvision >= 0.13
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        # 旧版本
        vgg = models.vgg16(pretrained=True)
    return vgg

class VGG16PerceptualLoss(nn.Module):
    """
    VGG-16 Perceptual Loss
    使用VGG-16的某层特征计算感知损失
    """

    def __init__(
        self,
        feature_layer: str = 'relu2_2',
        normalize: bool = True,
        resize_input: bool = True,
        requires_grad: bool = False,
        device: Optional[str] = 'cuda',     # 强制使用 GPU；若想自动，可设为 None
        dtype: torch.dtype = torch.float32, # 统一 dtype
        enable_timing: bool = True,         # 是否启用时间统计
    ):
        """
        Args:
            feature_layer: 目标特征层 ('relu1_2','relu2_2','relu3_3','relu4_3','relu5_1')
            normalize: 是否做 ImageNet 标准化
            resize_input: 是否缩放到 224x224
            requires_grad: VGG 特征是否参与反传
            device: 计算设备；默认 'cuda'。若为 None 且有 CUDA，则自动用 'cuda's
            dtype: 计算精度（默认 float32）
            enable_timing: 是否启用时间统计
        """
        super().__init__()

        # 设备选择
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("要求在 GPU 上运行，但当前环境未检测到 CUDA。")

        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_timing = enable_timing

        # 支持的层
        self.feature_layers = {
            'relu1_2': 3,   # features[0..4]
            'relu2_2': 8,   # features[0..9]
            'relu3_3': 15,  # features[0..16]
            'relu4_3': 22,  # features[0..23]
            'relu5_1': 25,  # features[0..26]
        }
        if feature_layer not in self.feature_layers:
            raise ValueError(f"不支持的特征层: {feature_layer}. 支持: {list(self.feature_layers.keys())}")

        self.feature_layer_idx = self.feature_layers[feature_layer]
        self.feature_layer_name = feature_layer
        
        # 加载 VGG16 并构造特征提取器
        vgg = _load_vgg16_pretrained()
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[: self.feature_layer_idx + 1])

        # 冻结与训练模式
        for p in self.feature_extractor.parameters():
            p.requires_grad = requires_grad
        self.feature_extractor.eval()

        # 标准化参数（作为 buffer，随 .to(device) 移动）
        self.normalize = normalize
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=self.dtype).view(1, 3, 1, 1))
            self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225], dtype=self.dtype).view(1, 3, 1, 1))
        else:
            # 也注册占位，便于类型一致
            self.register_buffer('mean', torch.zeros(1, 3, 1, 1, dtype=self.dtype))
            self.register_buffer('std',  torch.ones(1, 3, 1, 1, dtype=self.dtype))

        self.resize_input = resize_input

        # 把整个模块（参数+buffer）搬到设备/类型
        self.to(self.device, dtype=self.dtype)

        # 获取输出通道/尺寸信息
        self._get_layer_info()

    @torch.no_grad()
    def _get_layer_info(self):
        """获取目标层的通道数/尺寸信息（在 self.device 上求一次前向）"""
        test_input = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        x = test_input
        if self.normalize:
            x = (x - self.mean) / self.std
        features = self.feature_extractor(x)
        self.C_j = features.shape[1]
        self.H_j = features.shape[2]
        self.W_j = features.shape[3]
        # 可按需打印
        # print(f"[{self.feature_layer_name}] C:{self.C_j} H:{self.H_j} W:{self.W_j}")

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        预处理输入到 [B, C, H, W]，放到同一 device/dtype，按需 resize/normalize
        """
        # 移动设备/类型
        x = x.to(self.device, dtype=self.dtype, non_blocking=True)

        # 保证通道维在前
        if x.ndim == 4 and x.shape[-1] == 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()

        # 归一化到 [0,1]（若看起来像 0..255）
        if x.max() > 1.0:
            x = x / 255.0

        # 尺寸到 224x224
        if self.resize_input and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # ImageNet 标准化
        if self.normalize:
            x = (x - self.mean) / self.std

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 VGG 特征（在 self.device 上）"""
        if self.enable_timing:
            feature_start_time = time.time()
        
        x = self.preprocess_input(x)
        
        if self.enable_timing:
            preprocess_time = time.time() - feature_start_time
            vgg_start_time = time.time()
        
        # 如果不需要对 VGG 求梯度，可以用 no_grad 提速/省显存
        if not any(p.requires_grad for p in self.feature_extractor.parameters()):
            with torch.no_grad():
                features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
        
        if self.enable_timing:
            vgg_time = time.time() - vgg_start_time
            total_feature_time = time.time() - feature_start_time
            print(f"🔍 VGG特征提取时间统计:")
            print(f"   预处理时间: {preprocess_time:.4f}s")
            print(f"   VGG推理时间: {vgg_time:.4f}s")
            print(f"   总特征提取时间: {total_feature_time:.4f}s")
        
        return features

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        计算 perceptual loss： L2(φ(pred) - φ(target)) / (C*H*W)
        """
        if self.enable_timing:
            forward_start_time = time.time()
        
        # 统一设备/类型
        pred = pred.to(self.device, dtype=self.dtype, non_blocking=True)
        target = target.to(self.device, dtype=self.dtype, non_blocking=True)

        # 提取特征
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        if self.enable_timing:
            feature_time = time.time() - forward_start_time
            loss_calc_start_time = time.time()

        # 计算Loss
        diff = pred_features - target_features
        # [B]
        l2_sq = torch.sum(diff * diff, dim=(1, 2, 3))
        norm = float(self.C_j * self.H_j * self.W_j)
        loss = torch.mean(l2_sq / norm)
        
        if self.enable_timing:
            loss_calc_time = time.time() - loss_calc_start_time
            total_forward_time = time.time() - forward_start_time
            print(f"🔍 Perceptual Loss计算时间统计:")
            print(f"   特征提取时间: {feature_time:.4f}s")
            print(f"   Loss计算时间: {loss_calc_time:.4f}s")
            print(f"   总前向时间: {total_forward_time:.4f}s")
            print(f"   特征形状: {pred_features.shape}")
    
        if return_features:
            return loss, (pred_features, target_features)
        return loss

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

    def get_timing_stats(self) -> dict:
        """获取时间统计信息（如果启用）"""
        if not self.enable_timing:
            return {"timing_enabled": False}
        
        return {
            "timing_enabled": True,
            "feature_layer": self.feature_layer_name,
            "device": str(self.device),
            "dtype": str(self.dtype)
        }

class VGG16PerceptualLossWithMultipleLayers(nn.Module):
    """
    同时使用多个 VGG 层的 Perceptual Loss（全部在同一 GPU 上）
    """
    def __init__(
        self,
        feature_layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3'],
        weights: Optional[List[float]] = None,
        normalize: bool = True,
        resize_input: bool = True,
        requires_grad: bool = False,
        device: Optional[str] = 'cuda',
        dtype: torch.dtype = torch.float32,
        enable_timing: bool = True,
    ):
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights if weights is not None else [1.0] * len(feature_layers)
        if len(self.weights) != len(self.feature_layers):
            raise ValueError("特征层数量与权重数量必须一致")

        # 在同一 device/dtype 上构建所有子损失
        self.loss_modules = nn.ModuleList([
            VGG16PerceptualLoss(
                feature_layer=layer,
                normalize=normalize,
                resize_input=resize_input,
                requires_grad=requires_grad,
                device=device,
                dtype=dtype,
                enable_timing=enable_timing,
            ) for layer in feature_layers
        ])

        # 记录设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_timing = enable_timing
        self.to(self.device, dtype=self.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.enable_timing:
            multi_start_time = time.time()
            print(f"🔍 多层级Perceptual Loss开始计算...")
        
        pred = pred.to(self.device, dtype=self.dtype, non_blocking=True)
        target = target.to(self.device, dtype=self.dtype, non_blocking=True)

        total = 0.0
        for i, (loss_module, w) in enumerate(zip(self.loss_modules, self.weights)):
            if self.enable_timing:
                layer_start_time = time.time()
                print(f"   层级 {i+1}: {loss_module.feature_layer_name}")
            
            layer_loss = loss_module(pred, target)
            total = total + w * layer_loss
            
            if self.enable_timing:
                layer_time = time.time() - layer_start_time
                print(f"   层级 {i+1} 耗时: {layer_time:.4f}s, 权重: {w}, Loss: {layer_loss.item():.6f}")
        
        if self.enable_timing:
            total_time = time.time() - multi_start_time
            print(f"🔍 多层级总耗时: {total_time:.4f}s")
        
        return total

class VGG16DISTSLoss(nn.Module):
    """
    基于VGG16的DISTS (Deep Image Structure and Texture Similarity) Loss
    使用relu2_2和relu3_3层特征，按照用户提供的公式实现
    
    设 I=渲染图、T=生成图（软GT）
    F^(2) = relu2_2(I), G^(2) = relu2_2(T)
    F^(3) = relu3_3(I), G^(3) = relu3_3(T)
    
    对每层 l∈{2,3} 和每个通道 c 在空间维计算:
    - 均值: μ_F,l,c, μ_G,l,c
    - 标准差: σ_F,l,c, σ_G,l,c  
    - 协方差: σ_FG,l,c
    
    纹理/亮度相似（"l"项）:
    l_l = (1/C) * Σ_c [2*μ_F,l,c*μ_G,l,c + c1] / [μ_F,l,c² + μ_G,l,c² + c1]
    
    结构相似（"s"项）:
    s_l = (1/C) * Σ_c [2*σ_FG,l,c + c2] / [σ_F,l,c² + σ_G,l,c² + c2]
    
    两层汇总为距离:
    L_DISTS = Σ_l∈{2,3} [α_l*(1-s_l) + β_l*(1-l_l)]
    """
    
    def __init__(
        self,
        normalize: bool = True,
        resize_input: bool = True,
        requires_grad: bool = False,
        device: Optional[str] = 'cuda',
        dtype: torch.dtype = torch.float32,
        enable_timing: bool = True,
        # 纹理/结构权重 (α_l, β_l)
        alpha_2: float = 0.5,  # relu2_2层结构权重
        beta_2: float = 0.5,   # relu2_2层纹理权重
        alpha_3: float = 0.5,  # relu3_3层结构权重
        beta_3: float = 0.5,   # relu3_3层纹理权重
        # 常数
        c1: float = 1e-6,
        c2: float = 1e-6,
    ):
        """
        Args:
            normalize: 是否做 ImageNet 标准化
            resize_input: 是否缩放到 224x224
            requires_grad: VGG 特征是否参与反传
            device: 计算设备
            dtype: 计算精度
            enable_timing: 是否启用时间统计
            alpha_2, beta_2: relu2_2层的结构和纹理权重
            alpha_3, beta_3: relu3_3层的结构和纹理权重
            c1, c2: 稳定性常数
        """
        super().__init__()
        
        # 设备选择
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("要求在 GPU 上运行，但当前环境未检测到 CUDA。")
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_timing = enable_timing
        
        # 权重参数
        self.alpha_2 = alpha_2  # relu2_2结构权重
        self.beta_2 = beta_2    # relu2_2纹理权重
        self.alpha_3 = alpha_3  # relu3_3结构权重
        self.beta_3 = beta_3    # relu3_3纹理权重
        self.c1 = c1
        self.c2 = c2
        
        # 加载VGG16并提取relu2_2和relu3_3层
        vgg = _load_vgg16_pretrained()
        
        # relu2_2: features[0..8] (第9层)
        self.feature_extractor_2_2 = nn.Sequential(*list(vgg.features.children())[:9])
        # relu3_3: features[0..15] (第16层)  
        self.feature_extractor_3_3 = nn.Sequential(*list(vgg.features.children())[:16])
        
        # 冻结参数
        for p in self.feature_extractor_2_2.parameters():
            p.requires_grad = requires_grad
        for p in self.feature_extractor_3_3.parameters():
            p.requires_grad = requires_grad
            
        self.feature_extractor_2_2.eval()
        self.feature_extractor_3_3.eval()
        
        # 标准化参数
        self.normalize = normalize
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=self.dtype).view(1, 3, 1, 1))
            self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225], dtype=self.dtype).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 3, 1, 1, dtype=self.dtype))
            self.register_buffer('std',  torch.ones(1, 3, 1, 1, dtype=self.dtype))
            
        self.resize_input = resize_input
        
        # 移动到设备
        self.to(self.device, dtype=self.dtype)
        
        # 获取特征层信息
        self._get_layer_info()
    
    @torch.no_grad()
    def _get_layer_info(self):
        """获取relu2_2和relu3_3层的通道数和尺寸信息"""
        test_input = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        x = test_input
        if self.normalize:
            x = (x - self.mean) / self.std
            
        # relu2_2特征
        features_2_2 = self.feature_extractor_2_2(x)
        self.C_2_2 = features_2_2.shape[1]
        self.H_2_2 = features_2_2.shape[2]
        self.W_2_2 = features_2_2.shape[3]
        
        # relu3_3特征
        features_3_3 = self.feature_extractor_3_3(x)
        self.C_3_3 = features_3_3.shape[1]
        self.H_3_3 = features_3_3.shape[2]
        self.W_3_3 = features_3_3.shape[3]
        
        print(f"[DISTS] relu2_2: C:{self.C_2_2} H:{self.H_2_2} W:{self.W_2_2}")
        print(f"[DISTS] relu3_3: C:{self.C_3_3} H:{self.H_3_3} W:{self.W_3_3}")
    
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """预处理输入"""
        # 移动设备/类型
        x = x.to(self.device, dtype=self.dtype, non_blocking=True)
        
        # 保证通道维在前
        if x.ndim == 4 and x.shape[-1] == 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        
        # 归一化到 [0,1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # 尺寸到 224x224
        if self.resize_input and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # ImageNet 标准化
        if self.normalize:
            x = (x - self.mean) / self.std
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取relu2_2和relu3_3特征"""
        if self.enable_timing:
            feature_start_time = time.time()
        
        x = self.preprocess_input(x)
        
        if self.enable_timing:
            preprocess_time = time.time() - feature_start_time
            vgg_start_time = time.time()
        
        # 提取特征
        if not any(p.requires_grad for p in self.feature_extractor_2_2.parameters()):
            with torch.no_grad():
                features_2_2 = self.feature_extractor_2_2(x)
                features_3_3 = self.feature_extractor_3_3(x)
        else:
            features_2_2 = self.feature_extractor_2_2(x)
            features_3_3 = self.feature_extractor_3_3(x)
        
        if self.enable_timing:
            vgg_time = time.time() - vgg_start_time
            total_feature_time = time.time() - feature_start_time
            print(f"🔍 DISTS特征提取时间统计:")
            print(f"   预处理时间: {preprocess_time:.4f}s")
            print(f"   VGG推理时间: {vgg_time:.4f}s")
            print(f"   总特征提取时间: {total_feature_time:.4f}s")
        
        return features_2_2, features_3_3
    
    def compute_dists_loss_layer(self, F: torch.Tensor, G: torch.Tensor, 
                                layer_name: str, alpha: float, beta: float) -> torch.Tensor:
        """
        计算单层的DISTS损失，按照用户提供的公式实现
        
        Args:
            F: 渲染图特征 [B, C, H, W]
            G: 生成图特征 [B, C, H, W]
            layer_name: 层名称
            alpha: 结构权重
            beta: 纹理权重
        """
        B, C, H, W = F.shape
        
        # 将特征展平为 [B, C, H*W]
        F_flat = F.view(B, C, -1)  # [B, C, H*W]
        G_flat = G.view(B, C, -1)  # [B, C, H*W]
        
        # 计算每个通道的统计量
        # μ_F,l,c = mean(F_l,c)
        mu_F = torch.mean(F_flat, dim=2)  # [B, C]
        mu_G = torch.mean(G_flat, dim=2)  # [B, C]
        
        # σ_F,l,c = std(F_l,c)
        sigma_F = torch.std(F_flat, dim=2)  # [B, C]
        sigma_G = torch.std(G_flat, dim=2)  # [B, C]
        
        # σ_FG,l,c = cov(F_l,c, G_l,c)
        # cov(x,y) = E[(x-μ_x)(y-μ_y)] = E[xy] - μ_x*μ_y
        F_centered = F_flat - mu_F.unsqueeze(2)  # [B, C, H*W]
        G_centered = G_flat - mu_G.unsqueeze(2)  # [B, C, H*W]
        sigma_FG = torch.mean(F_centered * G_centered, dim=2)  # [B, C]
        
        # 计算纹理/亮度相似性 l_l
        # l_l = (1/C) * Σ_c [2*μ_F,l,c*μ_G,l,c + c1] / [μ_F,l,c² + μ_G,l,c² + c1]
        l_l = (2 * mu_F * mu_G + self.c1) / (mu_F**2 + mu_G**2 + self.c1)  # [B, C]
        l_l = torch.mean(l_l, dim=1)  # [B] - 对通道求平均
        
        # 计算结构相似性 s_l
        # s_l = (1/C) * Σ_c [2*σ_FG,l,c + c2] / [σ_F,l,c² + σ_G,l,c² + c2]
        s_l = (2 * sigma_FG + self.c2) / (sigma_F**2 + sigma_G**2 + self.c2)  # [B, C]
        s_l = torch.mean(s_l, dim=1)  # [B] - 对通道求平均
        
        # 计算损失项: α_l*(1-s_l) + β_l*(1-l_l)
        structure_loss = alpha * (1 - s_l)  # [B]
        texture_loss = beta * (1 - l_l)     # [B]
        
        # 对batch求平均
        layer_loss = torch.mean(structure_loss + texture_loss)  # 标量
        
        if self.enable_timing:
            print(f"   {layer_name} - 结构相似性: {torch.mean(s_l).item():.6f}, 纹理相似性: {torch.mean(l_l).item():.6f}")
            print(f"   {layer_name} - 结构损失: {torch.mean(structure_loss).item():.6f}, 纹理损失: {torch.mean(texture_loss).item():.6f}")
        
        return layer_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算DISTS损失，按照用户提供的公式:
        L_DISTS = Σ_l∈{2,3} [α_l*(1-s_l) + β_l*(1-l_l)]
        """
        if self.enable_timing:
            forward_start_time = time.time()
        
        # 统一设备/类型
        pred = pred.to(self.device, dtype=self.dtype, non_blocking=True)
        target = target.to(self.device, dtype=self.dtype, non_blocking=True)
        
        # 提取特征
        F_2, F_3 = self.extract_features(pred)   # 渲染图特征
        G_2, G_3 = self.extract_features(target) # 生成图特征
        
        if self.enable_timing:
            feature_time = time.time() - forward_start_time
            loss_calc_start_time = time.time()
        
        # 计算各层损失
        loss_2 = self.compute_dists_loss_layer(F_2, G_2, "relu2_2", self.alpha_2, self.beta_2)
        loss_3 = self.compute_dists_loss_layer(F_3, G_3, "relu3_3", self.alpha_3, self.beta_3)
        
        # 两层汇总: L_DISTS = loss_2 + loss_3
        total_loss = loss_2 + loss_3
        
        if self.enable_timing:
            loss_calc_time = time.time() - loss_calc_start_time
            total_forward_time = time.time() - forward_start_time
            print(f"🔍 DISTS Loss计算时间统计:")
            print(f"   特征提取时间: {feature_time:.4f}s")
            print(f"   Loss计算时间: {loss_calc_time:.4f}s")
            print(f"   总前向时间: {total_forward_time:.4f}s")
            print(f"   relu2_2损失: {loss_2.item():.6f}")
            print(f"   relu3_3损失: {loss_3.item():.6f}")
            print(f"   总DISTS损失: {total_loss.item():.6f}")
        
        return total_loss
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取relu2_2和relu3_3特征图"""
        return self.extract_features(x)


def create_perceptual_loss(
    feature_layer: str = 'relu2_2',
    use_multiple_layers: bool = False,
    use_dists: bool = False,
    enable_timing: bool = True,
    **kwargs
) -> nn.Module:
    """
    创建感知损失模块
    
    Args:
        feature_layer: 单层模式下的特征层
        use_multiple_layers: 是否使用多层级感知损失
        use_dists: 是否使用DISTS损失
        enable_timing: 是否启用时间统计
        **kwargs: 其他参数
    """
    if use_dists:
        return VGG16DISTSLoss(enable_timing=enable_timing, **kwargs)
    elif use_multiple_layers:
        return VGG16PerceptualLossWithMultipleLayers(enable_timing=enable_timing, **kwargs)
    else:
        return VGG16PerceptualLoss(feature_layer=feature_layer, enable_timing=enable_timing, **kwargs)

if __name__ == "__main__":
    # ==== 使用示例（全 GPU） ====
    device = 'cuda'  # 强制 GPU
    dtype = torch.float32
    enable_timing = True  # 启用时间统计

    print("=== 单层 Perceptual Loss 示例（GPU） ===")
    perceptual_loss = VGG16PerceptualLoss(
        feature_layer='relu2_2', 
        device=device, 
        dtype=dtype,
        enable_timing=enable_timing
    )

    B = 2
    pred = torch.randn(B, 3, 256, 256, device=device, dtype=dtype)
    target = torch.randn(B, 3, 256, 256, device=device, dtype=dtype)

    loss = perceptual_loss(pred, target)
    print(f"单层损失值: {loss.item():.6f}")

    print("\n=== 多层级 Perceptual Loss 示例（GPU） ===")
    multi_loss_module = VGG16PerceptualLossWithMultipleLayers(
        feature_layers=['relu1_2','relu2_2','relu3_3'],
        weights=[0.1, 1.0, 0.1],
        device=device,
        dtype=dtype,
        enable_timing=enable_timing,
    )
    multi_loss = multi_loss_module(pred, target)
    print(f"多层级损失值: {multi_loss.item():.6f}")

    print("\n=== DISTS Loss 示例（GPU） ===")
    dists_loss = VGG16DISTSLoss(
        alpha_2=0.5,  # relu2_2结构权重
        beta_2=0.5,   # relu2_2纹理权重
        alpha_3=0.5,  # relu3_3结构权重
        beta_3=0.5,   # relu3_3纹理权重
        device=device,
        dtype=dtype,
        enable_timing=enable_timing
    )
    dists_loss_value = dists_loss(pred, target)
    print(f"DISTS损失值: {dists_loss_value.item():.6f}")

    print("\n=== 特征提取测试（GPU） ===")
    feats = perceptual_loss.get_feature_maps(pred)
    print(f"单层特征形状: {feats.shape}")

    feats_2_2, feats_3_3 = dists_loss.get_feature_maps(pred)
    print(f"DISTS relu2_2特征形状: {feats_2_2.shape}")
    print(f"DISTS relu3_3特征形状: {feats_3_3.shape}")

    loss, (pf, tf) = perceptual_loss(pred, target, return_features=True)
    print(f"损失值: {loss.item():.6f} | 预测特征: {pf.shape} | 目标特征: {tf.shape}")

    print("\n=== 时间统计信息 ===")
    timing_stats = perceptual_loss.get_timing_stats()
    for k, v in timing_stats.items():
        print(f"   {k}: {v}")

    print("\n=== 使用create_perceptual_loss函数 ===")
    # 创建DISTS损失
    dists_loss_2 = create_perceptual_loss(
        use_dists=True,
        alpha_2=0.6,  # relu2_2结构权重
        beta_2=0.4,   # relu2_2纹理权重
        alpha_3=0.5,  # relu3_3结构权重
        beta_3=0.5,   # relu3_3纹理权重
        device=device,
        enable_timing=enable_timing
    )
    dists_loss_2_value = dists_loss_2(pred, target)
    print(f"通过create_perceptual_loss创建的DISTS损失: {dists_loss_2_value.item():.6f}")
