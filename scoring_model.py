#!/usr/bin/env python3
"""
虚拟视角质量打分模型

实现基于 PSNR 的虚拟视角质量评估，用于评估 DiFix3D 处理效果
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from torchmetrics.image import PeakSignalNoiseRatio


def calculate_psnr(image1: torch.Tensor, image2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    计算两张图像之间的 PSNR
    
    Args:
        image1: 第一张图像 [H, W, 3] 或 [1, H, W, 3]，范围 [0, 1]
        image2: 第二张图像 [H, W, 3] 或 [1, H, W, 3]，范围 [0, 1]
        data_range: 数据范围，默认 1.0
        
    Returns:
        PSNR 值
    """
    # 确保输入格式一致
    if image1.dim() == 4:
        image1 = image1.squeeze(0)  # [H, W, 3]
    if image2.dim() == 4:
        image2 = image2.squeeze(0)  # [H, W, 3]
    
    # 确保数据类型一致
    image1 = image1.float()
    image2 = image2.float()
    
    # 计算 MSE
    mse = torch.mean((image1 - image2) ** 2)
    
    # 避免除零错误
    if mse == 0:
        return float('inf')
    
    # 计算 PSNR
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def calculate_batch_psnr_statistics(
    original_images: List[torch.Tensor],
    processed_images: List[torch.Tensor]
) -> Tuple[float, float, List[float]]:
    """
    计算一批图像的 PSNR 统计信息
    
    Args:
        original_images: 原始图像列表
        processed_images: 处理后图像列表
        
    Returns:
        tuple: (均值, 方差, PSNR列表)
    """
    if len(original_images) != len(processed_images):
        raise ValueError("原始图像和处理图像数量不匹配")
    
    psnr_values = []
    for orig, proc in zip(original_images, processed_images):
        psnr = calculate_psnr(orig, proc)
        psnr_values.append(psnr)
    
    # 计算统计信息
    psnr_array = np.array(psnr_values)
    mean_psnr = np.mean(psnr_array)
    var_psnr = np.var(psnr_array)
    
    return mean_psnr, var_psnr, psnr_values


def calculate_quality_score(
    training_psnr_mean: float,
    training_psnr_variance: float,
    pseudo_view_psnr: float
) -> float:
    """
    计算质量分数 k = 均值 - psnr (不再除以方差)
    
    Args:
        training_psnr_mean: 训练视角 PSNR 均值
        training_psnr_variance: 训练视角 PSNR 方差 (保留参数以兼容现有接口)
        pseudo_view_psnr: 伪视角 PSNR
        
    Returns:
        质量分数 k
    """
    # 直接计算均值与伪视角PSNR的差值，不再除以方差
    k = training_psnr_mean - pseudo_view_psnr
    return k


class VirtualViewQualityScorer:
    """虚拟视角质量评分器"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        
        # 存储训练视角的统计信息
        self.training_psnr_mean = None
        self.training_psnr_variance = None
        self.training_psnr_values = []
    
    def evaluate_training_views(
        self,
        original_views: List[torch.Tensor],
        difix_processed_views: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """
        评估训练视角的 PSNR 统计信息
        
        Args:
            original_views: 原始训练视角列表
            difix_processed_views: DiFix 处理后的训练视角列表
            
        Returns:
            tuple: (PSNR均值, PSNR方差)
        """
        print(f"📊 开始评估训练视角 PSNR，共 {len(original_views)} 个视角")
        
        # 计算 PSNR 统计信息
        mean_psnr, var_psnr, psnr_values = calculate_batch_psnr_statistics(
            original_views, difix_processed_views
        )
        
        # 存储统计信息
        self.training_psnr_mean = mean_psnr
        self.training_psnr_variance = var_psnr
        self.training_psnr_values = psnr_values
        
        print(f"📊 训练视角 PSNR 统计:")
        print(f"   均值: {mean_psnr:.4f}")
        print(f"   方差: {var_psnr:.4f}")
        print(f"   最小值: {min(psnr_values):.4f}")
        print(f"   最大值: {max(psnr_values):.4f}")
        
        return mean_psnr, var_psnr
    
    def score_pseudo_view(
        self,
        pseudo_view_original: torch.Tensor,
        pseudo_view_difix: torch.Tensor
    ) -> Tuple[float, float]:
        """
        对伪视角进行打分
        
        Args:
            pseudo_view_original: 伪视角原始图像
            pseudo_view_difix: 伪视角 DiFix 处理后图像
            
        Returns:
            tuple: (伪视角PSNR, PSNR差值)
        """
        if self.training_psnr_mean is None or self.training_psnr_variance is None:
            raise ValueError("请先调用 evaluate_training_views 计算训练视角统计信息")
        
        # 计算伪视角 PSNR
        pseudo_psnr = calculate_psnr(pseudo_view_original, pseudo_view_difix)
        
        # 计算质量分数
        quality_score = calculate_quality_score(
            self.training_psnr_mean,
            self.training_psnr_variance,
            pseudo_psnr
        )
        
        print(f"📊 伪视角评分:")
        print(f"   伪视角 PSNR: {pseudo_psnr:.4f}")
        print(f"   训练视角 PSNR 均值: {self.training_psnr_mean:.4f}")
        print(f"   PSNR 差值 (均值-伪视角): {quality_score:.4f}")
        
        return pseudo_psnr, quality_score
    
    def batch_score_pseudo_views(
        self,
        pseudo_views_original: List[torch.Tensor],
        pseudo_views_difix: List[torch.Tensor]
    ) -> List[Tuple[float, float]]:
        """
        批量对伪视角进行打分
        
        Args:
            pseudo_views_original: 伪视角原始图像列表
            pseudo_views_difix: 伪视角 DiFix 处理后图像列表
            
        Returns:
            List[Tuple[float, float]]: [(伪视角PSNR, PSNR差值), ...]
        """
        if len(pseudo_views_original) != len(pseudo_views_difix):
            raise ValueError("原始伪视角和处理后伪视角数量不匹配")
        
        results = []
        for i, (orig, difix) in enumerate(zip(pseudo_views_original, pseudo_views_difix)):
            pseudo_psnr, quality_score = self.score_pseudo_view(orig, difix)
            results.append((pseudo_psnr, quality_score))
            print(f"   伪视角 {i+1}: PSNR={pseudo_psnr:.4f}, PSNR差值={quality_score:.4f}")
        
        return results
    
    def get_statistics_summary(self) -> dict:
        """
        获取统计信息摘要
        
        Returns:
            统计信息字典
        """
        return {
            "training_psnr_mean": self.training_psnr_mean,
            "training_psnr_variance": self.training_psnr_variance,
            "training_psnr_values": self.training_psnr_values,
            "num_training_views": len(self.training_psnr_values) if self.training_psnr_values else 0
        }


def test_scoring_model():
    """测试打分模型功能"""
    print("🧪 测试虚拟视角质量打分模型")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = VirtualViewQualityScorer(device=device)
    
    # 创建测试数据
    height, width = 256, 256
    num_training_views = 5
    
    # 模拟训练视角数据
    training_original = []
    training_difix = []
    
    for i in range(num_training_views):
        # 原始图像
        orig = torch.rand(height, width, 3, device=device)
        # 模拟 DiFix 处理（添加一些噪声）
        noise = torch.randn_like(orig) * 0.1
        difix = torch.clamp(orig + noise, 0, 1)
        
        training_original.append(orig)
        training_difix.append(difix)
    
    # 评估训练视角
    mean_psnr, var_psnr = scorer.evaluate_training_views(training_original, training_difix)
    
    # 创建伪视角数据
    pseudo_orig = torch.rand(height, width, 3, device=device)
    pseudo_difix = torch.clamp(pseudo_orig + torch.randn_like(pseudo_orig) * 0.05, 0, 1)
    
    # 评估伪视角
    pseudo_psnr, quality_score = scorer.score_pseudo_view(pseudo_orig, pseudo_difix)
    
    # 打印结果
    print(f"\n📊 测试结果:")
    print(f"   训练视角 PSNR 均值: {mean_psnr:.4f}")
    print(f"   训练视角 PSNR 方差: {var_psnr:.4f}")
    print(f"   伪视角 PSNR: {pseudo_psnr:.4f}")
    print(f"   质量分数 k: {quality_score:.4f}")
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_scoring_model()