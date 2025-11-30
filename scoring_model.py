#!/usr/bin/env python3
"""
Virtual View Quality Scoring Model

Implements PSNR-based quality evaluation for virtual views,
used to assess DiFix3D processing results.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from torchmetrics.image import PeakSignalNoiseRatio


def calculate_psnr(image1: torch.Tensor, image2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute PSNR between two images.

    Args:
        image1: First image [H, W, 3] or [1, H, W, 3], range [0, 1]
        image2: Second image [H, W, 3] or [1, H, W, 3], range [0, 1]
        data_range: Data range, default 1.0

    Returns:
        PSNR value
    """
    # Ensure consistent input shape
    if image1.dim() == 4:
        image1 = image1.squeeze(0)  # [H, W, 3]
    if image2.dim() == 4:
        image2 = image2.squeeze(0)  # [H, W, 3]
    
    # Ensure dtype consistency
    image1 = image1.float()
    image2 = image2.float()
    
    # Compute MSE
    mse = torch.mean((image1 - image2) ** 2)
    
    # Avoid division-by-zero
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def calculate_batch_psnr_statistics(
    original_images: List[torch.Tensor],
    processed_images: List[torch.Tensor]
) -> Tuple[float, float, List[float]]:
    """
    Compute PSNR statistics for a batch of images.

    Args:
        original_images: List of original images
        processed_images: List of processed images

    Returns:
        tuple: (mean, variance, list of PSNR values)
    """
    if len(original_images) != len(processed_images):
        raise ValueError("The number of original and processed images does not match")
    
    psnr_values = []
    for orig, proc in zip(original_images, processed_images):
        psnr = calculate_psnr(orig, proc)
        psnr_values.append(psnr)
    
    # Compute statistics
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
    Compute quality score k = mean - psnr (no variance division).

    Args:
        training_psnr_mean: PSNR mean for training views
        training_psnr_variance: PSNR variance for training views (kept for API compatibility)
        pseudo_view_psnr: PSNR for pseudo view

    Returns:
        Quality score k
    """
    # Directly compute mean minus pseudo-view PSNR; no variance division
    k = training_psnr_mean - pseudo_view_psnr
    return k


class VirtualViewQualityScorer:
    """Virtual view quality scorer"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        
        # Store statistics for training views
        self.training_psnr_mean = None
        self.training_psnr_variance = None
        self.training_psnr_values = []
    
    def evaluate_training_views(
        self,
        original_views: List[torch.Tensor],
        difix_processed_views: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Evaluate PSNR statistics for training views.

        Args:
            original_views: List of original training views
            difix_processed_views: List of training views processed by DiFix

        Returns:
            tuple: (PSNR mean, PSNR variance)
        """
        print(f" Start evaluating training view PSNR, total {len(original_views)} views")
        
        mean_psnr, var_psnr, psnr_values = calculate_batch_psnr_statistics(
            original_views, difix_processed_views
        )
        
        # Store statistics
        self.training_psnr_mean = mean_psnr
        self.training_psnr_variance = var_psnr
        self.training_psnr_values = psnr_values
        
        print(f" Training view PSNR statistics:")
        print(f"   Mean: {mean_psnr:.4f}")
        print(f"   Variance: {var_psnr:.4f}")
        print(f"   Min: {min(psnr_values):.4f}")
        print(f"   Max: {max(psnr_values):.4f}")
        
        return mean_psnr, var_psnr
    
    def score_pseudo_view(
        self,
        pseudo_view_original: torch.Tensor,
        pseudo_view_difix: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Score a pseudo view.

        Args:
            pseudo_view_original: Original pseudo view image
            pseudo_view_difix: Pseudo view image processed by DiFix

        Returns:
            tuple: (pseudo view PSNR, PSNR difference)
        """
        if self.training_psnr_mean is None or self.training_psnr_variance is None:
            raise ValueError("Please call evaluate_training_views first to compute training view statistics")
        
        # Compute pseudo-view PSNR
        pseudo_psnr = calculate_psnr(pseudo_view_original, pseudo_view_difix)
        
        # Compute quality score
        quality_score = calculate_quality_score(
            self.training_psnr_mean,
            self.training_psnr_variance,
            pseudo_psnr
        )

        print(f" Pseudo view score:")
        print(f"   Pseudo view PSNR: {pseudo_psnr:.4f}")
        print(f"   Training view PSNR mean: {self.training_psnr_mean:.4f}")
        print(f"   PSNR difference (mean - pseudo view): {quality_score:.4f}")
        
        return pseudo_psnr, quality_score
    
    def batch_score_pseudo_views(
        self,
        pseudo_views_original: List[torch.Tensor],
        pseudo_views_difix: List[torch.Tensor]
    ) -> List[Tuple[float, float]]:
        """
        Batch score pseudo views.

        Args:
            pseudo_views_original: List of original pseudo view images
            pseudo_views_difix: List of pseudo view images processed by DiFix

        Returns:
            List[Tuple[float, float]]: [(pseudo view PSNR, PSNR difference), ...]
        """
        if len(pseudo_views_original) != len(pseudo_views_difix):
            raise ValueError("The number of original and processed pseudo views does not match")
        
        results = []
        for i, (orig, difix) in enumerate(zip(pseudo_views_original, pseudo_views_difix)):
            pseudo_psnr, quality_score = self.score_pseudo_view(orig, difix)
            results.append((pseudo_psnr, quality_score))
            print(f"   Pseudo view {i+1}: PSNR={pseudo_psnr:.4f}, PSNR difference={quality_score:.4f}")
        
        return results
    
    def get_statistics_summary(self) -> dict:
        """
        Get statistics summary.

        Returns:
            Dictionary of statistics
        """
        return {
            "training_psnr_mean": self.training_psnr_mean,
            "training_psnr_variance": self.training_psnr_variance,
            "training_psnr_values": self.training_psnr_values,
            "num_training_views": len(self.training_psnr_values) if self.training_psnr_values else 0
        }


def test_scoring_model():
    """Test the scoring model functionality"""
    print(" Test virtual view quality scoring model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = VirtualViewQualityScorer(device=device)
    
    # Create test data
    height, width = 256, 256
    num_training_views = 5
    
    # Simulate training view data
    training_original = []
    training_difix = []
    
    for i in range(num_training_views):
        # Original image
        orig = torch.rand(height, width, 3, device=device)
        # Simulate DiFix processing (add some noise)
        noise = torch.randn_like(orig) * 0.1
        difix = torch.clamp(orig + noise, 0, 1)
        
        training_original.append(orig)
        training_difix.append(difix)
    
    # Evaluate training views
    mean_psnr, var_psnr = scorer.evaluate_training_views(training_original, training_difix)
    
    # Create pseudo view data
    pseudo_orig = torch.rand(height, width, 3, device=device)
    pseudo_difix = torch.clamp(pseudo_orig + torch.randn_like(pseudo_orig) * 0.05, 0, 1)
    
    # Evaluate pseudo view
    pseudo_psnr, quality_score = scorer.score_pseudo_view(pseudo_orig, pseudo_difix)
    
    # Print results
    print(f"\n Test results:")
    print(f"   Training view PSNR mean: {mean_psnr:.4f}")
    print(f"   Training view PSNR variance: {var_psnr:.4f}")
    print(f"   Pseudo view PSNR: {pseudo_psnr:.4f}")
    print(f"   Quality score k: {quality_score:.4f}")

    print(" Test completed")


if __name__ == "__main__":
    test_scoring_model()
