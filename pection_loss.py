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
    try:
        # torchvision >= 0.13
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        # Fallback for older torchvision versions
        vgg = models.vgg16(pretrained=True)
    return vgg

class VGG16PerceptualLoss(nn.Module):

    def __init__(
        self,
        feature_layer: str = 'relu2_2',
        normalize: bool = True,
        resize_input: bool = True,
        requires_grad: bool = False,
        device: Optional[str] = 'cuda',     # Force GPU; set None to auto-select
        dtype: torch.dtype = torch.float32, # Unified dtype
        enable_timing: bool = True,         # Enable timing logs
    ):
        super().__init__()

        # Device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("GPU required but no CUDA device is available.")

        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_timing = enable_timing

        # Supported layers
        self.feature_layers = {
            'relu1_2': 3,   # features[0..4]
            'relu2_2': 8,   # features[0..9]
            'relu3_3': 15,  # features[0..16]
            'relu4_3': 22,  # features[0..23]
            'relu5_1': 25,  # features[0..26]
        }
        if feature_layer not in self.feature_layers:
            raise ValueError(f"Unsupported feature layer: {feature_layer}. Supported: {list(self.feature_layers.keys())}")

        self.feature_layer_idx = self.feature_layers[feature_layer]
        self.feature_layer_name = feature_layer
        
        # Load VGG16 and build the feature extractor
        vgg = _load_vgg16_pretrained()
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[: self.feature_layer_idx + 1])

        # Freeze and set evaluation mode
        for p in self.feature_extractor.parameters():
            p.requires_grad = requires_grad
        self.feature_extractor.eval()

        # Normalization parameters (registered as buffers to move with .to(device))
        self.normalize = normalize
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=self.dtype).view(1, 3, 1, 1))
            self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225], dtype=self.dtype).view(1, 3, 1, 1))
        else:
            # Register placeholders to keep dtype consistent
            self.register_buffer('mean', torch.zeros(1, 3, 1, 1, dtype=self.dtype))
            self.register_buffer('std',  torch.ones(1, 3, 1, 1, dtype=self.dtype))

        self.resize_input = resize_input

        # Move the entire module (params + buffers) to device/dtype
        self.to(self.device, dtype=self.dtype)

        self._get_layer_info()

    @torch.no_grad()
    def _get_layer_info(self):
        test_input = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        x = test_input
        if self.normalize:
            x = (x - self.mean) / self.std
        features = self.feature_extractor(x)
        self.C_j = features.shape[1]
        self.H_j = features.shape[2]
        self.W_j = features.shape[3]
        # Optional debug print
        # print(f"[{self.feature_layer_name}] C:{self.C_j} H:{self.H_j} W:{self.W_j}")

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:

        # Move to device/dtype
        x = x.to(self.device, dtype=self.dtype, non_blocking=True)

        # Ensure channel-first
        if x.ndim == 4 and x.shape[-1] == 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()

        # Normalize to [0,1] if input looks like 0..255
        if x.max() > 1.0:
            x = x / 255.0

        # Resize to 224x224
        if self.resize_input and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # ImageNet normalization
        if self.normalize:
            x = (x - self.mean) / self.std

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_timing:
            feature_start_time = time.time()
        
        x = self.preprocess_input(x)
        
        if self.enable_timing:
            preprocess_time = time.time() - feature_start_time
            vgg_start_time = time.time()
        
        # Use no_grad to speed up and save memory when gradients are not needed
        if not any(p.requires_grad for p in self.feature_extractor.parameters()):
            with torch.no_grad():
                features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)

        if self.enable_timing:
            vgg_time = time.time() - vgg_start_time
            total_feature_time = time.time() - feature_start_time
            print(f" VGG feature extraction timing:")
            print(f"   Preprocess time: {preprocess_time:.4f}s")
            print(f"   VGG inference time: {vgg_time:.4f}s")
            print(f"   Total feature extraction time: {total_feature_time:.4f}s")
        
        return features

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Compute perceptual loss: L2(φ(pred) - φ(target)) / (C*H*W)
        """
        if self.enable_timing:
            forward_start_time = time.time()
        
        # Move tensors to device/dtype
        pred = pred.to(self.device, dtype=self.dtype, non_blocking=True)
        target = target.to(self.device, dtype=self.dtype, non_blocking=True)

        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        if self.enable_timing:
            feature_time = time.time() - forward_start_time
            loss_calc_start_time = time.time()


        diff = pred_features - target_features
        # [B]
        l2_sq = torch.sum(diff * diff, dim=(1, 2, 3))
        norm = float(self.C_j * self.H_j * self.W_j)
        loss = torch.mean(l2_sq / norm)
        
        if self.enable_timing:
            loss_calc_time = time.time() - loss_calc_start_time
            total_forward_time = time.time() - forward_start_time
            print(f" Perceptual Loss timing:")
            print(f"   Feature extraction time: {feature_time:.4f}s")
            print(f"   Loss computation time: {loss_calc_time:.4f}s")
            print(f"   Total forward time: {total_forward_time:.4f}s")
            print(f"   Feature shape: {pred_features.shape}")
    
        if return_features:
            return loss, (pred_features, target_features)
        return loss

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

    def get_timing_stats(self) -> dict:

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
    Perceptual Loss using multiple VGG layers (all on the same GPU)
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
            raise ValueError("Feature layer count must match weight count")

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

        # Record device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_timing = enable_timing
        self.to(self.device, dtype=self.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.enable_timing:
            multi_start_time = time.time()
            print(f" Multi-layer Perceptual Loss computation started...")
        
        pred = pred.to(self.device, dtype=self.dtype, non_blocking=True)
        target = target.to(self.device, dtype=self.dtype, non_blocking=True)

        total = 0.0
        for i, (loss_module, w) in enumerate(zip(self.loss_modules, self.weights)):
            if self.enable_timing:
                layer_start_time = time.time()
                print(f"   Layer {i+1}: {loss_module.feature_layer_name}")
            
            layer_loss = loss_module(pred, target)
            total = total + w * layer_loss
            
            if self.enable_timing:
                layer_time = time.time() - layer_start_time
                print(f"   Layer {i+1} time: {layer_time:.4f}s, weight: {w}, loss: {layer_loss.item():.6f}")
        
        if self.enable_timing:
            total_time = time.time() - multi_start_time
            print(f" Multi-layer total time: {total_time:.4f}s")
        
        return total

class VGG16DISTSLoss(nn.Module):
    
    def __init__(
        self,
        normalize: bool = True,
        resize_input: bool = True,
        requires_grad: bool = False,
        device: Optional[str] = 'cuda',
        dtype: torch.dtype = torch.float32,
        enable_timing: bool = True,
        alpha_2: float = 0.5,  
        beta_2: float = 0.5,   
        alpha_3: float = 0.5,  
        beta_3: float = 0.5,   
        c1: float = 1e-6,
        c2: float = 1e-6,
    ):
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but no CUDA device is available.")
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_timing = enable_timing
        

        self.alpha_2 = alpha_2  
        self.beta_2 = beta_2    
        self.alpha_3 = alpha_3  
        self.beta_3 = beta_3    
        self.c1 = c1
        self.c2 = c2
        
        vgg = _load_vgg16_pretrained()
        
        self.feature_extractor_2_2 = nn.Sequential(*list(vgg.features.children())[:9])
        self.feature_extractor_3_3 = nn.Sequential(*list(vgg.features.children())[:16])
        
        for p in self.feature_extractor_2_2.parameters():
            p.requires_grad = requires_grad
        for p in self.feature_extractor_3_3.parameters():
            p.requires_grad = requires_grad
            
        self.feature_extractor_2_2.eval()
        self.feature_extractor_3_3.eval()
        
        self.normalize = normalize
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=self.dtype).view(1, 3, 1, 1))
            self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225], dtype=self.dtype).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 3, 1, 1, dtype=self.dtype))
            self.register_buffer('std',  torch.ones(1, 3, 1, 1, dtype=self.dtype))
            
        self.resize_input = resize_input
        
        self.to(self.device, dtype=self.dtype)
        
        self._get_layer_info()
    
    @torch.no_grad()
    def _get_layer_info(self):
        test_input = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        x = test_input
        if self.normalize:
            x = (x - self.mean) / self.std
            
        features_2_2 = self.feature_extractor_2_2(x)
        self.C_2_2 = features_2_2.shape[1]
        self.H_2_2 = features_2_2.shape[2]
        self.W_2_2 = features_2_2.shape[3]
        
        features_3_3 = self.feature_extractor_3_3(x)
        self.C_3_3 = features_3_3.shape[1]
        self.H_3_3 = features_3_3.shape[2]
        self.W_3_3 = features_3_3.shape[3]
        
        print(f"[DISTS] relu2_2: C:{self.C_2_2} H:{self.H_2_2} W:{self.W_2_2}")
        print(f"[DISTS] relu3_3: C:{self.C_3_3} H:{self.H_3_3} W:{self.W_3_3}")
    
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:

        x = x.to(self.device, dtype=self.dtype, non_blocking=True)
        
        if x.ndim == 4 and x.shape[-1] == 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        
        if x.max() > 1.0:
            x = x / 255.0
        
        if self.resize_input and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        if self.normalize:
            x = (x - self.mean) / self.std
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_timing:
            feature_start_time = time.time()
        
        x = self.preprocess_input(x)
        
        if self.enable_timing:
            preprocess_time = time.time() - feature_start_time
            vgg_start_time = time.time()
        
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
        
        return features_2_2, features_3_3
    
    def compute_dists_loss_layer(self, F: torch.Tensor, G: torch.Tensor, 
                                layer_name: str, alpha: float, beta: float) -> torch.Tensor:
        B, C, H, W = F.shape
        
        F_flat = F.view(B, C, -1)  # [B, C, H*W]
        G_flat = G.view(B, C, -1)  # [B, C, H*W]
        
        # mu_F,l,c = mean(F_l,c)
        mu_F = torch.mean(F_flat, dim=2)  # [B, C]
        mu_G = torch.mean(G_flat, dim=2)  # [B, C]

        # sigma_F,l,c = std(F_l,c)
        sigma_F = torch.std(F_flat, dim=2)  # [B, C]
        sigma_G = torch.std(G_flat, dim=2)  # [B, C]

        # sigma_FG,l,c = cov(F_l,c, G_l,c)
        # cov(x,y) = E[(x-mu_x)(y-mu_y)] = E[xy] - mu_x*mu_y
        F_centered = F_flat - mu_F.unsqueeze(2)  # [B, C, H*W]
        G_centered = G_flat - mu_G.unsqueeze(2)  # [B, C, H*W]
        sigma_FG = torch.mean(F_centered * G_centered, dim=2)  # [B, C]

        # l_l = (1/C) * sum_c [2* mu_F * mu_G + c1] / [mu_F^2 + mu_G^2 + c1]
        l_l = (2 * mu_F * mu_G + self.c1) / (mu_F**2 + mu_G**2 + self.c1)  # [B, C]
        l_l = torch.mean(l_l, dim=1)  # [B] - averaged over channels
        
 
        # s_l = (1/C) * sum_c [2* sigma_FG + c2] / [sigma_F^2 + sigma_G^2 + c2]
        s_l = (2 * sigma_FG + self.c2) / (sigma_F**2 + sigma_G**2 + self.c2)  # [B, C]
        s_l = torch.mean(s_l, dim=1)  # [B] - averaged over channels
        
        structure_loss = alpha * (1 - s_l)  # [B]
        texture_loss = beta * (1 - l_l)     # [B]
        

        layer_loss = torch.mean(structure_loss + texture_loss)  # scalar
        
        if self.enable_timing:
            print(f"   {layer_name} - structural similarity: {torch.mean(s_l).item():.6f}, texture similarity: {torch.mean(l_l).item():.6f}")
            print(f"   {layer_name} - structure loss: {torch.mean(structure_loss).item():.6f}, texture loss: {torch.mean(texture_loss).item():.6f}")
        
        return layer_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.enable_timing:
            forward_start_time = time.time()
        
        # Unify device/dtype
        pred = pred.to(self.device, dtype=self.dtype, non_blocking=True)
        target = target.to(self.device, dtype=self.dtype, non_blocking=True)
        
        # Extract features
        F_2, F_3 = self.extract_features(pred)   # Rendered image features
        G_2, G_3 = self.extract_features(target) # Generated image features
        
        if self.enable_timing:
            feature_time = time.time() - forward_start_time
            loss_calc_start_time = time.time()
        
        # Compute per-layer losses
        loss_2 = self.compute_dists_loss_layer(F_2, G_2, "relu2_2", self.alpha_2, self.beta_2)
        loss_3 = self.compute_dists_loss_layer(F_3, G_3, "relu3_3", self.alpha_3, self.beta_3)
        
        # Sum two layers: L_DISTS = loss_2 + loss_3
        total_loss = loss_2 + loss_3
        
        if self.enable_timing:
            loss_calc_time = time.time() - loss_calc_start_time
            total_forward_time = time.time() - forward_start_time
            print(f" DISTS loss timing:")
            print(f"   Feature extraction time: {feature_time:.4f}s")
            print(f"   Loss computation time: {loss_calc_time:.4f}s")
            print(f"   Total forward time: {total_forward_time:.4f}s")
            print(f"   relu2_2 loss: {loss_2.item():.6f}")
            print(f"   relu3_3 loss: {loss_3.item():.6f}")
            print(f"   Total DISTS loss: {total_loss.item():.6f}")
        
        return total_loss
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get feature maps for relu2_2 and relu3_3"""
        return self.extract_features(x)


def create_perceptual_loss(
    feature_layer: str = 'relu2_2',
    use_multiple_layers: bool = False,
    use_dists: bool = False,
    enable_timing: bool = True,
    **kwargs
) -> nn.Module:

    if use_dists:
        return VGG16DISTSLoss(enable_timing=enable_timing, **kwargs)
    elif use_multiple_layers:
        return VGG16PerceptualLossWithMultipleLayers(enable_timing=enable_timing, **kwargs)
    else:
        return VGG16PerceptualLoss(feature_layer=feature_layer, enable_timing=enable_timing, **kwargs)

if __name__ == "__main__":
    # ==== Usage example (all GPU) ====
    device = 'cuda'  # Force GPU
    dtype = torch.float32
    enable_timing = True  # Enable timing

    print("=== Single-layer Perceptual Loss example (GPU) ===")
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
    print(f"Single-layer loss: {loss.item():.6f}")

    print("\n=== Multi-layer Perceptual Loss example (GPU) ===")
    multi_loss_module = VGG16PerceptualLossWithMultipleLayers(
        feature_layers=['relu1_2','relu2_2','relu3_3'],
        weights=[0.1, 1.0, 0.1],
        device=device,
        dtype=dtype,
        enable_timing=enable_timing,
    )
    multi_loss = multi_loss_module(pred, target)
    print(f"Multi-layer loss: {multi_loss.item():.6f}")

    print("\n=== DISTS Loss example (GPU) ===")
    dists_loss = VGG16DISTSLoss(
        alpha_2=0.5,  
        beta_2=0.5,   
        alpha_3=0.5,  
        beta_3=0.5,   
        device=device,
        dtype=dtype,
        enable_timing=enable_timing
    )
    dists_loss_value = dists_loss(pred, target)
    print(f"DISTS loss: {dists_loss_value.item():.6f}")

    print("\n=== Feature extraction test (GPU) ===")
    feats = perceptual_loss.get_feature_maps(pred)
    print(f"Single-layer feature shape: {feats.shape}")

    feats_2_2, feats_3_3 = dists_loss.get_feature_maps(pred)
    print(f"DISTS relu2_2 feature shape: {feats_2_2.shape}")
    print(f"DISTS relu3_3 feature shape: {feats_3_3.shape}")

    loss, (pf, tf) = perceptual_loss(pred, target, return_features=True)
    print(f"Loss: {loss.item():.6f} | pred features: {pf.shape} | target features: {tf.shape}")

    print("\n=== Timing stats ===")
    timing_stats = perceptual_loss.get_timing_stats()
    for k, v in timing_stats.items():
        print(f"   {k}: {v}")

    print("\n=== Using create_perceptual_loss ===")
    # Create DISTS loss
    dists_loss_2 = create_perceptual_loss(
        use_dists=True,
        alpha_2=0.6,  # Structure weight for relu2_2
        beta_2=0.4,   # Texture weight for relu2_2
        alpha_3=0.5,  # Structure weight for relu3_3
        beta_3=0.5,   # Texture weight for relu3_3
        device=device,
        enable_timing=enable_timing
    )
    dists_loss_2_value = dists_loss_2(pred, target)
    print(f"DISTS loss via create_perceptual_loss: {dists_loss_2_value.item():.6f}")
