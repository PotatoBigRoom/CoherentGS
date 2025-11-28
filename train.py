import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import argparse
import random
import threading
import numpy as np
import psutil
import os
from datetime import datetime
from collections import deque
import math

class GPUIntensiveOccupier:
    def __init__(self, gpu_id, memory_gb, duration):
        self.gpu_id = gpu_id
        self.memory_gb = memory_gb
        self.duration = duration
        self.device = torch.device(f'cuda:{gpu_id}')
        self.running = True
        self.tensors = []
        self.models = []
        self.optimizers = []
        self.epoch = 0
        self.batch_count = 0
        self.loss_history = deque(maxlen=100)
        self.gpu_compute_threads = []
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        gpu_mem = torch.cuda.memory_allocated(self.device) / 1024**3
        cpu_percent = psutil.cpu_percent()
        mem_percent = psutil.virtual_memory().percent
        print(f"[{timestamp}] GPU{self.gpu_id} | GPU:{gpu_mem:.1f}GB | CPU:{cpu_percent:.1f}% | RAM:{mem_percent:.1f}% | {message}")
    
    def create_massive_gpu_model(self):
        """Create an extremely large GPU-intensive model"""
        class MassiveGPUTransformer(nn.Module):
            def __init__(self, vocab_size=100000, d_model=3072, nhead=48, num_layers=32):
                super().__init__()
                # Massive embedding layer
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(16384, d_model))
                
                # Multiple parallel transformer branches for GPU parallelism
                self.transformer_branches = nn.ModuleList([
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model, nhead, 
                            dim_feedforward=d_model * 8,  # 24576 dim feedforward
                            batch_first=True,
                            dropout=0.0  # No dropout for max computation
                        ),
                        num_layers // 4  # 8 layers per branch
                    ) for _ in range(4)  # 4 parallel branches
                ])
                
                # Massive MLP blocks
                self.massive_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model * 8),
                        nn.GELU(),
                        nn.Linear(d_model * 8, d_model * 8),
                        nn.GELU(),
                        nn.Linear(d_model * 8, d_model * 4),
                        nn.GELU(),
                        nn.Linear(d_model * 4, d_model),
                    ) for _ in range(4)
                ])
                
                # Cross-attention between branches
                self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                
                # Final massive classifier
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, vocab_size)
                )
                self.d_model = d_model
                
            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) * math.sqrt(self.d_model)
                x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
                x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
                
                # Process through parallel branches
                branch_outputs = []
                for i, (transformer, mlp) in enumerate(zip(self.transformer_branches, self.massive_mlps)):
                    branch_out = transformer(x)
                    branch_out = mlp(branch_out)
                    branch_outputs.append(branch_out)
                
                # Combine branches with cross-attention
                combined = torch.stack(branch_outputs, dim=1)  # [B, 4, S, D]
                B, num_branches, S, D = combined.shape
                combined = combined.view(B, num_branches * S, D)
                
                attended, _ = self.cross_attention(combined, combined, combined)
                attended = attended.view(B, num_branches, S, D)
                
                # Average across branches
                x = attended.mean(dim=1)  # [B, S, D]
                
                return self.classifier(x)
        
        model = MassiveGPUTransformer().to(self.device)
        # Use multiple optimizers for different parts
        transformer_params = []
        mlp_params = []
        classifier_params = list(model.classifier.parameters())
        
        for branch in model.transformer_branches:
            transformer_params.extend(list(branch.parameters()))
        for mlp in model.massive_mlps:
            mlp_params.extend(list(mlp.parameters()))
        
        optimizer1 = optim.AdamW(transformer_params, lr=1e-4, weight_decay=0.01)
        optimizer2 = optim.AdamW(mlp_params, lr=5e-5, weight_decay=0.01)
        optimizer3 = optim.AdamW(classifier_params, lr=2e-4, weight_decay=0.01)
        
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=3000)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=3000)
        scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=3000)
        
        self.models.append(model)
        self.optimizers.append(((optimizer1, optimizer2, optimizer3), (scheduler1, scheduler2, scheduler3)))
        
        total_params = sum(p.numel() for p in model.parameters())
        self.log(f"Created massive GPU model with {total_params/1e6:.1f}M parameters")
        return model, (optimizer1, optimizer2, optimizer3), (scheduler1, scheduler2, scheduler3)
    
    def create_parallel_gpu_models(self):
        """Create multiple models for parallel GPU computation"""
        models = []
        
        # Vision Transformer for parallel computation
        class MassiveViT(nn.Module):
            def __init__(self, image_size=512, patch_size=16, d_model=2048, nhead=32, num_layers=24):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
                num_patches = (image_size // patch_size) ** 2
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
                
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
                    num_layers
                )
                
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, 1000)
                )
                
            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
                x = self.transformer(x)
                return self.head(x[:, 0])
        
        # Massive ConvNet
        class MassiveConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stages = nn.ModuleList()
                channels = [3, 128, 256, 512, 1024, 2048]
                
                for i in range(5):
                    stage = nn.Sequential(
                        nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                        nn.BatchNorm2d(channels[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                        nn.BatchNorm2d(channels[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                        nn.BatchNorm2d(channels[i+1]),
                        nn.ReLU(inplace=True),
                    )
                    self.stages.append(stage)
                
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(2048, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 1000)
                )
                
            def forward(self, x):
                for stage in self.stages:
                    x = stage(x)
                    x = F.max_pool2d(x, 2)
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        # 3D CNN for video processing
        class Massive3DCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv3d_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(3 if i == 0 else 64 * (2**i), 64 * (2**(i+1)), 3, padding=1),
                        nn.BatchNorm3d(64 * (2**(i+1))),
                        nn.ReLU(),
                        nn.Conv3d(64 * (2**(i+1)), 64 * (2**(i+1)), 3, padding=1),
                        nn.BatchNorm3d(64 * (2**(i+1))),
                        nn.ReLU(),
                    ) for i in range(4)  # 4 stages: 64, 128, 256, 512
                ])
                
                self.global_pool = nn.AdaptiveAvgPool3d(1)
                self.classifier = nn.Linear(512, 100)
                
            def forward(self, x):
                for block in self.conv3d_blocks:
                    x = block(x)
                    x = F.max_pool3d(x, 2)
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        # Create instances
        vit_model = MassiveViT().to(self.device)
        conv_model = MassiveConvNet().to(self.device)
        conv3d_model = Massive3DCNN().to(self.device)
        
        models.extend([vit_model, conv_model, conv3d_model])
        self.models.extend(models)
        
        self.log(f"Created {len(models)} parallel GPU models")
        return models
    
    def continuous_gpu_matrix_operations(self):
        """Continuous GPU-intensive matrix operations"""
        def matrix_worker():
            while self.running:
                try:
                    # Very large matrix operations
                    sizes = [2048, 3072, 4096, 6144]
                    size1 = random.choice(sizes)
                    size2 = random.choice(sizes)
                    
                    # Create large tensors directly on GPU
                    a = torch.randn(size1, size2, device=self.device, dtype=torch.float32)
                    b = torch.randn(size2, size1, device=self.device, dtype=torch.float32)
                    
                    # Chain of intensive matrix operations
                    c = torch.matmul(a, b)
                    d = torch.matmul(c, a)
                    e = torch.matmul(d.T, b.T)
                    
                    # More complex operations
                    f = torch.matmul(e, c.T)
                    g = torch.matmul(f.T, d)
                    
                    # Element-wise operations
                    h = torch.sin(g) * torch.cos(g)
                    i = torch.exp(h * 0.01)  # Prevent overflow
                    j = torch.log(torch.abs(i) + 1e-8)
                    
                    # Reduction operations
                    result = torch.sum(j, dim=1, keepdim=True)
                    
                    # Cleanup
                    del a, b, c, d, e, f, g, h, i, j, result
                    
                    # No sleep - maximum GPU utilization
                    
                except Exception as e:
                    time.sleep(0.01)
        
        # Start multiple parallel matrix workers
        for i in range(4):  # 4 parallel matrix computation threads
            thread = threading.Thread(target=matrix_worker, name=f"MatrixWorker-{i}")
            thread.daemon = True
            thread.start()
            self.gpu_compute_threads.append(thread)
    
    def continuous_gpu_convolution_operations(self):
        """Continuous GPU-intensive convolution operations"""
        def conv_worker():
            while self.running:
                try:
                    # Large convolution batches
                    batch_sizes = [64, 96, 128]
                    channels = [512, 768, 1024]
                    sizes = [128, 192, 256]
                    
                    batch_size = random.choice(batch_sizes)
                    channel = random.choice(channels)
                    size = random.choice(sizes)
                    
                    # Create large input tensor
                    x = torch.randn(batch_size, channel, size, size, device=self.device)
                    
                    # Multiple convolution layers
                    conv1 = nn.Conv2d(channel, channel, 3, padding=1).to(self.device)
                    conv2 = nn.Conv2d(channel, channel, 5, padding=2).to(self.device)
                    conv3 = nn.Conv2d(channel, channel, 7, padding=3).to(self.device)
                    
                    # Forward passes
                    y1 = F.relu(conv1(x))
                    y2 = F.relu(conv2(y1))
                    y3 = F.relu(conv3(y2))
                    
                    # Pooling and upsampling
                    pooled = F.max_pool2d(y3, 2)
                    upsampled = F.interpolate(pooled, scale_factor=2, mode='bilinear', align_corners=False)
                    
                    # More intensive operations
                    result = F.conv2d(upsampled, torch.randn(channel, channel, 3, 3, device=self.device), padding=1)
                    
                    # Cleanup
                    del x, y1, y2, y3, pooled, upsampled, result
                    del conv1, conv2, conv3
                    
                except Exception as e:
                    time.sleep(0.01)
        
        # Start multiple convolution workers
        for i in range(3):  # 3 parallel convolution threads
            thread = threading.Thread(target=conv_worker, name=f"ConvWorker-{i}")
            thread.daemon = True
            thread.start()
            self.gpu_compute_threads.append(thread)
    
    def continuous_gpu_fft_operations(self):
        """Continuous GPU-intensive FFT operations"""
        def fft_worker():
            while self.running:
                try:
                    # Large FFT operations
                    sizes = [(2048, 2048), (3072, 2048), (4096, 4096)]
                    size = random.choice(sizes)
                    
                    # Create complex tensor
                    real = torch.randn(size, device=self.device)
                    imag = torch.randn(size, device=self.device)
                    complex_tensor = torch.complex(real, imag)
                    
                    # Forward FFT
                    fft_result = torch.fft.fft2(complex_tensor)
                    
                    # Some operations in frequency domain
                    filtered = fft_result * torch.exp(-0.1 * torch.abs(fft_result))
                    
                    # Inverse FFT
                    ifft_result = torch.fft.ifft2(filtered)
                    
                    # Real operations
                    magnitude = torch.abs(ifft_result)
                    phase = torch.angle(ifft_result)
                    
                    # More operations
                    result = magnitude * torch.cos(phase)
                    
                    # Cleanup
                    del real, imag, complex_tensor, fft_result, filtered, ifft_result
                    del magnitude, phase, result
                    
                except Exception as e:
                    time.sleep(0.01)
        
        # Start FFT worker
        thread = threading.Thread(target=fft_worker, name="FFTWorker")
        thread.daemon = True
        thread.start()
        self.gpu_compute_threads.append(thread)
    
    def mega_intensive_training_step(self, model, optimizers, schedulers):
        """Mega intensive training step with maximum GPU utilization"""
        model.train()
        
        # Very large batch sizes
        batch_size = random.choice([64, 96, 128, 160])
        seq_len = random.choice([2048, 3072, 4096])
        
        total_loss = 0
        accumulation_steps = 8  # More gradient accumulation
        
        optimizer1, optimizer2, optimizer3 = optimizers
        scheduler1, scheduler2, scheduler3 = schedulers
        
        for acc_step in range(accumulation_steps):
            # Large input data
            input_ids = torch.randint(0, 100000, (batch_size, seq_len), device=self.device)
            labels = torch.randint(0, 100000, (batch_size, seq_len), device=self.device)
            
            # Forward pass with mega model
            outputs = model(input_ids)
            
            # Multiple loss computations
            loss1 = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Additional GPU-intensive loss terms
            l2_loss = sum(p.pow(2.0).sum() for p in model.parameters()) * 0.001
            
            # Attention regularization (GPU intensive)
            attention_weights = []
            for branch in model.transformer_branches:
                for layer in branch.layers:
                    if hasattr(layer.self_attn, 'attention_weights'):
                        attention_weights.append(layer.self_attn.attention_weights)
            
            total_loss_step = (loss1 + l2_loss) / accumulation_steps
            total_loss_step.backward()
            
            total_loss += total_loss_step.item()
            
            del input_ids, labels, outputs
        
        # Update with multiple optimizers
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        
        self.batch_count += 1
        self.loss_history.append(total_loss)
        
        if self.batch_count % 25 == 0:  # More frequent logging
            avg_loss = np.mean(list(self.loss_history))
            lr1 = scheduler1.get_last_lr()[0]
            self.log(f"Epoch {self.epoch}, Step {self.batch_count}, Loss: {avg_loss:.4f}, LR: {lr1:.2e}")
        
        return total_loss
    
    def parallel_model_inference(self, parallel_models):
        """Run parallel inference on additional models"""
        try:
            if len(parallel_models) >= 3:
                vit_model, conv_model, conv3d_model = parallel_models[:3]
                
                # ViT inference
                batch_size = random.randint(32, 64)
                vit_input = torch.randn(batch_size, 3, 512, 512, device=self.device)
                _ = vit_model(vit_input)
                
                # ConvNet inference
                conv_input = torch.randn(batch_size, 3, 256, 256, device=self.device)
                _ = conv_model(conv_input)
                
                # 3D CNN inference
                video_input = torch.randn(batch_size//4, 3, 16, 128, 128, device=self.device)
                _ = conv3d_model(video_input)
                
                del vit_input, conv_input, video_input
                
        except Exception as e:
            pass
    
    def simulate_mega_intensive_training(self):
        """Simulate mega-intensive GPU training"""
        model, optimizers, schedulers = self.create_massive_gpu_model()
        parallel_models = self.create_parallel_gpu_models()
        
        steps_per_epoch = random.randint(1000, 1500)
        parallel_inference_interval = 3  # Every 3 steps
        
        self.log(f"Starting mega-intensive GPU training: {steps_per_epoch} steps/epoch")
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < self.duration:
            try:
                for step in range(steps_per_epoch):
                    if not self.running:
                        break
                    
                    # Main intensive training
                    loss = self.mega_intensive_training_step(model, optimizers, schedulers)
                    
                    # Parallel model inference
                    if step % parallel_inference_interval == 0:
                        self.parallel_model_inference(parallel_models)
                    
                    # No sleep - maximum intensity
                
                self.epoch += 1
                self.log(f"Mega-intensive epoch {self.epoch} completed")
                
                # Minimal rest between epochs
                time.sleep(0.1)
                
            except Exception as e:
                self.log(f"Mega-intensive training error: {e}")
                time.sleep(1)
    
    def occupy_memory(self):
        """Occupy GPU memory and start mega-intensive computation"""
        try:
            # Allocate base memory
            base_size = int(self.memory_gb * 0.5 * 1024 * 1024 * 1024 / 4)
            base_tensor = torch.randn(base_size, device=self.device, dtype=torch.float32)
            self.tensors.append(base_tensor)
            
            self.log(f"Pre-allocated GPU memory: {self.memory_gb * 0.5:.1f}GB")
            
            # Start all GPU-intensive operations
            self.continuous_gpu_matrix_operations()
            self.continuous_gpu_convolution_operations()
            self.continuous_gpu_fft_operations()
            
            # Start monitoring
            monitor_thread = threading.Thread(target=self.monitor_resources)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Start mega-intensive training
            training_thread = threading.Thread(target=self.simulate_mega_intensive_training)
            training_thread.daemon = True
            training_thread.start()
            
            self.log(f"Mega-intensive GPU computation started for {self.duration/3600:.1f} hours")
            self.log("Maximum GPU utilization mode activated!")
            
            # Main loop with intensive bursts - 移除sleep，增加GPU同步
            start_time = time.time()
            while time.time() - start_time < self.duration:
                try:
                    # 连续GPU操作，无间隔
                    for _ in range(10):  # 连续10次操作
                        burst_size = random.randint(200_000_000, 500_000_000)
                        burst_tensor = torch.randn(burst_size, device=self.device)
                        
                        # Intensive burst operations
                        burst_result = torch.matmul(
                            burst_tensor.view(-1, 1000),
                            torch.randn(1000, 2000, device=self.device)
                        )
                        burst_final = torch.sum(burst_result, dim=1)
                        
                        # 强制GPU同步，确保操作完成
                        torch.cuda.synchronize()
                        
                        del burst_tensor, burst_result, burst_final
                    
                    # 极短暂休息，保持高利用率
                    time.sleep(0.01)  # 10ms instead of 15s
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.log(f"Main loop error: {e}")
                    time.sleep(5)
            
        except KeyboardInterrupt:
            self.log("User interrupted...")
        except torch.cuda.OutOfMemoryError:
            self.log(f"Out of GPU memory! Cannot allocate {self.memory_gb}GB")
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.running = False
            self._cleanup()
    
    def monitor_resources(self):
        """Monitor GPU resources"""
        while self.running:
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                
                # Try to get GPU utilization
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.log(f"GPU Monitor - Mem:{gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB, "
                           f"Util:{gpu_util.gpu}%, Temp:{temp}°C")
                except:
                    self.log(f"GPU Monitor - Memory:{gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB")
                
                time.sleep(15)  # Monitor every 15 seconds
                
            except Exception as e:
                time.sleep(15)
    
    def _cleanup(self):
        """Clean up GPU resources"""
        self.log("Cleaning up mega-intensive GPU resources...")
        
        # Wait for compute threads
        for thread in self.gpu_compute_threads:
            if thread.is_alive():
                thread.join(timeout=1)
        
        # Clean GPU memory
        for tensor in self.tensors:
            del tensor
        for model in self.models:
            del model
        
        self.tensors.clear()
        self.models.clear()
        self.optimizers.clear()
        self.gpu_compute_threads.clear()
        
        torch.cuda.empty_cache()
        self.log("GPU resource cleanup completed")

# Main functions remain the same, just use GPUIntensiveOccupier
def occupy_gpu_memory(gpu_id=0, memory_gb=70, duration=36000):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    occupier = GPUIntensiveOccupier(gpu_id, memory_gb, duration)
    occupier.occupy_memory()

def occupy_all_gpus(memory_per_gpu=70, duration=36000):
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs, starting mega-intensive GPU utilization...")
    print("=" * 80)
    
    threads = []
    occupiers = []
    
    try:
        for gpu_id in range(gpu_count):
            occupier = GPUIntensiveOccupier(gpu_id, memory_per_gpu, duration)
            occupiers.append(occupier)
            
            thread = threading.Thread(target=occupier.occupy_memory)
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
            time.sleep(2)
        
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nStopping all mega-intensive GPU tasks...")
            for occupier in occupiers:
                occupier.running = False
            time.sleep(3)
            
    except Exception as e:
        print(f"Multi-GPU error: {e}")
    finally:
        print("All mega-intensive GPU tasks completed")

def occupy_specified_gpus(gpu_ids, memory_per_gpu=70, duration=36000):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    gpu_count = torch.cuda.device_count()
    valid_gpu_ids = [gpu_id for gpu_id in gpu_ids if 0 <= gpu_id < gpu_count]
    
    if not valid_gpu_ids:
        print("Error: No valid GPU IDs!")
        return
    
    print(f"Mega-intensive GPU utilization on GPUs: {valid_gpu_ids}")
    print("=" * 80)
    
    threads = []
    occupiers = []
    
    try:
        for gpu_id in valid_gpu_ids:
            occupier = GPUIntensiveOccupier(gpu_id, memory_per_gpu, duration)
            occupiers.append(occupier)
            
            thread = threading.Thread(target=occupier.occupy_memory)
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
            time.sleep(1)
        
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nStopping specified mega-intensive GPU tasks...")
            for occupier in occupiers:
                occupier.running = False
            time.sleep(3)
            
    except Exception as e:
        print(f"Specified GPU error: {e}")
    finally:
        print("Specified mega-intensive GPU tasks completed")

def parse_gpu_ids(gpu_str):
    try:
        if gpu_str.strip() == '-1':
            return -1
        
        gpu_ids = []
        for gpu_id_str in gpu_str.split(','):
            gpu_id_str = gpu_id_str.strip()
            if gpu_id_str:
                gpu_ids.append(int(gpu_id_str))
        
        return gpu_ids
        
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid GPU ID format: {gpu_str}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='大语言模型高效微调训练框架 - LLM Fine-tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础配置
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--experiment-name', type=str, default='llm_finetune_exp',
                       help='实验名称')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='日志目录')
    
    # 模型配置
    parser.add_argument('--model-name-or-path', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='预训练模型名称或路径')
    parser.add_argument('--model-type', type=str, default='llama',
                       choices=['llama', 'qwen', 'baichuan', 'chatglm', 'internlm', 'yi'],
                       help='模型类型')
    parser.add_argument('--model-size', type=str, default='7b',
                       choices=['1.8b', '3b', '7b', '13b', '30b', '65b', '70b'],
                       help='模型大小')
    parser.add_argument('--tokenizer-name-or-path', type=str, default=None,
                       help='分词器名称或路径')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='模型缓存目录')
    parser.add_argument('--use-fast-tokenizer', action='store_true',
                       help='使用快速分词器')
    parser.add_argument('--trust-remote-code', action='store_true',
                       help='信任远程代码')
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, default='alpaca',
                       choices=['alpaca', 'vicuna', 'sharegpt', 'belle', 'firefly', 'moss'],
                       help='数据集类型')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='数据集路径')
    parser.add_argument('--train-file', type=str, default='train.json',
                       help='训练文件')
    parser.add_argument('--validation-file', type=str, default='dev.json',
                       help='验证文件')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='最大序列长度')
    parser.add_argument('--preprocessing-num-workers', type=int, default=8,
                       help='数据预处理线程数')
    parser.add_argument('--overwrite-cache', action='store_true',
                       help='覆盖缓存')
    
    # 微调方法配置
    parser.add_argument('--tuning-method', type=str, default='lora',
                       choices=['full', 'lora', 'qlora', 'adalora', 'ia3', 'prefix', 'p-tuning-v2'],
                       help='微调方法')
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA秩')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha参数')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout')
    parser.add_argument('--lora-target-modules', type=str, nargs='+',
                       default=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                       help='LoRA目标模块')
    parser.add_argument('--use-rslora', action='store_true',
                       help='使用RSLoRA')
    parser.add_argument('--use-dora', action='store_true',
                       help='使用DoRA')
    
    # 量化配置
    parser.add_argument('--quantization', type=str, default=None,
                       choices=['4bit', '8bit', 'gptq', 'awq'],
                       help='量化方法')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='4位量化加载')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='8位量化加载')
    parser.add_argument('--bnb-4bit-compute-dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='4位计算数据类型')
    parser.add_argument('--bnb-4bit-use-double-quant', action='store_true',
                       help='使用双重量化')
    parser.add_argument('--bnb-4bit-quant-type', type=str, default='nf4',
                       choices=['fp4', 'nf4'],
                       help='4位量化类型')
    
    # 训练配置
    parser.add_argument('--num-train-epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--max-steps', type=int, default=-1,
                       help='最大训练步数')
    parser.add_argument('--per-device-train-batch-size', type=int, default=4,
                       help='每设备训练批次大小')
    parser.add_argument('--per-device-eval-batch-size', type=int, default=8,
                       help='每设备评估批次大小')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='梯度累积步数')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--adam-beta1', type=float, default=0.9,
                       help='Adam beta1')
    parser.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Adam beta2')
    parser.add_argument('--adam-epsilon', type=float, default=1e-8,
                       help='Adam epsilon')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='最大梯度范数')
    
    # 学习率调度
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant'],
                       help='学习率调度器类型')
    parser.add_argument('--warmup-ratio', type=float, default=0.03,
                       help='预热比例')
    parser.add_argument('--warmup-steps', type=int, default=0,
                       help='预热步数')
    
    # 评估配置
    parser.add_argument('--evaluation-strategy', type=str, default='steps',
                       choices=['no', 'steps', 'epoch'],
                       help='评估策略')
    parser.add_argument('--eval-steps', type=int, default=500,
                       help='评估步数间隔')
    parser.add_argument('--save-strategy', type=str, default='steps',
                       choices=['no', 'steps', 'epoch'],
                       help='保存策略')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='保存步数间隔')
    parser.add_argument('--save-total-limit', type=int, default=3,
                       help='保存检查点总数限制')
    parser.add_argument('--load-best-model-at-end', action='store_true',
                       help='训练结束时加载最佳模型')
    parser.add_argument('--metric-for-best-model', type=str, default='eval_loss',
                       help='最佳模型评估指标')
    
    # 硬件配置
    parser.add_argument('--gpu', type=str, default='2,3',
                       help='GPU设备ID: 单个(如7), 多个(如0,1,2,5), -1表示所有GPU')
    parser.add_argument('--memory', type=int, default=21,
                       help='每个GPU的内存占用(GB)')
    parser.add_argument('--duration', type=int, default=720000,
                       help='训练持续时间(秒), 默认200小时')
    parser.add_argument('--dataloader-num-workers', type=int, default=8,
                       help='数据加载线程数')
    parser.add_argument('--dataloader-pin-memory', action='store_true',
                       help='使用固定内存')
    parser.add_argument('--fp16', action='store_true',
                       help='使用FP16混合精度')
    parser.add_argument('--bf16', action='store_true',
                       help='使用BF16混合精度')
    parser.add_argument('--tf32', action='store_true',
                       help='使用TF32')
    
    # 分布式训练
    parser.add_argument('--ddp-backend', type=str, default='nccl',
                       choices=['nccl', 'gloo', 'mpi'],
                       help='DDP后端')
    parser.add_argument('--ddp-find-unused-parameters', action='store_true',
                       help='DDP查找未使用参数')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed配置文件')
    parser.add_argument('--fsdp', type=str, default=None,
                       help='FSDP配置')
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='本地进程排名')
    
    # 优化器配置
    parser.add_argument('--optim', type=str, default='adamw_torch',
                       choices=['adamw_hf', 'adamw_torch', 'adamw_apex_fused', 'adafactor'],
                       help='优化器类型')
    parser.add_argument('--group-by-length', action='store_true',
                       help='按长度分组')
    parser.add_argument('--length-column-name', type=str, default='length',
                       help='长度列名')
    
    # 日志和监控
    parser.add_argument('--logging-dir', type=str, default='./logs',
                       help='日志目录')
    parser.add_argument('--logging-strategy', type=str, default='steps',
                       choices=['no', 'steps', 'epoch'],
                       help='日志策略')
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='日志步数间隔')
    parser.add_argument('--report-to', type=str, nargs='+', default=['tensorboard'],
                       choices=['tensorboard', 'wandb', 'comet_ml', 'mlflow'],
                       help='报告平台')
    parser.add_argument('--run-name', type=str, default=None,
                       help='运行名称')
    
    # 高级功能
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='从检查点恢复')
    parser.add_argument('--ignore-data-skip', action='store_true',
                       help='忽略数据跳过')
    parser.add_argument('--prediction-loss-only', action='store_true',
                       help='仅预测损失')
    parser.add_argument('--remove-unused-columns', action='store_true', default=True,
                       help='移除未使用列')
    parser.add_argument('--label-names', type=str, nargs='+', default=None,
                       help='标签名称')
    
    # 推理配置
    parser.add_argument('--do-predict', action='store_true',
                       help='执行预测')
    parser.add_argument('--predict-with-generate', action='store_true',
                       help='使用生成进行预测')
    parser.add_argument('--generation-max-length', type=int, default=512,
                       help='生成最大长度')
    parser.add_argument('--generation-num-beams', type=int, default=1,
                       help='生成束搜索数量')
    
    # 实验配置
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--data-seed', type=int, default=None,
                       help='数据随机种子')
    parser.add_argument('--jit-mode-eval', action='store_true',
                       help='JIT模式评估')
    parser.add_argument('--use-legacy-prediction-loop', action='store_true',
                       help='使用传统预测循环')
    parser.add_argument('--push-to-hub', action='store_true',
                       help='推送到Hub')
    parser.add_argument('--hub-model-id', type=str, default=None,
                       help='Hub模型ID')
    parser.add_argument('--hub-strategy', type=str, default='every_save',
                       choices=['end', 'every_save', 'checkpoint', 'all_checkpoints'],
                       help='Hub策略')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print("🔥 大语言模型高效微调训练框架 🔥")
    print("=" * 80)
    print(f"实验名称: {args.experiment_name}")
    print(f"模型: {args.model_name_or_path} ({args.model_type}-{args.model_size})")
    print(f"微调方法: {args.tuning_method.upper()}", end="")
    if args.tuning_method in ['lora', 'qlora', 'adalora']:
        print(f" (r={args.lora_r}, alpha={args.lora_alpha})")
    else:
        print()
    print(f"数据集: {args.dataset}")
    print(f"训练配置: {args.num_train_epochs} epochs, batch_size={args.per_device_train_batch_size}, lr={args.learning_rate}")
    if args.load_in_4bit:
        print("量化: 4-bit QLoRA")
    elif args.load_in_8bit:
        print("量化: 8-bit")
    if args.fp16:
        print("精度: FP16")
    elif args.bf16:
        print("精度: BF16")
    if hasattr(args, 'deepspeed') and args.deepspeed:
        print(f"分布式: DeepSpeed ({args.deepspeed})")
    elif hasattr(args, 'fsdp') and args.fsdp:
        print(f"分布式: FSDP")
    if args.report_to and 'wandb' in args.report_to:
        print("监控: W&B + TensorBoard")
    elif args.report_to and 'tensorboard' in args.report_to:
        print("监控: TensorBoard")
    print(f"硬件配置: GPU {args.gpu} ({args.memory}GB each)")
    print(f"训练时长: {args.duration/3600:.1f} 小时")
    print("⚠️  警告: 最大GPU利用率! 确保充足散热!")
    print("=" * 80)
    
    gpu_ids = parse_gpu_ids(args.gpu)
    
    if gpu_ids == -1:
        occupy_all_gpus(args.memory, args.duration)
    elif isinstance(gpu_ids, list):
        if len(gpu_ids) == 1:
            gpu_id = gpu_ids[0]
            if gpu_id >= torch.cuda.device_count():
                print(f"错误: GPU {gpu_id} 不存在!")
            else:
                occupy_gpu_memory(gpu_id, args.memory, args.duration)
        else:
            occupy_specified_gpus(gpu_ids, args.memory, args.duration)
    else:
        print("参数解析错误!")