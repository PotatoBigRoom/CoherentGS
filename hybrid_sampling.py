#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def se3_interpolate_midpoint(
    pose1: torch.Tensor,
    K1: torch.Tensor,
    pose2: torch.Tensor,
    K2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return se3_interpolate_to_target(pose1, K1, pose2, K2, t=0.5)


def se3_reverse_interpolate_from_midpoint(
    midpoint_pose: torch.Tensor,
    midpoint_K: torch.Tensor,
    end_pose: torch.Tensor,
    end_K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = midpoint_pose.device
    
    # Ensure inputs have correct shapes
    if midpoint_pose.dim() == 2:
        midpoint_pose = midpoint_pose.unsqueeze(0)  # [1, 4, 4]
    if end_pose.dim() == 2:
        end_pose = end_pose.unsqueeze(0)  # [1, 4, 4]
    if midpoint_K.dim() == 2:
        midpoint_K = midpoint_K.unsqueeze(0)  # [1, 3, 3]
    if end_K.dim() == 2:
        end_K = end_K.unsqueeze(0)  # [1, 3, 3]
    
    # Extract rotation matrix and translation vector
    midpoint_R = midpoint_pose[0, :3, :3]  # [3, 3]
    midpoint_t = midpoint_pose[0, :3, 3]   # [3]
    end_R = end_pose[0, :3, :3]           # [3, 3]
    end_t = end_pose[0, :3, 3]            # [3]
    
    # 1. Reverse-lerp translation: start_t = 2 * midpoint_t - end_t
    start_t = 2 * midpoint_t - end_t  # [3]
    
    # 2. Reverse SLERP for rotation using quaternions
    start_R = reverse_slerp_rotation(midpoint_R, end_R)  # [3, 3]
    
    # 3. Build recovered start pose matrix
    start_pose = torch.eye(4, device=device)
    start_pose[:3, :3] = start_R
    start_pose[:3, 3] = start_t
    
    # 4. Reverse-lerp intrinsics: start_K = 2 * midpoint_K - end_K
    start_K = 2 * midpoint_K[0] - end_K[0]  # [3, 3]
    
    return start_pose, start_K


def reverse_slerp_rotation(midpoint_R: torch.Tensor, end_R: torch.Tensor) -> torch.Tensor:
    """
    Reverse spherical linear interpolation (SLERP) for rotation matrices.
    Given midpoint and endpoint, recover the start rotation.
    
    Args:
        midpoint_R: midpoint rotation matrix [3, 3]
        end_R: end rotation matrix [3, 3]
        
    Returns:
        start_R: recovered start rotation matrix [3, 3]
    """
    device = midpoint_R.device
    
    # Convert rotation matrices to quaternions
    midpoint_q = rotation_matrix_to_quaternion(midpoint_R)  # [4]
    end_q = rotation_matrix_to_quaternion(end_R)            # [4]
    
    # Reverse quaternion SLERP
    start_q = reverse_slerp_quaternion(midpoint_q, end_q)  # [4]
    
    # Convert quaternion back to rotation matrix
    start_R = quaternion_to_rotation_matrix(start_q)  # [3, 3]
    
    return start_R


def reverse_slerp_quaternion(midpoint_q: torch.Tensor, end_q: torch.Tensor) -> torch.Tensor:
    """
    Reverse spherical linear interpolation (SLERP) for quaternions.
    Given midpoint and endpoint, recover the start quaternion.
    
    For SLERP: midpoint_q = slerp(start_q, end_q, 0.5).
    Reverse solution: start_q = slerp(midpoint_q, end_q, -1), then normalize.
    
    Args:
        midpoint_q: midpoint quaternion [4] (w, x, y, z)
        end_q: end quaternion [4] (w, x, y, z)
        
    Returns:
        start_q: recovered start quaternion [4] (w, x, y, z)
    """
    device = midpoint_q.device
    
    # Dot product
    dot = torch.dot(midpoint_q, end_q)
    
    # If dot < 0, flip one quaternion to choose the shorter path
    if dot < 0.0:
        end_q = -end_q
        dot = -dot
    
    # Angle
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    
    # Reverse interpolation: t = -1 (midpoint to the opposite direction)
    theta = -theta_0  # negative angle
    sin_theta = torch.sin(theta)
    
    # Reverse SLERP
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    start_q = s0 * midpoint_q + s1 * end_q
    
    # Normalize
    start_q = start_q / torch.norm(start_q)
    
    return start_q


def se3_interpolate_with_perturbation(
    start_pose: torch.Tensor,
    start_K: torch.Tensor,
    end_pose: torch.Tensor,
    end_K: torch.Tensor,
    num_samples: int,
    cfg=None,
    perturbation_std: float = 0.01
) -> Tuple[list, list]:
    """
    SE(3)插值并添加扰动（兼容maximum_circle_hybrid_sampling.py的接口）
    
    Args:
        start_pose: 起始pose [4, 4]
        start_K: 起始内参 [3, 3]
        end_pose: 结束pose [4, 4]
        end_K: 结束内参 [3, 3]
        num_samples: 采样数量
        cfg: 配置对象（可选）
        perturbation_std: 扰动标准差
        
    Returns:
        interpolated_poses: 插值poses列表
        interpolated_Ks: 插值内参列表
    """
    interpolated_poses = []
    interpolated_Ks = []
    
    for i in range(num_samples):
        # 计算插值参数
        if num_samples == 1:
            t = 0.5  # 单个样本时使用中点
        else:
            t = i / (num_samples - 1)  # 均匀分布
        
        # 执行插值
        interp_pose, interp_K = se3_interpolate_to_target(
            start_pose, start_K, end_pose, end_K, t
        )
        
        # 添加小的随机扰动（如果需要）
        if perturbation_std > 0:
            # 对位移添加高斯噪声
            translation_noise = torch.randn(3, device=interp_pose.device) * perturbation_std
            interp_pose[:3, 3] += translation_noise
            
            # 对旋转添加小的随机旋转
            rotation_noise = torch.randn(3, device=interp_pose.device) * perturbation_std * 0.1
            noise_so3 = pp.so3(rotation_noise).Exp()
            current_rotation = pp.SO3(interp_pose[:3, :3].unsqueeze(0))
            perturbed_rotation = current_rotation @ noise_so3
            interp_pose[:3, :3] = perturbed_rotation.matrix()[0]
        
        interpolated_poses.append(interp_pose)
        interpolated_Ks.append(interp_K)
    
    return interpolated_poses, interpolated_Ks


def se3_interpolate_to_target(
    source_pose: torch.Tensor,
    source_K: torch.Tensor,
    target_pose: torch.Tensor,
    target_K: torch.Tensor,
    t: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SE(3)插帧：从源相机pose插帧到目标pose的t位置
    使用纯PyTorch实现，不依赖pypose
    
    Args:
        source_pose: 源相机pose [4, 4] - 作为SE(3)起点
        source_K: 源相机内参 [3, 3]
        target_pose: 目标相机pose [4, 4] - 作为插帧目标点
        target_K: 目标相机内参 [3, 3]
        t: 插帧参数，范围[0, 1]，默认0.5表示中点
        
    Returns:
        interpolated_pose: 插帧后的pose [4, 4]
        interpolated_K: 插帧后的内参 [3, 3]
    """
    device = source_pose.device
    
    # 确保所有输入张量都在正确的设备上
    source_pose = source_pose.to(device)
    target_pose = target_pose.to(device)
    source_K = source_K.to(device)
    target_K = target_K.to(device)
    
    # 确保输入是正确的形状
    if source_pose.dim() == 2:
        source_pose = source_pose.unsqueeze(0)  # [1, 4, 4]
    if target_pose.dim() == 2:
        target_pose = target_pose.unsqueeze(0)  # [1, 4, 4]
    if source_K.dim() == 2:
        source_K = source_K.unsqueeze(0)  # [1, 3, 3]
    if target_K.dim() == 2:
        target_K = target_K.unsqueeze(0)  # [1, 3, 3]
    
    # 提取旋转矩阵和平移向量
    source_R = source_pose[0, :3, :3]  # [3, 3]
    source_t = source_pose[0, :3, 3]   # [3]
    target_R = target_pose[0, :3, :3]  # [3, 3]
    target_t = target_pose[0, :3, 3]   # [3]
    
    # 1. 平移向量线性插值
    interpolated_t = (1 - t) * source_t + t * target_t  # [3]
    
    # 2. 旋转矩阵球面线性插值 (SLERP)
    interpolated_R = slerp_rotation(source_R, target_R, t)  # [3, 3]
    
    # 3. 构建插值后的pose矩阵
    interpolated_pose = torch.eye(4, device=device)
    interpolated_pose[:3, :3] = interpolated_R
    interpolated_pose[:3, 3] = interpolated_t
    
    # 4. 内参线性插值
    interpolated_K = (1 - t) * source_K[0] + t * target_K[0]  # [3, 3]
    
    return interpolated_pose, interpolated_K


def slerp_rotation(R1: torch.Tensor, R2: torch.Tensor, t: float) -> torch.Tensor:
    """
    旋转矩阵的球面线性插值 (SLERP)
    
    Args:
        R1: 第一个旋转矩阵 [3, 3]
        R2: 第二个旋转矩阵 [3, 3]
        t: 插值参数 [0, 1]
        
    Returns:
        interpolated_R: 插值后的旋转矩阵 [3, 3]
    """
    device = R1.device
    
    # 将旋转矩阵转换为四元数
    q1 = rotation_matrix_to_quaternion(R1)  # [4]
    q2 = rotation_matrix_to_quaternion(R2)  # [4]
    
    # 四元数球面线性插值
    q_interp = slerp_quaternion(q1, q2, t)  # [4]
    
    # 将四元数转换回旋转矩阵
    interpolated_R = quaternion_to_rotation_matrix(q_interp)  # [3, 3]
    
    return interpolated_R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    
    Args:
        R: 旋转矩阵 [3, 3]
        
    Returns:
        q: 四元数 [4] (w, x, y, z)
    """
    device = R.device
    
    # 确保输入是3x3矩阵
    if R.shape != (3, 3):
        raise ValueError(f"旋转矩阵形状必须是(3, 3)，得到{R.shape}")
    
    # 计算四元数分量
    trace = torch.trace(R)
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    q = torch.stack([qw, qx, qy, qz], dim=0)  # [4] (w, x, y, z)
    
    # 归一化四元数
    q = q / torch.norm(q)
    
    return q


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: 四元数 [4] (w, x, y, z)
        
    Returns:
        R: 旋转矩阵 [3, 3]
    """
    device = q.device
    
    # 确保输入是4维向量
    if q.shape != (4,):
        raise ValueError(f"四元数形状必须是(4,)，得到{q.shape}")
    
    # 归一化四元数
    q = q / torch.norm(q)
    
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # 计算旋转矩阵
    R = torch.zeros(3, 3, device=device)
    
    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - w*z)
    R[0, 2] = 2 * (x*z + w*y)
    
    R[1, 0] = 2 * (x*y + w*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - w*x)
    
    R[2, 0] = 2 * (x*z - w*y)
    R[2, 1] = 2 * (y*z + w*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    
    return R


def slerp_quaternion(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """
    四元数球面线性插值
    
    Args:
        q1: 第一个四元数 [4] (w, x, y, z)
        q2: 第二个四元数 [4] (w, x, y, z)
        t: 插值参数 [0, 1]
        
    Returns:
        q_interp: 插值后的四元数 [4] (w, x, y, z)
    """
    device = q1.device
    
    # 计算点积
    dot = torch.dot(q1, q2)
    
    # 如果点积为负，取反其中一个四元数以选择较短的路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    if dot > 0.9995:
        q_interp = (1 - t) * q1 + t * q2
        return q_interp / torch.norm(q_interp)
    
    # 计算角度
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    # 球面线性插值
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    q_interp = s0 * q1 + s1 * q2
    
    return q_interp


# 测试函数
def test_hybrid_sampling():
    """测试SE(3)插帧功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    start_pose = torch.eye(4, device=device)
    start_pose[:3, 3] = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
    
    end_pose = torch.eye(4, device=device)
    end_pose[:3, 3] = torch.tensor([2, 2, 2], device=device, dtype=torch.float32)
    
    K1 = torch.eye(3, device=device) * 500
    K1[2, 2] = 1
    K2 = K1.clone()
    
    print("🧪 测试SE(3)插帧功能")
    
    # 测试1: 正向插值 - 计算中点
    print("\n📐 测试1: 正向插值 - 计算两个pose的中点")
    midpoint_pose, midpoint_K = se3_interpolate_midpoint(start_pose, K1, end_pose, K2)
    print(f"起点位移: {start_pose[:3, 3]}")
    print(f"终点位移: {end_pose[:3, 3]}")
    print(f"计算出的中点位移: {midpoint_pose[:3, 3]}")
    print(f"预期中点位移: [1.0, 1.0, 1.0]")
    
    # 测试2: 不同插值参数
    print("\n📐 测试2: 不同插值参数")
    for t in [0.25, 0.5, 0.75]:
        interp_pose, interp_K = se3_interpolate_to_target(start_pose, K1, end_pose, K2, t)
        expected_t = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
        actual_t = interp_pose[:3, 3]
        error = torch.norm(expected_t - actual_t).item()
        print(f"  t={t}: 预期位移={expected_t}, 实际位移={actual_t}, 误差={error:.6f}")
    
    # 测试3: 旋转插值
    print("\n📐 测试3: 旋转插值测试")
    # 创建有旋转的测试数据
    angle = torch.pi / 4  # 45度
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    
    rotated_pose = torch.eye(4, device=device)
    rotated_pose[0, 0] = cos_a
    rotated_pose[0, 1] = -sin_a
    rotated_pose[1, 0] = sin_a
    rotated_pose[1, 1] = cos_a
    rotated_pose[:3, 3] = torch.tensor([1, 1, 0], device=device)
    
    interp_pose, _ = se3_interpolate_to_target(start_pose, K1, rotated_pose, K2, t=0.5)
    print(f"起点旋转矩阵:\n{start_pose[:3, :3]}")
    print(f"终点旋转矩阵:\n{rotated_pose[:3, :3]}")
    print(f"插值旋转矩阵:\n{interp_pose[:3, :3]}")
    
    # 测试4: 反向插值测试
    print("\n📐 测试4: 反向插值测试")
    # 使用前面的插值结果作为中点
    midpoint_pose = interp_pose
    midpoint_K = K1  # 使用相同的K
    
    # 反向插值：从中点和终点反推起点
    reconstructed_start_pose, reconstructed_start_K = se3_reverse_interpolate_from_midpoint(
        midpoint_pose, midpoint_K, rotated_pose, K2
    )
    
    print(f"原始起点位移: {start_pose[:3, 3]}")
    print(f"反推出的起点位移: {reconstructed_start_pose[:3, 3]}")
    translation_error = torch.norm(start_pose[:3, 3] - reconstructed_start_pose[:3, 3]).item()
    print(f"位移误差: {translation_error:.6f}")
    
    print(f"原始起点旋转矩阵:\n{start_pose[:3, :3]}")
    print(f"反推出的起点旋转矩阵:\n{reconstructed_start_pose[:3, :3]}")
    rotation_error = torch.norm(start_pose[:3, :3] - reconstructed_start_pose[:3, :3]).item()
    print(f"旋转误差: {rotation_error:.6f}")
    
    # 测试5: 验证一致性
    print("\n📐 测试5: 验证一致性 - 用反推的起点重新计算中点")
    verify_midpoint_pose, verify_midpoint_K = se3_interpolate_midpoint(
        reconstructed_start_pose, reconstructed_start_K, rotated_pose, K2
    )
    print(f"原始中点位移: {midpoint_pose[:3, 3]}")
    print(f"验证中点位移: {verify_midpoint_pose[:3, 3]}")
    midpoint_error = torch.norm(midpoint_pose[:3, 3] - verify_midpoint_pose[:3, 3]).item()
    print(f"中点位移误差: {midpoint_error:.6f}")
    
    print("\n✅ 测试完成")


def generate_camera_trajectory(
    train_poses: torch.Tensor,
    train_Ks: torch.Tensor,
    num_poses: int,
    cfg=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于训练视角生成相机轨迹，对所有训练视角进行插帧
    使用纯PyTorch实现，不依赖pypose
    
    Args:
        train_poses: 训练视角poses [N, 4, 4]
        train_Ks: 训练视角内参 [N, 3, 3]
        num_poses: 要生成的轨迹pose数量
        cfg: 配置对象（可选）
        
    Returns:
        trajectory_poses: 生成的轨迹poses [num_poses, 4, 4]
        trajectory_Ks: 生成的轨迹内参 [num_poses, 3, 3]
    """
    print(f"🔧 开始生成相机轨迹，训练视角数量: {len(train_poses)}")
    
    if len(train_poses) < 2:
        print("❌ 训练视角不足，需要至少2个视角进行插值")
        return None, None
    
    device = train_poses.device
    trajectory_poses = []
    trajectory_Ks = []
    
    # 第一步：对所有相邻的训练视角进行插帧
    print("📐 第一步：对相邻训练视角进行插帧")
    for i in range(len(train_poses) - 1):
        pose1 = train_poses[i]      # [4, 4]
        pose2 = train_poses[i + 1]  # [4, 4]
        K1 = train_Ks[i]          # [3, 3]
        K2 = train_Ks[i + 1]      # [3, 3]
        
        # 正向插帧：计算中点pose
        midpoint_pose, midpoint_K = se3_interpolate_midpoint(pose1, K1, pose2, K2)
        
        # 添加到轨迹中：起点 -> 中点 -> 终点
        if i == 0:  # 第一对，添加起点
            trajectory_poses.append(pose1)
            trajectory_Ks.append(K1)
        
        trajectory_poses.append(midpoint_pose)
        trajectory_Ks.append(midpoint_K)
        trajectory_poses.append(pose2)
        trajectory_Ks.append(K2)
    
    print(f"   第一步完成，生成了 {len(trajectory_poses)} 个轨迹点")
    trajectory_poses = torch.stack(trajectory_poses)  # [num_poses, 4, 4]
    trajectory_Ks = torch.stack(trajectory_Ks)        # [num_poses, 3, 3]
    
    print(f"✅ 轨迹生成完成: {trajectory_poses.shape}")
    return trajectory_poses, trajectory_Ks


if __name__ == "__main__":
    test_hybrid_sampling()
