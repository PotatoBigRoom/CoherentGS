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
    interpolated_poses = []
    interpolated_Ks = []
    
    for i in range(num_samples):
        # Compute interpolation parameter
        if num_samples == 1:
            t = 0.5  # use midpoint when there is a single sample
        else:
            t = i / (num_samples - 1)  # uniformly distributed
        
        # Perform interpolation
        interp_pose, interp_K = se3_interpolate_to_target(
            start_pose, start_K, end_pose, end_K, t
        )
        
        # Add small random perturbations (if needed)
        if perturbation_std > 0:
            # Add Gaussian noise to translation
            translation_noise = torch.randn(3, device=interp_pose.device) * perturbation_std
            interp_pose[:3, 3] += translation_noise
            
            # Add small random rotation to orientation
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
    SE(3) interpolation: interpolate from the source camera pose to the target pose at parameter t.
    Implemented purely in PyTorch, without pypose dependency.

    Args:
        source_pose: source camera pose [4, 4] — SE(3) start
        source_K: source camera intrinsics [3, 3]
        target_pose: target camera pose [4, 4] — interpolation endpoint
        target_K: target camera intrinsics [3, 3]
        t: interpolation parameter in [0, 1], default 0.5 for midpoint

    Returns:
        interpolated_pose: interpolated pose [4, 4]
        interpolated_K: interpolated intrinsics [3, 3]
    """
    device = source_pose.device
    
    # Ensure all input tensors are on the correct device
    source_pose = source_pose.to(device)
    target_pose = target_pose.to(device)
    source_K = source_K.to(device)
    target_K = target_K.to(device)
    
    # Ensure inputs have correct shapes
    if source_pose.dim() == 2:
        source_pose = source_pose.unsqueeze(0)  # [1, 4, 4]
    if target_pose.dim() == 2:
        target_pose = target_pose.unsqueeze(0)  # [1, 4, 4]
    if source_K.dim() == 2:
        source_K = source_K.unsqueeze(0)  # [1, 3, 3]
    if target_K.dim() == 2:
        target_K = target_K.unsqueeze(0)  # [1, 3, 3]
    
    # Extract rotation matrices and translation vectors
    source_R = source_pose[0, :3, :3]  # [3, 3]
    source_t = source_pose[0, :3, 3]   # [3]
    target_R = target_pose[0, :3, :3]  # [3, 3]
    target_t = target_pose[0, :3, 3]   # [3]
    
    # 1. Linear interpolation for translation vector
    interpolated_t = (1 - t) * source_t + t * target_t  # [3]
    
    # 2. Spherical linear interpolation (SLERP) for rotation matrices
    interpolated_R = slerp_rotation(source_R, target_R, t)  # [3, 3]
    
    # 3. Construct the interpolated pose matrix
    interpolated_pose = torch.eye(4, device=device)
    interpolated_pose[:3, :3] = interpolated_R
    interpolated_pose[:3, 3] = interpolated_t
    
    # 4. Linear interpolation for intrinsics
    interpolated_K = (1 - t) * source_K[0] + t * target_K[0]  # [3, 3]
    
    return interpolated_pose, interpolated_K


def slerp_rotation(R1: torch.Tensor, R2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) for rotation matrices.

    Args:
        R1: first rotation matrix [3, 3]
        R2: second rotation matrix [3, 3]
        t: interpolation parameter [0, 1]

    Returns:
        interpolated_R: interpolated rotation matrix [3, 3]
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
    Convert rotation matrix to quaternion (w, x, y, z).

    Args:
        R: rotation matrix [3, 3]

    Returns:
        q: quaternion [4] (w, x, y, z)
    """
    device = R.device
    
    # Ensure input is a 3x3 matrix
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be shape (3, 3), got {R.shape}")
    
    # Compute quaternion components
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
    
    # Normalize quaternion
    q = q / torch.norm(q)
    
    return q


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: quaternion [4] (w, x, y, z)

    Returns:
        R: rotation matrix [3, 3]
    """
    device = q.device
    
    # Ensure input is a 4D vector
    if q.shape != (4,):
        raise ValueError(f"Quaternion must be shape (4,), got {q.shape}")
    
    # Normalize quaternion
    q = q / torch.norm(q)
    
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Compute rotation matrix
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
    Spherical linear interpolation for quaternions.

    Args:
        q1: first quaternion [4] (w, x, y, z)
        q2: second quaternion [4] (w, x, y, z)
        t: interpolation parameter [0, 1]

    Returns:
        q_interp: interpolated quaternion [4] (w, x, y, z)
    """
    device = q1.device
    
    # Compute dot product
    dot = torch.dot(q1, q2)
    
    # If dot < 0, flip one quaternion to choose the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        q_interp = (1 - t) * q1 + t * q2
        return q_interp / torch.norm(q_interp)
    
    # Compute angle
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    # Spherical linear interpolation
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    q_interp = s0 * q1 + s1 * q2
    
    return q_interp


# Test functions
def test_hybrid_sampling():
    """Test SE(3) interpolation functionality"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    start_pose = torch.eye(4, device=device)
    start_pose[:3, 3] = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
    
    end_pose = torch.eye(4, device=device)
    end_pose[:3, 3] = torch.tensor([2, 2, 2], device=device, dtype=torch.float32)
    
    K1 = torch.eye(3, device=device) * 500
    K1[2, 2] = 1
    K2 = K1.clone()
    
    print(" Test SE(3) interpolation functionality")
    
    # Test 1: forward interpolation — compute midpoint
    print("\n Test 1: Forward interpolation — compute midpoint of two poses")
    midpoint_pose, midpoint_K = se3_interpolate_midpoint(start_pose, K1, end_pose, K2)
    print(f"Start translation: {start_pose[:3, 3]}")
    print(f"End translation: {end_pose[:3, 3]}")
    print(f"Computed midpoint translation: {midpoint_pose[:3, 3]}")
    print(f"Expected midpoint translation: [1.0, 1.0, 1.0]")
    
    # Test 2: different interpolation parameters
    print("\n Test 2: Different interpolation parameters")
    for t in [0.25, 0.5, 0.75]:
        interp_pose, interp_K = se3_interpolate_to_target(start_pose, K1, end_pose, K2, t)
        expected_t = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
        actual_t = interp_pose[:3, 3]
        error = torch.norm(expected_t - actual_t).item()
        print(f"  t={t}: expected translation={expected_t}, actual translation={actual_t}, error={error:.6f}")
    
    # Test 3: rotation interpolation
    print("\n Test 3: Rotation interpolation test")
    # Create test data with rotation
    angle = torch.pi / 4  # 45度
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    
    rotated_pose = torch.eye(4, device=device)
    rotated_pose[0, 0] = cos_a
    rotated_pose[0, 1] = -sin_a
    rotated_pose[1, 0] = sin_a
    rotated_pose[1, 1] = cos_a
    rotated_pose[:3, 3] = torch.tensor([1, 1, 0], device=device)
    
    interp_pose, _ = se3_interpolate_to_target(start_pose, K1, rotated_pose, K2, t=0.5)
    print(f"Start rotation matrix:\n{start_pose[:3, :3]}")
    print(f"End rotation matrix:\n{rotated_pose[:3, :3]}")
    print(f"Interpolated rotation matrix:\n{interp_pose[:3, :3]}")
    
    # Test 4: reverse interpolation
    print("\n Test 4: Reverse interpolation test")
    # Use the previous interpolation result as midpoint
    midpoint_pose = interp_pose
    midpoint_K = K1  # Use the same K
    
    # Reverse interpolation: recover start from midpoint and end
    reconstructed_start_pose, reconstructed_start_K = se3_reverse_interpolate_from_midpoint(
        midpoint_pose, midpoint_K, rotated_pose, K2
    )
    
    print(f"Original start translation: {start_pose[:3, 3]}")
    print(f"Recovered start translation: {reconstructed_start_pose[:3, 3]}")
    translation_error = torch.norm(start_pose[:3, 3] - reconstructed_start_pose[:3, 3]).item()
    print(f"Translation error: {translation_error:.6f}")
    
    print(f"Original start rotation matrix:\n{start_pose[:3, :3]}")
    print(f"Recovered start rotation matrix:\n{reconstructed_start_pose[:3, :3]}")
    rotation_error = torch.norm(start_pose[:3, :3] - reconstructed_start_pose[:3, :3]).item()
    print(f"Rotation error: {rotation_error:.6f}")
    
    # Test 5: consistency verification
    print("\n Test 5: Consistency check — recompute midpoint from recovered start")
    verify_midpoint_pose, verify_midpoint_K = se3_interpolate_midpoint(
        reconstructed_start_pose, reconstructed_start_K, rotated_pose, K2
    )
    print(f"Original midpoint translation: {midpoint_pose[:3, 3]}")
    print(f"Verified midpoint translation: {verify_midpoint_pose[:3, 3]}")
    midpoint_error = torch.norm(midpoint_pose[:3, 3] - verify_midpoint_pose[:3, 3]).item()
    print(f"Midpoint translation error: {midpoint_error:.6f}")
    
    print("\n Test completed")


def generate_camera_trajectory(
    train_poses: torch.Tensor,
    train_Ks: torch.Tensor,
    num_poses: int,
    cfg=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a camera trajectory from training views by interpolating all adjacent views.
    Implemented purely in PyTorch, without pypose dependency.

    Args:
        train_poses: training view poses [N, 4, 4]
        train_Ks: training view intrinsics [N, 3, 3]
        num_poses: number of trajectory poses to generate
        cfg: optional configuration object

    Returns:
        trajectory_poses: generated trajectory poses [num_poses, 4, 4]
        trajectory_Ks: generated trajectory intrinsics [num_poses, 3, 3]
    """
    print(f" Start generating camera trajectory, number of training views: {len(train_poses)}")
    
    if len(train_poses) < 2:
        print(" Insufficient training views; need at least 2 views for interpolation")
        return None, None
    
    device = train_poses.device
    trajectory_poses = []
    trajectory_Ks = []
    
    # Step 1: interpolate across adjacent training views
    print(" Step 1: Interpolate adjacent training views")
    for i in range(len(train_poses) - 1):
        pose1 = train_poses[i]      # [4, 4]
        pose2 = train_poses[i + 1]  # [4, 4]
        K1 = train_Ks[i]          # [3, 3]
        K2 = train_Ks[i + 1]      # [3, 3]
        
        # Forward interpolation: compute midpoint pose
        midpoint_pose, midpoint_K = se3_interpolate_midpoint(pose1, K1, pose2, K2)
        
        # Add to trajectory: start -> midpoint -> end
        if i == 0:  # first pair, add the start
            trajectory_poses.append(pose1)
            trajectory_Ks.append(K1)
        
        trajectory_poses.append(midpoint_pose)
        trajectory_Ks.append(midpoint_K)
        trajectory_poses.append(pose2)
        trajectory_Ks.append(K2)
    
    print(f"   Step 1 complete, generated {len(trajectory_poses)} trajectory points")
    trajectory_poses = torch.stack(trajectory_poses)  # [num_poses, 4, 4]
    trajectory_Ks = torch.stack(trajectory_Ks)        # [num_poses, 3, 3]
    
    print(f" Trajectory generation complete: {trajectory_poses.shape}")
    return trajectory_poses, trajectory_Ks


if __name__ == "__main__":
    test_hybrid_sampling()
