#!/usr/bin/env python3
"""
混合采样策略可视化工具

用于分析和可视化混合采样策略的效果，包括：
1. 球形路径可视化
2. 相机轨迹分析
3. 视角覆盖度计算
4. 性能指标统计
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

class HybridSamplingVisualizer:
    """混合采样策略可视化器"""
    
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.camera_data = {}
        self.load_camera_data()
    
    def load_camera_data(self):
        """加载相机数据"""
        json_files = list(self.result_dir.glob("difix3d_camera_poses_step_*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                self.camera_data = json.load(f)
            print(f"✅ 加载相机数据: {latest_file}")
        else:
            print("⚠️ 未找到相机数据文件")
    
    def visualize_spherical_path(self, save_path: str = None):
        """可视化球形路径"""
        if not self.camera_data:
            print("❌ 没有相机数据可供可视化")
            return
        
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1: 训练相机分布
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_train_cameras(ax1)
        ax1.set_title('训练相机分布')
        
        # 子图2: DiFix3D虚拟相机分布
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_difix3d_cameras(ax2)
        ax2.set_title('DiFix3D虚拟相机分布')
        
        # 子图3: 混合分布
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_combined_cameras(ax3)
        ax3.set_title('混合相机分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可视化图像已保存: {save_path}")
        
        plt.show()
    
    def _plot_train_cameras(self, ax):
        """绘制训练相机"""
        if 'train_cameras' not in self.camera_data:
            return
        
        train_cameras = self.camera_data['train_cameras']
        positions = np.array([cam['position'] for cam in train_cameras])
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=50, alpha=0.7, label='训练相机')
        
        # 绘制相机朝向
        for i, cam in enumerate(train_cameras[:10]):  # 只显示前10个
            pos = np.array(cam['position'])
            rot_matrix = np.array(cam['rotation_matrix'])
            direction = -rot_matrix[:, 2]  # 相机朝向
            ax.quiver(pos[0], pos[1], pos[2], 
                     direction[0], direction[1], direction[2],
                     length=0.1, color='blue', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    def _plot_difix3d_cameras(self, ax):
        """绘制DiFix3D虚拟相机"""
        if 'difix3d_virtual_camera_batches' not in self.camera_data:
            return
        
        all_positions = []
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for batch_idx, batch in enumerate(self.camera_data['difix3d_virtual_camera_batches']):
            positions = np.array([cam['position'] for cam in batch['cameras']])
            all_positions.extend(positions)
            
            color = colors[batch_idx % len(colors)]
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c=color, s=30, alpha=0.6, 
                      label=f'批次 {batch_idx+1} ({len(positions)}个)')
        
        if all_positions:
            all_positions = np.array(all_positions)
            # 绘制球形路径（如果存在）
            self._draw_sphere_path(ax, all_positions)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    def _plot_combined_cameras(self, ax):
        """绘制混合相机分布"""
        # 训练相机
        if 'train_cameras' in self.camera_data:
            train_cameras = self.camera_data['train_cameras']
            train_positions = np.array([cam['position'] for cam in train_cameras])
            ax.scatter(train_positions[:, 0], train_positions[:, 1], train_positions[:, 2], 
                      c='blue', s=50, alpha=0.7, label='训练相机')
        
        # DiFix3D虚拟相机
        if 'difix3d_virtual_camera_batches' in self.camera_data:
            all_virtual_positions = []
            for batch in self.camera_data['difix3d_virtual_camera_batches']:
                positions = np.array([cam['position'] for cam in batch['cameras']])
                all_virtual_positions.extend(positions)
            
            if all_virtual_positions:
                all_virtual_positions = np.array(all_virtual_positions)
                ax.scatter(all_virtual_positions[:, 0], all_virtual_positions[:, 1], all_virtual_positions[:, 2], 
                          c='red', s=30, alpha=0.6, label='DiFix3D虚拟相机')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    def _draw_sphere_path(self, ax, positions: np.ndarray):
        """绘制球形路径"""
        if len(positions) < 3:
            return
        
        # 计算场景中心
        center = np.mean(positions, axis=0)
        
        # 计算球形半径
        distances = np.linalg.norm(positions - center, axis=1)
        radius = np.mean(distances)
        
        # 绘制球形路径
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')
        
        # 标记场景中心
        ax.scatter(center[0], center[1], center[2], c='black', s=100, marker='*', label='场景中心')
    
    def analyze_coverage(self) -> Dict:
        """分析视角覆盖度"""
        if not self.camera_data:
            return {}
        
        analysis = {
            'train_cameras': 0,
            'difix3d_batches': 0,
            'total_virtual_cameras': 0,
            'coverage_metrics': {}
        }
        
        # 统计相机数量
        if 'train_cameras' in self.camera_data:
            analysis['train_cameras'] = len(self.camera_data['train_cameras'])
        
        if 'difix3d_virtual_camera_batches' in self.camera_data:
            analysis['difix3d_batches'] = len(self.camera_data['difix3d_virtual_camera_batches'])
            total_virtual = sum(len(batch['cameras']) for batch in self.camera_data['difix3d_virtual_camera_batches'])
            analysis['total_virtual_cameras'] = total_virtual
        
        # 计算覆盖度指标
        all_positions = []
        
        # 添加训练相机位置
        if 'train_cameras' in self.camera_data:
            train_positions = np.array([cam['position'] for cam in self.camera_data['train_cameras']])
            all_positions.extend(train_positions)
        
        # 添加虚拟相机位置
        if 'difix3d_virtual_camera_batches' in self.camera_data:
            for batch in self.camera_data['difix3d_virtual_camera_batches']:
                positions = np.array([cam['position'] for cam in batch['cameras']])
                all_positions.extend(positions)
        
        if all_positions:
            all_positions = np.array(all_positions)
            analysis['coverage_metrics'] = self._calculate_coverage_metrics(all_positions)
        
        return analysis
    
    def _calculate_coverage_metrics(self, positions: np.ndarray) -> Dict:
        """计算覆盖度指标"""
        metrics = {}
        
        # 计算空间分布
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        metrics['spatial_center'] = center.tolist()
        metrics['mean_distance'] = float(np.mean(distances))
        metrics['std_distance'] = float(np.std(distances))
        metrics['min_distance'] = float(np.min(distances))
        metrics['max_distance'] = float(np.max(distances))
        
        # 计算角度分布（简化版本）
        if len(positions) > 1:
            # 计算相对于中心的角度分布
            relative_positions = positions - center
            angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
            metrics['angle_std'] = float(np.std(angles))
            metrics['angle_range'] = float(np.max(angles) - np.min(angles))
        
        return metrics
    
    def generate_report(self, save_path: str = None):
        """生成分析报告"""
        analysis = self.analyze_coverage()
        
        report = f"""
# 混合采样策略分析报告

## 相机统计
- 训练相机数量: {analysis.get('train_cameras', 0)}
- DiFix3D批次数量: {analysis.get('difix3d_batches', 0)}
- DiFix3D虚拟相机总数: {analysis.get('total_virtual_cameras', 0)}
- 相机总数: {analysis.get('train_cameras', 0) + analysis.get('total_virtual_cameras', 0)}

## 覆盖度指标
"""
        
        if 'coverage_metrics' in analysis and analysis['coverage_metrics']:
            metrics = analysis['coverage_metrics']
            report += f"""
- 空间中心: {metrics.get('spatial_center', 'N/A')}
- 平均距离: {metrics.get('mean_distance', 0):.3f}
- 距离标准差: {metrics.get('std_distance', 0):.3f}
- 最小距离: {metrics.get('min_distance', 0):.3f}
- 最大距离: {metrics.get('max_distance', 0):.3f}
- 角度标准差: {metrics.get('angle_std', 0):.3f}
- 角度范围: {metrics.get('angle_range', 0):.3f}
"""
        
        report += f"""
## 建议
- 如果距离标准差过大，考虑调整球形路径半径
- 如果角度范围过小，考虑增加球形路径点数
- 如果虚拟相机数量不足，考虑增加插帧数量

## 配置优化
基于当前分析，建议的配置参数：
- spherical_path_radius: {analysis.get('coverage_metrics', {}).get('mean_distance', 0.2) * 0.8:.3f}
- spherical_path_points: {max(20, analysis.get('total_virtual_cameras', 0) // 2)}
- camera_perturbation_translation: {analysis.get('coverage_metrics', {}).get('std_distance', 0.05) * 0.5:.3f}
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 分析报告已保存: {save_path}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='混合采样策略可视化工具')
    parser.add_argument('--result_dir', type=str, required=True, help='结果目录路径')
    parser.add_argument('--save_plot', type=str, help='保存可视化图像路径')
    parser.add_argument('--save_report', type=str, help='保存分析报告路径')
    parser.add_argument('--show_plot', action='store_true', help='显示可视化图像')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = HybridSamplingVisualizer(args.result_dir)
    
    # 生成可视化
    if args.show_plot or args.save_plot:
        visualizer.visualize_spherical_path(args.save_plot)
    
    # 生成分析报告
    visualizer.generate_report(args.save_report)

if __name__ == "__main__":
    main()

