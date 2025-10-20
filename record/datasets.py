"""
FPV无人机数据集加载器
作用：
    加载和处理LeRobot格式的Liftoff飞行数据集
主要功能：
    - FPVDataset: 读取parquet格式的episode数据
    - 自动加载图像和状态
    - 支持数据增强和归一化
依赖：
    torch, pandas, numpy, PIL, torchvision
注意：
    数据集结构必须符合LeRobot标准格式
示例:
    dataset = FPVDataset(
        dataset_root='/path/to/lerobot_datasets/liftoff_drone_dataset',
        image_size=224,
        augment=True
    )
    sample = dataset[0]  # {'image': Tensor[3,H,W], 'state': Tensor[13], 'action': Tensor[4]}
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FPVDataset(Dataset):
    """
    FPV无人机数据集

    数据格式：
    - 图像：observation.images.cam_front (224x224 RGB)
    - 状态：observation.state (13维) [vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, ax, ay, az]
    - 动作：action (4维) [throttle, yaw, pitch, roll]
    """

    def __init__(
        self,
        dataset_root: str,
        image_size: int = 224,
        augment: bool = False,
        normalize: bool = True,
        state_dim: int = 13,
        action_dim: int = 4,
        action_history_len: int = 4
    ):
        """
        初始化数据集

        Args:
            dataset_root: 数据集根目录
            image_size: 图像尺寸（正方形）
            augment: 是否使用数据增强
            normalize: 是否归一化图像
            state_dim: 状态维度
            action_dim: 动作维度
            action_history_len: 历史动作长度（包含当前帧之前的N帧动作）
        """
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_history_len = action_history_len

        # 加载元数据
        meta_dir = self.dataset_root / 'meta'
        info_file = meta_dir / 'info.json'
        if not info_file.exists():
            raise FileNotFoundError(f"Dataset info not found: {info_file}")

        with open(info_file, 'r') as f:
            self.info = json.load(f)

        # 加载所有episode
        self.episodes = []
        self.episode_lengths = []
        self._load_episodes()

        # 构建索引映射 (frame_idx → (episode_idx, local_frame_idx))
        self.frame_to_episode = []
        for ep_idx, ep_len in enumerate(self.episode_lengths):
            self.frame_to_episode.extend([(ep_idx, i) for i in range(ep_len)])

        # 数据变换
        self.transform = self._build_transform(augment, normalize)

        print(f"✓ Loaded FPVDataset from {self.dataset_root}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Total frames: {len(self.frame_to_episode)}")
        print(f"  Image size: {self.image_size}x{self.image_size}")
        print(f"  Augment: {augment}, Normalize: {normalize}")

    def _load_episodes(self):
        """加载所有episode数据"""
        # 检查chunk结构
        chunk_dirs = sorted(self.dataset_root.glob('chunk-*'))
        if chunk_dirs:
            # 新chunk结构
            for chunk_dir in chunk_dirs:
                data_path = chunk_dir / 'data'
                if data_path.exists():
                    episode_files = sorted(data_path.glob('episode_*.parquet'))
                    for ep_file in episode_files:
                        df = pd.read_parquet(ep_file)
                        self.episodes.append(df)
                        self.episode_lengths.append(len(df))
        else:
            # legacy flat结构
            data_path = self.dataset_root / 'data'
            if data_path.exists():
                episode_files = sorted(data_path.glob('episode_*.parquet'))
                for ep_file in episode_files:
                    df = pd.read_parquet(ep_file)
                    self.episodes.append(df)
                    self.episode_lengths.append(len(df))

        if not self.episodes:
            raise ValueError(f"No episodes found in {self.dataset_root}")

    def _build_transform(self, augment: bool, normalize: bool):
        """构建图像变换流程"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]

        if augment:
            transform_list.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            ])

        transform_list.append(transforms.ToTensor())

        if normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.frame_to_episode)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Args:
            idx: 全局帧索引

        Returns:
            {
                'image': Tensor[3, H, W],
                'state': Tensor[state_dim],
                'action': Tensor[action_dim],
                'action_history': Tensor[action_history_len, action_dim]  # 历史动作
            }
        """
        # 找到对应的episode和局部索引
        ep_idx, local_idx = self.frame_to_episode[idx]
        episode = self.episodes[ep_idx]
        row = episode.iloc[local_idx]

        # 加载图像
        image_path_str = row['observation.images.cam_front']
        # 处理路径（相对于dataset_root）
        if isinstance(image_path_str, str):
            image_path = self.dataset_root / 'videos' / image_path_str
        else:
            # 某些实现直接存储路径对象
            image_path = Path(image_path_str)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 加载状态 (13维)
        state = np.array(row['observation.state'], dtype=np.float32)
        if len(state) != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {len(state)}"
            )
        state = torch.from_numpy(state)

        # 加载当前动作 (4维: throttle, yaw, pitch, roll)
        action = np.array(row['action'], dtype=np.float32)
        if len(action) != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {len(action)}"
            )
        action = torch.from_numpy(action)

        # 加载历史动作 (action_history_len, 4)
        # 获取从 (local_idx - action_history_len) 到 (local_idx - 1) 的动作
        action_history = []
        for i in range(self.action_history_len):
            hist_idx = local_idx - self.action_history_len + i
            if hist_idx < 0:
                # Episode开始前，使用零动作填充
                hist_action = np.zeros(self.action_dim, dtype=np.float32)
            else:
                hist_action = np.array(episode.iloc[hist_idx]['action'], dtype=np.float32)
            action_history.append(hist_action)

        action_history = np.stack(action_history, axis=0)  # [action_history_len, action_dim]
        action_history = torch.from_numpy(action_history)

        return {
            'image': image,
            'state': state,
            'action': action,
            'action_history': action_history
        }

    def get_episode(self, episode_idx: int) -> pd.DataFrame:
        """获取完整episode数据"""
        return self.episodes[episode_idx]

    def get_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        计算数据集统计信息

        Returns:
            {
                'state': {'mean': [...], 'std': [...]},
                'action': {'mean': [...], 'std': [...]}
            }
        """
        # 收集所有状态和动作
        all_states = []
        all_actions = []

        for ep in self.episodes:
            all_states.extend(ep['observation.state'].tolist())
            all_actions.extend(ep['action'].tolist())

        all_states = np.array(all_states, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.float32)

        return {
            'state': {
                'mean': all_states.mean(axis=0),
                'std': all_states.std(axis=0)
            },
            'action': {
                'mean': all_actions.mean(axis=0),
                'std': all_actions.std(axis=0)
            }
        }


# 测试代码
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test FPVDataset')
    parser.add_argument('dataset_root', type=str, help='Path to dataset root')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    print("="*70)
    print("FPVDataset Test")
    print("="*70)
    print()

    # 加载数据集
    dataset = FPVDataset(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        augment=args.augment
    )

    print()
    print("="*70)
    print("Sample Data")
    print("="*70)

    # 显示第一个样本
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"State shape: {sample['state'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print()
    print(f"State: {sample['state']}")
    print(f"Action: {sample['action']}")

    print()
    print("="*70)
    print("Dataset Statistics")
    print("="*70)

    stats = dataset.get_stats()
    print("\nState statistics:")
    print(f"  Mean: {stats['state']['mean']}")
    print(f"  Std:  {stats['state']['std']}")

    print("\nAction statistics:")
    print(f"  Mean: {stats['action']['mean']}")
    print(f"  Std:  {stats['action']['std']}")

    print()
    print("="*70)
    print("✓ Test completed successfully!")
    print("="*70)
