"""
录制模块
作用：
    提供数据集录制、加载和预处理功能
主要组件：
    - FPVDataset: LeRobot数据集加载器
    - liftoff_capture: Liftoff屏幕捕获与数据录制工具
依赖：
    torch, pandas, numpy, PIL
注意：
    数据集格式遵循LeRobot标准
示例：
    from record.datasets import FPVDataset
    dataset = FPVDataset('/path/to/lerobot_datasets/liftoff_drone_dataset')
"""

from record.datasets import FPVDataset

__all__ = ['FPVDataset']
